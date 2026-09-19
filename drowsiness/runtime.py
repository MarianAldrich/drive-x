"""Local single-driver inference with explicit unavailable and warmup states."""
from collections import deque
import math
from pathlib import Path
import threading
import time

import cv2
from PIL import Image
import torch
from torch import nn
from torchvision import models

from .model import VideoMobileNet, TASKS, image_transform
from .vision import CROP_VERSION, FaceLocator, crop

YAWN_LIVE_WINDOW = 8
EYE_LIVE_WINDOW = 8
YAWN_THRESHOLD = .45
YAWN_CONFIRM_SAMPLES = 2


class AlertState:
    def __init__(self, threshold=.65, hold_seconds=2., maximum_gap=3.):
        self.threshold, self.hold_seconds, self.maximum_gap = threshold, hold_seconds, maximum_gap
        self.reset()

    def reset(self):
        self.since, self.last = None, None

    def update(self, probability, now):
        if self.last is not None and (now <= self.last or now - self.last > self.maximum_gap):
            self.reset()
        self.last = now
        if probability is None or probability < self.threshold:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.hold_seconds


class YawnState:
    """Require consecutive sampled bags so one weak-label spike is not an alert."""
    def __init__(self, threshold=YAWN_THRESHOLD, required=YAWN_CONFIRM_SAMPLES):
        self.threshold, self.required = threshold, required
        self.reset()

    def reset(self):
        self.streak = 0
        self.confirmed = False

    def update(self, probability, new_sample=True):
        if new_sample:
            self.streak = self.streak + 1 if probability is not None and probability >= self.threshold else 0
            self.confirmed = self.streak >= self.required
        return self.confirmed


class HeadMovementTracker:
    """Track person-relative head changes from YuNet's five facial landmarks."""
    def __init__(self, calibration_samples=10):
        self.calibration_samples = calibration_samples
        self.reset()

    def reset(self):
        self.calibration = []
        self.neutral = None
        self.history = deque(maxlen=6)

    @staticmethod
    def features(landmarks):
        right_eye, left_eye, nose, right_mouth, left_mouth = landmarks
        eye_mid = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2)
        mouth_mid = ((right_mouth[0] + left_mouth[0]) / 2, (right_mouth[1] + left_mouth[1]) / 2)
        eye_distance, face_height = math.dist(right_eye, left_eye), math.dist(eye_mid, mouth_mid)
        if eye_distance < 4 or face_height < 4:
            raise ValueError('Facial landmarks are too close for head tracking.')
        return ((nose[0] - eye_mid[0]) / eye_distance,
                (nose[1] - eye_mid[1]) / face_height,
                math.degrees(math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))

    def update(self, landmarks):
        current = self.features(landmarks)
        if self.neutral is None:
            self.calibration.append(current)
            if len(self.calibration) >= self.calibration_samples:
                self.neutral = tuple(sorted(values)[len(values) // 2]
                                     for values in zip(*self.calibration))
            return dict(calibrated=self.neutral is not None, cue=False, status='calibrating',
                        calibration_progress=round(len(self.calibration) / self.calibration_samples * 100),
                        yaw=0., pitch=0., roll=0.)
        yaw, pitch = current[0] - self.neutral[0], current[1] - self.neutral[1]
        roll = (current[2] - self.neutral[2] + 180) % 360 - 180
        deviated = abs(yaw) >= .18 or abs(pitch) >= .15 or abs(roll) >= 12
        self.history.append(deviated)
        cue = len(self.history) == self.history.maxlen and sum(self.history) >= 4
        status = ('turned sideways' if abs(yaw) >= .18 else
                  'vertical movement' if abs(pitch) >= .15 else
                  'head tilted' if abs(roll) >= 12 else 'stable')
        return dict(calibrated=True, cue=cue, status=status,
                    calibration_progress=100, yaw=round(yaw, 3),
                    pitch=round(pitch, 3), roll=round(roll, 1))


class InferenceEngine:
    def __init__(self, detector_path, model_root):
        self.lock = threading.RLock()
        self.locator = FaceLocator(detector_path)
        self.model_root = Path(model_root)
        self.transform = image_transform()
        self.alert = AlertState()
        self.yawn_alert = YawnState()
        self.head = HeadMovementTracker()
        self.models, self.errors, self.buffers = {}, {}, {}
        self.session = None
        self.reload()

    def reload(self):
        with self.lock:
            self.models, self.errors = {}, {}
            for task in ('eye_state', 'yawn'):
                path = self.model_root / task / 'best.pt'
                if not path.is_file():
                    self.errors[task] = f'Missing {task}/best.pt'
                    continue
                try:
                    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                    config = checkpoint['config']
                    if task == 'eye_state':
                        if (config['architecture'] != 'mobilenet_v3_small' or config['task'] != 'eye_state'
                                or config['class_to_idx'] != {'closed': 0, 'open': 1}):
                            raise ValueError('Eye-state checkpoint is incompatible with this interface.')
                        model = models.mobilenet_v3_small(weights=None)
                        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
                    else:
                        if (config['architecture'] != 'mobilenet_v3_small_video' or config['task'] != task
                                or config['classes'] != TASKS[task] or config['crop_version'] != CROP_VERSION):
                            raise ValueError('Yawn checkpoint is incompatible with this interface.')
                        model = VideoMobileNet(task, top_fraction=config['top_fraction'])
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    self.models[task] = dict(model=model, config=config, epoch=checkpoint['epoch'])
                except (KeyError, ValueError, RuntimeError, OSError) as exc:
                    self.errors[task] = str(exc)
            self.reset()
            return self.status()

    def reset(self):
        with self.lock:
            self.buffers = {task: deque(maxlen=(EYE_LIVE_WINDOW if task == 'eye_state' else item['config']['bag_size']))
                            for task, item in self.models.items()}
            self.last_frame = self.last_sample = None
            self.last_box = None
            self.face_misses = 0
            self.alert.reset()
            self.yawn_alert.reset()
            self.head.reset()

    def status(self):
        return dict(models={task: dict(loaded=task in self.models,
                                      error=self.errors.get(task),
                                      epoch=self.models[task]['epoch'] if task in self.models else None)
                            for task in ('eye_state', 'yawn')}, detector=True)

    @staticmethod
    def overlap(a, b):
        x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        intersection = x * y
        return intersection / max(1, (a[2] - a[0]) * (a[3] - a[1]) +
                                  (b[2] - b[0]) * (b[3] - b[1]) - intersection)

    def detect(self, frame, session='default', now=None):
        with self.lock, torch.inference_mode():
            now = time.monotonic() if now is None else now
            begin = time.perf_counter()
            if session != self.session or (self.last_frame is not None and now - self.last_frame > 3.):
                self.reset()
            self.session = session
            self.last_frame = now
            faces = self.locator.locate(frame)
            output = dict(**self.status(), state='no_face', message='Position your face in the camera',
                          face_count=len(faces), boxes=None, predictions={}, alert=False,
                          sampled_this_request=False, instant_eye_closed=0., yawn_detected=False,
                          head=dict(calibrated=False, cue=False, status='unavailable',
                                    calibration_progress=0, yaw=0., pitch=0., roll=0.),
                          sampled_frames=min((len(buffer) for buffer in self.buffers.values()), default=0),
                          required_frames=max((EYE_LIVE_WINDOW if task == 'eye_state' else m['config']['bag_size']
                                                                 for task, m in self.models.items()), default=0))
            if len(faces) != 1:
                # Do not erase a 16-second window for one transient detector miss.
                # Three misses at the 2 Hz UI rate still reset state promptly.
                self.face_misses += 1
                if faces or self.face_misses >= 3:
                    self.reset()
                    output['sampled_frames'] = 0
                output.update(state='multiple_faces' if faces else 'no_face',
                              message='Keep only the driver in view' if faces else 'No face detected')
            else:
                self.face_misses = 0
                boxes = faces[0]
                if self.last_box and self.overlap(self.last_box, boxes['face']) < .25:
                    self.reset()
                self.last_box = boxes['face']
                output['boxes'] = boxes
                try:
                    output['head'] = self.head.update(boxes['landmarks'])
                except ValueError:
                    self.head.reset()
                if not self.models:
                    output.update(state='model_missing', message='Camera ready — add trained checkpoints')
                else:
                    if self.last_sample is None or now - self.last_sample >= 1.:
                        output['sampled_this_request'] = True
                        for task, item in self.models.items():
                            if task == 'eye_state':
                                tensors = []
                                for eye_box in boxes['eyes']:
                                    roi = crop(frame, eye_box)
                                    image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                                    tensors.append(self.transform(image))
                                probabilities = item['model'](torch.stack(tensors)).softmax(-1).mean(0)
                                output['instant_eye_closed'] = float(probabilities[0])
                                self.buffers[task].append(output['instant_eye_closed'])
                            else:
                                roi = crop(frame, boxes['mouth'])
                                image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                                logits = item['model'].encoder(self.transform(image).unsqueeze(0))[0]
                                self.buffers[task].append(logits)
                        self.last_sample = now
                    for task, buffer in self.buffers.items():
                        if task == 'yawn' and len(buffer) >= YAWN_LIVE_WINDOW:
                            recent = list(buffer)[-YAWN_LIVE_WINDOW:]
                            pooled = self.models[task]['model'].pool(torch.stack(recent).unsqueeze(0))
                            probs = pooled.softmax(-1)[0].tolist()
                            output['predictions'][task] = dict(zip(TASKS[task], probs))
                    output['sampled_frames'] = min(map(len, self.buffers.values())) if self.buffers else 0
                    yawn = output['predictions'].get('yawn')
                    probability = yawn['yawn'] if yawn else None
                    output['yawn_detected'] = self.yawn_alert.update(
                        probability, output['sampled_this_request'])
                    eye_buffer = self.buffers.get('eye_state')
                    if eye_buffer is None:
                        output.update(state='model_missing', message='Eye-state model unavailable')
                        self.alert.update(None, now)
                    elif len(eye_buffer) < eye_buffer.maxlen:
                        output.update(state='warming_up', message='Collecting an eye-closure window')
                        self.alert.update(None, now)
                    else:
                        closed = sum(eye_buffer) / len(eye_buffer)
                        yawn = output['predictions'].get('yawn')
                        yawn_probability = yawn['yawn'] if yawn else 0.
                        # These are interpretable evidence scores, not learned 3-class probabilities.
                        drowsy_evidence = min(1., closed / .65)
                        low_evidence = min(1., max(closed / .35, yawn_probability,
                                                   1. if output['head']['cue'] else 0.))
                        alert_evidence = max(0., 1. - max(closed, yawn_probability))
                        output['predictions']['drowsiness'] = dict(
                            alert=alert_evidence, low_vigilance=low_evidence, drowsy=drowsy_evidence)
                        output['closed_eye_proportion'] = closed
                        confirmed = self.alert.update(closed, now)
                        if confirmed:
                            output.update(state='drowsy', message='Sustained eye closure detected — take a break', alert=True)
                        elif closed >= .65:
                            output.update(state='checking', message='Checking sustained drowsiness')
                        elif closed >= .35 or output['yawn_detected'] or output['head']['cue']:
                            message = ('Sustained head tilt or movement detected' if output['head']['cue']
                                       else 'Fatigue cue detected')
                            output.update(state='low_vigilance', message=message)
                        else:
                            output.update(state='alert', message='Alert state detected')
            output['inference_ms'] = round((time.perf_counter() - begin) * 1000, 1)
            return output
