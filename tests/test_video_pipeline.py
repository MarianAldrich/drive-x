import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from drowsiness.model import VideoMobileNet, TASKS
from drowsiness.runtime import AlertState, HeadMovementTracker, InferenceEngine, YawnState
from drowsiness.vision import CROP_VERSION
from prepare_video_dataset import uta_label, yaw_label, safe_member
from train_video import load_records
from app.server import make_handler
from http.server import ThreadingHTTPServer


class VideoPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dataset_labels(self):
        self.assertEqual(uta_label('Fold1/01/10.mp4', 1)['label'], 2)
        self.assertEqual(uta_label('01/5.MOV', 1)['subject'], 'uta:01')
        with self.assertRaises(ValueError):
            uta_label('01/sleepy.mp4', 1)
        self.assertIsNone(yaw_label('YawDD dataset/Dash/Male/1-MaleGlasses.avi'))
        self.assertEqual(yaw_label('Mirror/Female/1-FemaleNoGlasses-Yawning.avi')['label'], 1)
        self.assertEqual(yaw_label('Mirror/Male/1-MaleGlasses-Talking.avi')['label'], 0)
        self.assertEqual(yaw_label('Mirror/Male/1-MaleGlasses-Talking&Yawning.avi')['label'], 1)
        self.assertNotEqual(yaw_label('Mirror/Female/1-FemaleNoGlasses-Yawning.avi')['subject'],
                            yaw_label('Mirror/Male/1-MaleGlasses-Yawning.avi')['subject'])
        for path in ('../bad.avi', '/bad.avi', 'C:/bad.avi', '..\\bad.avi'):
            with self.assertRaises(ValueError):
                safe_member(path)

    def test_mil_pool_and_alert_gaps(self):
        model = VideoMobileNet('yawn')
        logits = torch.tensor([[[0., -4.], [0., -4.], [0., -4.], [0., 4.]]])
        self.assertGreater(model.pool(logits).softmax(-1)[0, 1].item(), .95)
        output = model(torch.rand(2, 4, 3, 64, 64))
        torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
        self.assertGreater(model.encoder.classifier[-1].weight.grad.abs().sum().item(), 0)
        alert = AlertState()
        self.assertFalse(alert.update(.9, 0))
        self.assertFalse(alert.update(.9, 1))
        self.assertTrue(alert.update(.9, 2))
        self.assertFalse(alert.update(None, 3))
        self.assertFalse(alert.update(.9, 4))
        self.assertFalse(alert.update(.9, 10))  # A camera outage cannot count as sustained evidence.
        yawn = YawnState()
        self.assertFalse(yawn.update(.6))
        self.assertTrue(yawn.update(.6))
        self.assertTrue(yawn.update(.1, new_sample=False))
        self.assertFalse(yawn.update(.1))

    def test_head_movement_calibration_and_sustained_cue(self):
        tracker = HeadMovementTracker(calibration_samples=3)
        neutral = [[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]]
        self.assertFalse(tracker.update(neutral)['calibrated'])
        tracker.update(neutral)
        self.assertTrue(tracker.update(neutral)['calibrated'])
        moved = [[30, 30], [70, 30], [70, 50], [35, 70], [65, 70]]
        for _ in range(5):
            self.assertFalse(tracker.update(moved)['cue'])
        result = tracker.update(moved)
        self.assertTrue(result['cue'])
        self.assertEqual(result['status'], 'turned sideways')
        tracker.reset()
        self.assertFalse(tracker.update(neutral)['calibrated'])

    def test_training_resume_and_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            records = []
            for split_index, split in enumerate(('train', 'val', 'test')):
                for label in range(3):
                    frames = []
                    video_id = f'{split}_{label}'
                    for frame in range(4):
                        path = f'{video_id}_{frame}.jpg'
                        Image.new('RGB', (48, 48), (label * 80, frame * 50, split_index * 60)).save(data / path)
                        frames.append(dict(path=path, seconds=frame))
                    records.append(dict(id=video_id, subject=f's{split_index}', split=split, label=label, frames=frames))
            (data / 'videos.json').write_text(json.dumps(records))
            (data / 'metadata.json').write_text(json.dumps(dict(task='drowsiness', classes=TASKS['drowsiness'], crop_version=CROP_VERSION)))
            model_root = root / 'models'
            command = [sys.executable, str(ROOT / 'scripts/train_video.py'), '--data', str(data),
                       '--task', 'drowsiness', '--output', str(model_root / 'drowsiness'),
                       '--epochs', '1', '--weights', 'none', '--workers', '0', '--batch-size', '2',
                       '--bag-size', '2', '--device', 'cpu']
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            command[command.index('--epochs') + 1] = '2'
            result = subprocess.run(command + ['--resume'], env=env, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            history = json.loads((model_root / 'drowsiness/history.json').read_text())
            self.assertEqual([entry['epoch'] for entry in history], [1, 2])
            report = json.loads((model_root / 'drowsiness/test_metrics.json').read_text())
            self.assertEqual(sum(map(sum, report['confusion_matrix'])), 3)
            eye = models.mobilenet_v3_small(weights=None)
            eye.classifier[-1] = nn.Linear(eye.classifier[-1].in_features, 2)
            eye_dir = model_root / 'eye_state'
            eye_dir.mkdir()
            torch.save(dict(model_state_dict=eye.state_dict(), epoch=1,
                            config=dict(architecture='mobilenet_v3_small', task='eye_state',
                                        class_to_idx={'closed': 0, 'open': 1})), eye_dir / 'best.pt')
            detector = ROOT / 'models/assets/face_detection_yunet_2023mar.onnx'
            engine = InferenceEngine(detector, model_root)
            self.assertTrue(engine.status()['models']['eye_state']['loaded'])
            image = np.zeros((120, 160, 3), np.uint8)
            box = dict(face=[10, 10, 110, 110], mouth=[30, 60, 90, 105],
                       eyes=[[25, 30, 55, 55], [65, 30, 95, 55]],
                       landmarks=[[35, 40], [75, 40], [55, 60], [42, 85], [68, 85]],
                       confidence=.99)
            with patch.object(engine.locator, 'locate', return_value=[box]):
                self.assertEqual(engine.detect(image, now=0)['state'], 'warming_up')
                output = None
                for second in range(1, 8):
                    output = engine.detect(image, now=second)
                self.assertIn('drowsiness', output['predictions'])
            with patch.object(engine.locator, 'locate', return_value=[]):
                output = engine.detect(image, now=8)
                self.assertEqual(output['state'], 'no_face')
                self.assertFalse(output['alert'])
                self.assertEqual(len(engine.buffers['eye_state']), 8)  # tolerate one transient miss
                engine.detect(image, now=8.2)
                output = engine.detect(image, now=8.4)
                self.assertEqual(len(engine.buffers['eye_state']), 0)  # sustained loss clears evidence
            with patch.object(engine.locator, 'locate', return_value=[box]):
                self.assertEqual(engine.detect(image, now=9)['state'], 'warming_up')
            records[-1]['subject'] = 's0'
            (data / 'videos.json').write_text(json.dumps(records))
            with self.assertRaises(ValueError):
                load_records(data, 'drowsiness')

    def test_http_missing_models_and_invalid_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = InferenceEngine(ROOT / 'models/assets/face_detection_yunet_2023mar.onnx', temp)
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(engine))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with urllib.request.urlopen(base + '/') as response:
                    self.assertIn(b'DRIVE-X', response.read())
                with urllib.request.urlopen(base + '/api/status') as response:
                    self.assertFalse(json.load(response)['models']['eye_state']['loaded'])
                request = urllib.request.Request(base + '/api/detect', data=b'bad image')
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request)
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
                _, jpg = cv2.imencode('.jpg', np.zeros((100, 100, 3), np.uint8))
                request = urllib.request.Request(base + '/api/detect', data=jpg.tobytes())
                with urllib.request.urlopen(request) as response:
                    result = json.load(response)
                    self.assertEqual(result['state'], 'no_face')
                    self.assertFalse(result['alert'])
                request = urllib.request.Request(base + '/api/reset', data=b'', headers={'Origin': 'http://example.com'})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request)
                self.assertEqual(error.exception.code, 403)
                error.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_notebook_code_compiles(self):
        for name in ('Train_Driver_Drowsiness.ipynb', 'Train_Driver_Drowsiness_Kaggle.ipynb'):
            notebook = json.loads((ROOT / 'notebooks' / name).read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    compile(cell['source'], name, 'exec')


if __name__ == '__main__':
    unittest.main()
