"""The same YuNet crop geometry is used offline and in the camera interface."""
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
             'models/face_detection_yunet/face_detection_yunet_2026may.onnx')
YUNET_NAME = 'face_detection_yunet_2026may.onnx'
CROP_VERSION = 'yunet-square-v1'


def bounded_box(x, y, width, height, shape):
    h, w = shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w, int(x + width)), min(h, int(y + height))
    return [x1, y1, x2, y2] if x2 - x1 >= 8 and y2 - y1 >= 8 else None


def crop(frame, box):
    x1, y1, x2, y2 = box
    return cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LINEAR)


class FaceLocator:
    def __init__(self, model_path, threshold=0.6):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f'Face detector missing: {model_path}. Run scripts/download_assets.py.')
        self.dynamic_input = '2026may' in Path(model_path).name
        self.detector = cv2.FaceDetectorYN.create(str(model_path), '', (320, 320), threshold, .3, 5000)

    def locate(self, frame):
        h, w = frame.shape[:2]
        if self.dynamic_input:
            # The current YuNet model accepts the native frame shape. This preserves
            # aspect ratio and landmark detail instead of stretching 4:3 video to 1:1.
            self.detector.setInputSize((w, h))
            image, scale_x, scale_y = frame, 1., 1.
        else:
            # Backward-compatible path for the fixed-shape 2023 model.
            image = cv2.resize(frame, (320, 320))
            scale_x, scale_y = w / 320, h / 320
        _, detected = self.detector.detect(image)
        if detected is None:
            return []
        found = []
        for face in detected:
            face = face.copy()
            face[[0, 2, 4, 6, 8, 10, 12]] *= scale_x
            face[[1, 3, 5, 7, 9, 11, 13]] *= scale_y
            x, y, fw, fh = face[:4]
            face_box = bounded_box(x - .08 * fw, y - .08 * fh, fw * 1.16, fh * 1.16, frame.shape)
            mx, my = (face[10:12] + face[12:14]) / 2
            mw = max(float(np.linalg.norm(face[10:12] - face[12:14])) * 1.9, .55 * fw)
            mouth_box = bounded_box(mx - mw / 2, my - .17 * fh, mw, .44 * fh, frame.shape)
            eye_width, eye_height = .36 * fw, .24 * fh
            eyes = [bounded_box(ex - eye_width / 2, ey - eye_height / 2,
                                eye_width, eye_height, frame.shape)
                    for ex, ey in (face[4:6], face[6:8])]
            if face_box and mouth_box and all(eyes):
                found.append(dict(face=face_box, mouth=mouth_box, eyes=eyes,
                                  landmarks=[face[i:i + 2].tolist() for i in (4, 6, 8, 10, 12)],
                                  confidence=float(face[-1])))
        return sorted(found, key=lambda item: (item['face'][2] - item['face'][0]) *
                      (item['face'][3] - item['face'][1]), reverse=True)
