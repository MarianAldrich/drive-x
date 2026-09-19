"""Run once after the old notebook's download_assets.py failure, then rerun that cell."""
from pathlib import Path
import urllib.request

from drowsiness.vision import FaceLocator, YUNET_NAME, YUNET_URL

target = Path('/content/driver_project/models/assets') / YUNET_NAME
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix('.download.onnx')
urllib.request.urlretrieve(YUNET_URL, temporary)
FaceLocator(temporary)  # Validate the download while its filename still identifies ONNX.
temporary.replace(target)
FaceLocator(target)
print('Face detector fixed and verified:', target)
