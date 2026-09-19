from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from download_assets import ensure_detector, YUNET_NAME


class DownloadAssetsTests(unittest.TestCase):
    def test_first_download_loads_actual_onnx_and_reuses_existing_asset(self):
        source = ROOT / 'models/assets' / YUNET_NAME
        if not source.is_file():
            self.skipTest('Install the YuNet asset before running the real OpenCV regression test.')
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'assets' / YUNET_NAME

            def offline_download(url, filename):
                self.assertEqual(Path(filename).suffix, '.onnx')
                shutil.copyfile(source, filename)

            with patch('download_assets.urllib.request.urlretrieve', side_effect=offline_download) as download:
                ensure_detector(destination)  # Uses the real OpenCV loader on a newly downloaded file.
                self.assertEqual(destination.read_bytes(), source.read_bytes())
                self.assertFalse(destination.with_suffix('.download.onnx').exists())
                ensure_detector(destination)
                download.assert_called_once()


if __name__ == '__main__':
    unittest.main()
