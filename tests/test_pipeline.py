"""Small integration checks, including a real forward/backward training run."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_dataset import parse_name, prepare, split_subjects


class PipelineTests(unittest.TestCase):
    def test_labels_and_splits(self):
        self.assertEqual(parse_name('s0001_00001_0_0_1_0_0_01.png'), ('s0001', 1))
        with self.assertRaises(ValueError):
            parse_name('unknown.png')
        subjects = [f's{i:04d}' for i in range(37)]
        assignments = split_subjects(subjects, 42, .2, .1)
        self.assertEqual(assignments, split_subjects(reversed(subjects), 42, .2, .1))
        self.assertEqual(set(assignments.values()), {'train', 'val', 'test'})
        with self.assertRaises(ValueError):
            split_subjects(subjects, 42, .8, .3)

    def test_prepare_and_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'eyes.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for subject in range(1, 7):
                    for label in (0, 1):
                        buffer = io.BytesIO()
                        Image.new('RGB', (40, 40), (subject * 30, label * 200, 50)).save(buffer, format='PNG')
                        name = f's{subject:04d}_00001_0_0_{label}_0_0_01.png'
                        z.writestr('eyes/' + name, buffer.getvalue())
                z.writestr('eyes/s0001_00002_0_0_0_0_0_01.png', b'corrupt')
                buffer = io.BytesIO()
                Image.new('RGB', (40, 40), (255, 255, 255)).save(buffer, format='PNG')
                for label in (0, 1):
                    z.writestr(f'eyes/s0001_00003_0_0_{label}_0_0_01.png', buffer.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(archive, root / 'data')
            metadata = json.loads((root / 'data/metadata.json').read_text())
            self.assertEqual(len(metadata['rejected']), 3)
            with self.assertRaises(ValueError):
                prepare(archive, root / 'data')
            environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            command = [
                sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/train.py'),
                '--data', str(root / 'data'), '--output', str(root / 'run'),
                '--weights', 'none', '--epochs', '1', '--batch-size', '4', '--device', 'cpu']
            result = subprocess.run(command,
                env=environment, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / 'run/test_metrics.json').read_text())
            self.assertEqual(sum(map(sum, report['confusion_matrix'])), 2)
            self.assertTrue((root / 'run/best.pt').is_file())
            command[command.index('--epochs') + 1] = '2'
            result = subprocess.run(command + ['--resume'], env=environment,
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            history = json.loads((root / 'run/history.json').read_text())
            self.assertEqual([entry['epoch'] for entry in history], [1, 2])


if __name__ == '__main__':
    unittest.main()
