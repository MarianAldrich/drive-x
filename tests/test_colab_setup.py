import hashlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('colab_setup', ROOT / 'notebooks/Colab_Setup.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class ColabSetupTests(unittest.TestCase):
    def test_private_folder_pagination(self):
        service = Mock()
        metadata = dict(id='archive-id', name='YawDD.rar.gz', size='123', mimeType='application/gzip')
        service.files.return_value.list.return_value.execute.side_effect = [
            {'files': [], 'nextPageToken': 'next-page'}, {'files': [metadata]}]
        self.assertEqual(setup.find_yawdd(service, setup.YAWDD_FOLDER_ID), metadata)
        arguments = service.files.return_value.list.call_args_list
        self.assertIn(setup.YAWDD_FOLDER_ID, arguments[0].kwargs['q'])
        self.assertEqual(arguments[1].kwargs['pageToken'], 'next-page')

    def test_missing_ambiguous_or_shortcut_archive_rejected(self):
        for entries in ([], [{'id': 'a'}, {'id': 'b'}],
                        [dict(id='a', size='10', mimeType='application/vnd.google-apps.shortcut')]):
            with self.subTest(entries=entries):
                service = Mock()
                service.files.return_value.list.return_value.execute.return_value = {'files': entries}
                with self.assertRaises(RuntimeError):
                    setup.find_yawdd(service, setup.YAWDD_FOLDER_ID)

    def test_archive_validation(self):
        data = b'\x1f\x8b' + b'test archive bytes'
        path = Mock()
        path.is_file.return_value = True
        path.stat.return_value.st_size = len(data)
        path.open.side_effect = lambda mode: io.BytesIO(data)
        metadata = dict(size=str(len(data)), md5Checksum=hashlib.md5(data).hexdigest())
        self.assertTrue(setup.matches_archive(path, metadata))
        metadata['md5Checksum'] = '0' * 32
        self.assertFalse(setup.matches_archive(path, metadata))
        metadata['size'] = '1'
        self.assertFalse(setup.matches_archive(path, metadata))
        metadata = dict(size=str(len(data)))
        path.open.side_effect = lambda mode: io.BytesIO(b'xx' + data[2:])
        self.assertFalse(setup.matches_archive(path, metadata))


if __name__ == '__main__':
    unittest.main()
