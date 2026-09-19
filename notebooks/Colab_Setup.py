"""Paste this entire file into the FIRST code cell of the existing Colab notebook.

Reads YawDD from your private folder using your own Colab login. No public sharing
or moving the original archive is necessary. Outputs still go to DriverDrowsiness.
"""
from pathlib import Path
import hashlib
import re
import shutil

YAWDD_FOLDER_ID = '1hRWkRD1xKQ0reWvM5o9OSGs1oPhCVSov'
EPOCHS = 40
BATCH_SIZE = 4


def find_yawdd(service, folder_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', folder_id):
        raise ValueError('Set YAWDD_FOLDER_ID to the folder ID, not its full URL.')
    matches, page = [], None
    while True:
        result = service.files().list(
            q=f"'{folder_id}' in parents and name = 'YawDD.rar.gz' and trashed = false",
            fields='nextPageToken,files(id,name,size,md5Checksum,mimeType)',
            pageSize=100, pageToken=page, supportsAllDrives=True,
            includeItemsFromAllDrives=True).execute()
        matches.extend(result.get('files', []))
        page = result.get('nextPageToken')
        if not page:
            break
    if len(matches) != 1:
        raise RuntimeError(f'Found {len(matches)} files named YawDD.rar.gz in the specified folder. '
                           'Use the account that can access it and keep exactly one archive directly in that folder.')
    if not matches[0].get('size') or matches[0]['mimeType'] == 'application/vnd.google-apps.shortcut':
        raise RuntimeError('Expected the uploaded archive itself, not a Drive shortcut.')
    return matches[0]


def matches_archive(path, metadata):
    if not path.is_file() or path.stat().st_size != int(metadata['size']):
        return False
    with path.open('rb') as source:
        if source.read(2) != b'\x1f\x8b':
            return False
        if metadata.get('md5Checksum'):
            source.seek(0)
            digest = hashlib.md5(usedforsecurity=False)
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(block)
            return digest.hexdigest() == metadata['md5Checksum']
    return True


def setup_colab():
    import torch
    from google.colab import drive, auth  # pyright: ignore[reportMissingImports]
    import google.auth  # pyright: ignore[reportMissingImports]
    from googleapiclient.discovery import build  # pyright: ignore[reportMissingImports]
    from googleapiclient.http import MediaIoBaseDownload  # pyright: ignore[reportMissingImports]

    if not torch.cuda.is_available():
        raise RuntimeError('Select Runtime > Change runtime type > T4 GPU before running this cell.')
    print('GPU:', torch.cuda.get_device_name(0))
    drive.mount('/content/drive')
    output_root = Path('/content/drive/MyDrive/DriverDrowsiness')
    output_root.mkdir(parents=True, exist_ok=True)
    target = Path('/content/driver_sources/YawDD.rar.gz')
    if (output_root / 'prepared_v1/yawn.zip').is_file():
        print('Completed YawDD crop cache found; preparation will restore it without downloading the archive.')
        return output_root, target

    # Authentication stays inside the user's Colab session. Never paste or share tokens.
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    metadata = find_yawdd(service, YAWDD_FOLDER_ID)
    print(f"Found {metadata['name']} ({int(metadata['size']) / 1e9:.2f} GB) in your selected folder.")
    if matches_archive(target, metadata):
        print('Reusing the verified local archive:', target)
        return output_root, target
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < int(metadata['size']) + 10_000_000_000:
        raise RuntimeError('Need archive size plus at least 10 GB free local disk space for preparation.')
    temporary = target.with_suffix('.gz.part')
    request = service.files().get_media(fileId=metadata['id'], supportsAllDrives=True)
    with temporary.open('wb') as destination:
        downloader = MediaIoBaseDownload(destination, request, chunksize=16 * 1024 * 1024)
        done, previous = False, -1
        while not done:
            status, done = downloader.next_chunk(num_retries=5)
            if status:
                percent = int(status.progress() * 100)
                if percent // 5 != previous or done:
                    print(f'YawDD download: {percent}%', flush=True)
                    previous = percent // 5
    if not matches_archive(temporary, metadata):
        raise RuntimeError('Downloaded archive failed size, gzip-header, or checksum validation. Rerun this cell.')
    temporary.replace(target)
    print('YawDD ready:', target)
    return output_root, target


if __name__ == '__main__':
    DRIVE_ROOT, YAWDD_PATH = setup_colab()
