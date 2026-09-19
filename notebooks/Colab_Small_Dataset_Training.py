"""Prepare MRL Eyes locally, train visibly, resume safely, and bundle both models."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
import torch


def run_training(command):
    """Use unbuffered output so every batch-progress line appears in Colab."""
    print('Starting eye training on:', torch.cuda.get_device_name(0), flush=True)
    environment = dict(os.environ, PYTHONUNBUFFERED='1')
    subprocess.run([str(value) for value in command], check=True, env=environment)

MRL_URL = 'https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip'
drive_source = DRIVE_ROOT / 'mrlEyes_2018_01.zip'
local_source = Path('/content/driver_sources/mrlEyes_2018_01.zip')
source = drive_source if drive_source.is_file() else local_source
prepared = DATA / 'eye_state'
eye_run = DRIVE_ROOT / 'training' / 'eye_state_v2'
prepared_cache = DRIVE_ROOT / 'prepared_v2' / 'eye_state.zip'

if not (prepared / 'metadata.json').is_file():
    if prepared_cache.is_file():
        print('Restoring prepared MRL dataset cache from Drive...', flush=True)
        prepared.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(prepared_cache) as archive:
            archive.extractall(prepared.parent)
        if not (prepared / 'metadata.json').is_file():
            raise RuntimeError('Prepared dataset cache is incomplete. Delete it and rerun.')
        print('Prepared MRL dataset restored.', flush=True)
    else:
        local_source.parent.mkdir(parents=True, exist_ok=True)
        if drive_source.is_file():
            print('Using MRL Eyes archive from Drive:', drive_source, flush=True)
        elif not local_source.is_file():
            print('Downloading official MRL Eyes archive (~342 MB)...', flush=True)
            temporary = local_source.with_suffix('.zip.part')
            last_reported = [-1]
            def report_download(blocks, block_size, total):
                if total > 0:
                    percent = min(100, int(blocks * block_size * 100 / total))
                    if percent // 10 != last_reported[0]:
                        print(f'MRL download: {percent}%', flush=True)
                        last_reported[0] = percent // 10
            urllib.request.urlretrieve(MRL_URL, temporary, reporthook=report_download)
            with zipfile.ZipFile(temporary) as archive:
                bad = archive.testzip()
            if bad:
                raise RuntimeError(f'MRL ZIP integrity check failed at {bad}. Rerun this cell.')
            temporary.replace(local_source)
            source = local_source
        else:
            source = local_source
            print('Using verified local MRL archive:', source, flush=True)
        with zipfile.ZipFile(source) as archive:
            bad = archive.testzip()
        if bad:
            raise RuntimeError(f'MRL ZIP integrity check failed at {bad}. Upload/download it again.')
        run(sys.executable, 'scripts/prepare_dataset.py', '--source', source, '--output', prepared)
        prepared_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = prepared_cache.with_suffix('.zip.part')
        print('Caching prepared MRL dataset to Drive for future sessions...', flush=True)
        with zipfile.ZipFile(temporary_cache, 'w', zipfile.ZIP_STORED) as archive:
            for path in prepared.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(prepared.parent))
        temporary_cache.replace(prepared_cache)
        print('Prepared dataset cache saved:', prepared_cache, flush=True)
else:
    print('Using prepared MRL eye dataset from this runtime.')

eye_complete = (eye_run / 'best.pt').is_file() and (eye_run / 'test_metrics.json').is_file()
if not eye_complete:
    command = [sys.executable, 'scripts/train.py', '--data', prepared, '--output', eye_run,
               '--epochs', '15', '--freeze-epochs', '1', '--batch-size', '128',
               '--workers', '4', '--patience', '4', '--device', 'cuda']
    if (eye_run / 'last.pt').is_file():
        print('Resuming interrupted eye-state training from last.pt.', flush=True)
        run_training(command + ['--resume'])
    else:
        if eye_run.exists():
            suffix = 1
            backup = eye_run.with_name(f'{eye_run.name}_incomplete_{suffix}')
            while backup.exists():
                suffix += 1
                backup = eye_run.with_name(f'{eye_run.name}_incomplete_{suffix}')
            eye_run.rename(backup)
            print('Preserved incomplete eye run at:', backup)
        print('Starting the new clean eye-state training run.', flush=True)
        run_training(command)
else:
    print('Using completed eye-state checkpoint:', eye_run / 'best.pt')

outputs = {'eye_state': eye_run, 'yawn': DRIVE_ROOT / 'training' / 'yawn'}
for task, output in outputs.items():
    if not (output / 'best.pt').is_file() or not (output / 'test_metrics.json').is_file():
        raise RuntimeError(f'{task} training is incomplete: {output}')
    print(task.upper())
    print(json.dumps(json.loads((output / 'test_metrics.json').read_text()), indent=2))

bundle = DRIVE_ROOT / 'trained_models.zip'
temporary_bundle = bundle.with_suffix('.zip.part')
with zipfile.ZipFile(temporary_bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
    for task in ('eye_state', 'yawn'):
        folder = outputs[task]
        for name in ('best.pt', 'config.json', 'test_metrics.json', 'history.json'):
            if (folder / name).is_file():
                archive.write(folder / name, f'{task}/{name}')
        for name in ('dataset_report.json', 'status.json', 'encoder.onnx', 'encoder.json'):
            if (folder / name).is_file():
                archive.write(folder / name, f'{task}/{name}')
temporary_bundle.replace(bundle)
print('Saved smaller-pipeline models:', bundle)
from google.colab import files
files.download(str(bundle))
