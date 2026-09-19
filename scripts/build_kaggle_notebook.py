"""Build a standalone Kaggle GPU notebook with embedded code and model assets."""
import base64
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    item = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip() + '\n')
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    return item


def main():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        files = list((ROOT / 'drowsiness').glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
        files += list((ROOT / 'configs').glob('*.json')) + [ROOT / 'requirements.txt']
        files += list((ROOT / 'models/assets').glob('*.onnx'))
        files += list((ROOT / 'models/cache/hub/checkpoints').glob('mobilenet_v3_small-*.pth'))
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    payload = base64.b64encode(buffer.getvalue()).decode()

    cells = [cell('markdown', '''
        # Driver drowsiness training on Kaggle

        This notebook trains MobileNetV3-Small models for **open/closed eyes** and
        **yawn/no-yawn**. Head movement is calculated from face landmarks at runtime
        and does not require a third model.

        Before running:

        1. Open **Settings > Accelerator** and select a GPU.
        2. Add a private Kaggle dataset containing `mrlEyes_2018_01.zip`.
        3. Add a private Kaggle dataset containing either `YawDD.rar.gz` or the
           extracted YawDD video folders.
        4. Enable Internet only if YawDD is still a RAR archive and the runtime does
           not already have `bsdtar`. Project code, YuNet, and ImageNet weights are embedded.

        Outputs are written under `/kaggle/working/driver_output`. Rerunning in the
        same session resumes from `last.pt`. To retain checkpoints between Kaggle
        sessions, save a notebook version with outputs.
        '''), cell('code', f'''
        from pathlib import Path
        import base64, io, json, os, shutil, subprocess, sys, time, zipfile
        import torch

        print('KAGGLE DRIVER-DROWSINESS SETUP STARTED', flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError('Enable a GPU in Kaggle Settings > Accelerator, then restart the session.')
        print('GPU:', torch.cuda.get_device_name(0), flush=True)

        PROJECT = Path('/kaggle/working/driver_project')
        PROJECT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:
            archive.extractall(PROJECT)
        os.chdir(PROJECT)
        os.environ['TORCH_HOME'] = str(PROJECT / 'models/cache')
        INPUT = Path('/kaggle/input')
        OUTPUT = Path('/kaggle/working/driver_output')
        DATA = Path('/kaggle/working/driver_data')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        DATA.mkdir(parents=True, exist_ok=True)
        print('Project ready:', PROJECT)
        '''), cell('code', '''
        def exactly_one(label, candidates):
            candidates = sorted(set(Path(p) for p in candidates))
            if len(candidates) != 1:
                listing = '\\n'.join(str(p) for p in candidates[:20]) or '(none found)'
                raise RuntimeError(f'Expected exactly one {label}; found {len(candidates)}:\\n{listing}')
            print(f'{label}: {candidates[0]}', flush=True)
            return candidates[0]

        mrl_archive = exactly_one('MRL archive', INPUT.rglob('mrlEyes_2018_01.zip'))
        yaw_archives = list(INPUT.rglob('YawDD.rar.gz'))
        yaw_wrappers = list(INPUT.rglob('YawDD_Kaggle.zip'))
        yaw_video_zips = list(INPUT.rglob('YawDD_Mirror_Kaggle.zip'))
        if yaw_archives:
            yaw_source = exactly_one('YawDD archive', yaw_archives)
        elif yaw_video_zips:
            video_zip = exactly_one('sanitized YawDD Mirror ZIP', yaw_video_zips)
            yaw_source = Path('/kaggle/working/yawdd_mirror_videos')
            if not yaw_source.is_dir():
                print('Extracting sanitized YawDD Mirror videos...', flush=True)
                with zipfile.ZipFile(video_zip) as archive:
                    if archive.testzip() is not None:
                        raise RuntimeError('YawDD_Mirror_Kaggle.zip is damaged.')
                    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in archive.namelist()):
                        raise RuntimeError('YawDD_Mirror_Kaggle.zip contains an unsafe path.')
                    archive.extractall(yaw_source)
        elif yaw_wrappers:
            wrapper = exactly_one('YawDD Kaggle ZIP wrapper', yaw_wrappers)
            yaw_source = Path('/kaggle/working/yaw_source/YawDD.rar.gz')
            yaw_source.parent.mkdir(parents=True, exist_ok=True)
            if not yaw_source.is_file():
                print('Extracting the inner YawDD archive to Kaggle working storage...', flush=True)
                with zipfile.ZipFile(wrapper) as archive:
                    if archive.namelist() != ['YawDD.rar.gz'] or archive.testzip() is not None:
                        raise RuntimeError('YawDD_Kaggle.zip is damaged or has unexpected contents.')
                    archive.extract('YawDD.rar.gz', yaw_source.parent)
        else:
            yaw_videos = [p for p in INPUT.rglob('*')
                          if p.suffix.lower() in {{'.avi', '.mp4', '.mov', '.mkv', '.m4v'}}
                          and 'mirror' in p.as_posix().lower()]
            if not yaw_videos:
                raise RuntimeError('YawDD was not found. Attach YawDD_Kaggle.zip, YawDD.rar.gz, or extracted videos.')
            yaw_source = INPUT
            print(f'Using extracted YawDD input containing {len(yaw_videos)} Mirror videos.', flush=True)

        if yaw_source.is_file() and yaw_source.name == 'YawDD.rar.gz':
            if not shutil.which('bsdtar'):
                subprocess.run(['apt-get', 'update', '-qq'], check=True)
                subprocess.run(['apt-get', 'install', '-y', '-qq', 'libarchive-tools'], check=True)

        def run(*arguments):
            print('RUNNING:', ' '.join(str(value) for value in arguments), flush=True)
            environment = dict(os.environ, PYTHONUNBUFFERED='1')
            process = subprocess.Popen([str(value) for value in arguments], env=environment)
            started = time.monotonic()
            while process.poll() is None:
                time.sleep(30)
                print(f'STILL RUNNING ({(time.monotonic() - started) / 60:.1f} min): '
                      f'{Path(str(arguments[1])).name if len(arguments) > 1 else arguments[0]}', flush=True)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, process.args)
            print('COMPLETED:', arguments[1] if len(arguments) > 1 else arguments[0], flush=True)
        '''), cell('markdown', '''
        ## Prepare and train eye-state model

        Preparation and training use Kaggle's local working disk. Progress is printed
        every 25 batches. A checkpoint is saved after every completed epoch.
        '''), cell('code', '''
        print('STAGE 1/3: PREPARING AND TRAINING EYE MODEL', flush=True)
        eye_data = DATA / 'eye_state'
        eye_run = OUTPUT / 'eye_state'
        if not (eye_data / 'metadata.json').is_file():
            run(sys.executable, '-u', 'scripts/prepare_dataset.py',
                '--source', mrl_archive, '--output', eye_data)

        eye_complete = (eye_run / 'best.pt').is_file() and (eye_run / 'test_metrics.json').is_file()
        if not eye_complete:
            command = [sys.executable, '-u', 'scripts/train.py', '--data', eye_data,
                       '--output', eye_run, '--epochs', '15', '--freeze-epochs', '1',
                       '--batch-size', '128', '--workers', '2', '--patience', '4',
                       '--device', 'cuda']
            if (eye_run / 'last.pt').is_file():
                command.append('--resume')
                print('Resuming eye model from:', eye_run / 'last.pt', flush=True)
            elif eye_run.exists():
                shutil.rmtree(eye_run)
            run(*command)
        else:
            print('Eye model already complete:', eye_run / 'best.pt')
        '''), cell('markdown', '''
        ## Prepare and train yawn model

        YawDD Mirror videos provide video-level yawn labels. The preparation excludes
        Dash videos because they do not have equivalent unambiguous video labels.
        '''), cell('code', '''
        print('STAGE 2/3: PREPARING AND TRAINING YAWN MODEL', flush=True)
        yawn_data = DATA / 'yawn'
        yawn_run = OUTPUT / 'yawn'
        if not (yawn_data / 'metadata.json').is_file():
            if yawn_data.exists():
                shutil.rmtree(yawn_data)
            run(sys.executable, '-u', 'scripts/prepare_video_dataset.py',
                '--source', yaw_source, '--output', yawn_data, '--task', 'yawn')

        yawn_complete = (yawn_run / 'best.pt').is_file() and (yawn_run / 'test_metrics.json').is_file()
        if not yawn_complete:
            command = [sys.executable, '-u', 'scripts/train_video.py', '--task', 'yawn',
                       '--data', yawn_data, '--output', yawn_run, '--epochs', '40',
                       '--batch-size', '4', '--workers', '2', '--device', 'cuda']
            if (yawn_run / 'last.pt').is_file():
                command.append('--resume')
                print('Resuming yawn model from:', yawn_run / 'last.pt', flush=True)
            elif yawn_run.exists():
                shutil.rmtree(yawn_run)
            run(*command)
        else:
            print('Yawn model already complete:', yawn_run / 'best.pt')
        '''), cell('markdown', '''
        ## Validate and bundle the trained models
        '''), cell('code', '''
        print('STAGE 3/3: VALIDATING AND BUNDLING MODELS', flush=True)
        outputs = {'eye_state': eye_run, 'yawn': yawn_run}
        for task, folder in outputs.items():
            if not (folder / 'best.pt').is_file() or not (folder / 'test_metrics.json').is_file():
                raise RuntimeError(f'{task} training is incomplete: {folder}')
            print(task.upper(), json.loads((folder / 'test_metrics.json').read_text()), flush=True)

        bundle = Path('/kaggle/working/trained_models.zip')
        temporary = bundle.with_suffix('.zip.part')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for task, folder in outputs.items():
                for name in ('best.pt', 'last.pt', 'config.json', 'test_metrics.json',
                             'history.json', 'dataset_report.json', 'status.json'):
                    path = folder / name
                    if path.is_file():
                        archive.write(path, f'{task}/{name}')
        temporary.replace(bundle)
        print('Download from the Kaggle Output panel:', bundle, flush=True)
        ''')]

    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(name='python3', display_name='Python 3'),
                                  language_info=dict(name='python')),
                    cells=cells)
    for index, item in enumerate(cells):
        item['id'] = f'cell-{index:02d}'
    target = ROOT / 'notebooks/Train_Driver_Drowsiness_Kaggle.ipynb'
    target.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
