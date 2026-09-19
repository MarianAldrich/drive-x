"""Build a standalone Colab notebook embedding this project's training code."""
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
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    payload = base64.b64encode(buffer.getvalue()).decode()
    cells = [cell('markdown', '''
        # Driver drowsiness · MobileNetV3-Small

        This notebook trains **two models**: UTA-RLDD alert/low-vigilance/drowsy and a
        YawDD yawn/no-yawn video model. It contains the project code; no repository setup is needed.

        **Before Run all:**
        1. Select **Runtime → Change runtime type → T4 GPU** (or another CUDA GPU).
        2. YawDD is read from your folder `1hRWkRD1xKQ0reWvM5o9OSGs1oPhCVSov`.
           Sign in with the Google account that has access; no public sharing is needed.
        3. Run all cells and authorize Drive mounting in your own browser.

        This smaller pipeline uses MRL Eyes (~342 MB) plus YawDD (~5.3 GB), avoiding
        the 119.5 GB UTA-RLDD download and its public Google Drive quota.

        Both datasets use subject-disjoint training, validation, and test splits.

        YawDD labels supervise bags of mouth frames. They are not frame-level yawn
        annotations. Scores on short live windows still need independent validation.
        This notebook has not been GPU-executed by the coding assistant.
        '''), cell('code', (ROOT / 'notebooks/Colab_Setup.py').read_text(encoding='utf-8')), cell('code', f'''
        import base64, io, os, zipfile
        PROJECT = Path('/content/driver_project')
        PROJECT.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:
            archive.extractall(PROJECT)
        os.chdir(PROJECT)
        print("Project restored:", PROJECT)
        '''), cell('code', '''
        import subprocess, sys
        def run(*args):
            subprocess.run([str(arg) for arg in args], check=True)
        run(sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt',
            'gdown>=5.2,<6', 'onnx>=1.16', 'onnxruntime>=1.18')
        run('apt-get', 'update', '-qq')
        run('apt-get', 'install', '-y', '-qq', 'libarchive-tools')
        # Print the child traceback explicitly so setup failures are visible in Colab.
        asset_result = subprocess.run([sys.executable, 'scripts/download_assets.py'],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(asset_result.stdout, flush=True)
        asset_result.check_returncode()
        environment = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (DRIVE_ROOT / 'environment.txt').write_text(environment)
        DATA = Path('/content/driver_data')
        '''), cell('markdown', '''
        ## Prepare and train YawDD first
        This provides an initial trained artifact before the much larger UTA download.
        The script excludes unlabeled Dash videos. Talking-and-yawning videos are
        positive bags because they contain a yawn; their individual frames are not labeled positive.
        Check `metadata.json` in the cached preparation for exclusions and crop failures.
        '''), cell('code', '''
        run(sys.executable, 'scripts/colab_prepare.py', '--task', 'yawn',
            '--drive-root', DRIVE_ROOT, '--work', DATA, '--yawdd', YAWDD_PATH)
        def train(task):
            output = DRIVE_ROOT / 'training' / task
            if (output / 'best.pt').is_file() and (output / 'test_metrics.json').is_file():
                print(f'Reusing completed {task} model:', output / 'best.pt')
                return output
            args = [sys.executable, 'scripts/train_video.py', '--task', task,
                    '--data', DATA / task, '--output', output, '--epochs', EPOCHS,
                    '--batch-size', BATCH_SIZE, '--workers', '2', '--device', 'cuda']
            if (output / 'last.pt').is_file():
                args += ['--resume']
            elif output.exists():
                raise RuntimeError(f"Incomplete run before its first checkpoint: {output}. Rename that directory and rerun.")
            run(*args)
            return output
        yawn_run = train('yawn')
        '''), cell('markdown', '''
        ## Train the smaller MRL eye-state branch and bundle both models
        MRL supplies open/closed eye supervision. The live interface fuses sustained
        eye closure with the separately trained YawDD yawn evidence.
        '''), cell('code', (ROOT / 'notebooks/Colab_Small_Dataset_Training.py').read_text(encoding='utf-8')), cell('markdown', '''
        '''), cell('markdown', '''
        ## Open the Windows live interface
        Extract `trained_models.zip` into the project's `models/trained/` directory.
        It should contain `eye_state/best.pt` and `yawn/best.pt`.
        Run `python app/server.py`, open **http://127.0.0.1:8765**, and click **Start camera**.
        Allow about 16 seconds to fill the first model window. If the interface was already
        running, click **Reload models**.

        References: [MRL Eyes](https://mrl.cs.vsb.cz/eyedataset.html),
        [YawDD](https://doi.org/10.21227/e1qm-hb90),
        [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
        ''')]
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(colab=dict(name='Train_Driver_Drowsiness.ipynb'), accelerator='GPU',
                                  kernelspec=dict(name='python3', display_name='Python 3')),
                    cells=cells)
    for i, item in enumerate(cells):
        item['id'] = f'cell-{i:02d}'
    output = ROOT / 'notebooks/Train_Driver_Drowsiness.ipynb'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
