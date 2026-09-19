"""Convert ONNX to ORT with the matching host runtime and save test vectors."""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models_ort'

def main():
    if ort.__version__ != '1.18.1':
        raise RuntimeError('Use ONNX Runtime 1.18.1 to match the QNX build')
    DEST.mkdir(exist_ok=True)
    report = {}
    for name in ('eye_state', 'yawn_frame', 'face_yunet'):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        # Basic optimizations avoid host-specific packed weights and CPU kernels.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(DEST / (name + '.ort'))
        options.add_session_config_entry('session.save_model_format', 'ORT')
        source = ort.InferenceSession(str(ROOT / 'models_onnx' / (name + '.onnx')),
                                     options, providers=['CPUExecutionProvider'])
        inp = source.get_inputs()[0]
        if not all(isinstance(d, int) and d > 0 for d in inp.shape):
            raise RuntimeError(f'{name}: expected a fixed input shape, got {inp.shape}')
        pixels = np.random.default_rng(7).uniform(-1, 1, inp.shape).astype(np.float32)
        expected = source.run(None, {inp.name: pixels})
        target_options = ort.SessionOptions()
        target_options.intra_op_num_threads = 1
        target_options.inter_op_num_threads = 1
        target_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        target = ort.InferenceSession(str(DEST / (name + '.ort')), target_options,
                                     providers=['CPUExecutionProvider'])
        actual = target.run(None, {inp.name: pixels})
        pixels.astype('<f4').tofile(DEST / (name + '.input.f32'))
        outputs = []
        for i, (ref, got) in enumerate(zip(expected, actual)):
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)
            ref.astype('<f4').tofile(DEST / f'{name}.output{i}.f32')
            outputs.append({'name': source.get_outputs()[i].name,
                            'shape': list(ref.shape),
                            'max_conversion_error': float(np.max(np.abs(got-ref)))})
        report[name] = {'input': inp.name, 'shape': inp.shape, 'outputs': outputs}
        print(f'{name}: ORT conversion verified, input={inp.shape}, outputs={len(outputs)}', flush=True)
    (DEST / 'verification.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
