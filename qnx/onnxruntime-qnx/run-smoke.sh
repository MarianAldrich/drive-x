#!/bin/sh
# Run from the extracted trial directory. The first failure stops the test.
set -eu
cd "$(dirname "$0")"
chmod 755 ./ort_smoke
for model in eye_state yawn_frame face_yunet; do
    echo "Testing $model on QNX"
    ./ort_smoke "models_ort/$model"
done
echo "All three ONNX Runtime QNX model tests passed."
