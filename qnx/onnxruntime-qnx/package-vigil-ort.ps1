$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Binary = Join-Path $Root 'build/vigil_ort_inference'
$Models = Join-Path $Root 'models_ort'
$Bundle = Join-Path $PSScriptRoot 'ort_pipeline_update'
if (-not (Test-Path $Binary)) { throw 'Run build-vigil-ort.ps1 first.' }
foreach ($Name in @('face_yunet.ort', 'eye_state.ort', 'yawn_frame.ort')) {
    if (-not (Test-Path (Join-Path $Models $Name))) { throw "Missing model: $Name" }
}
Remove-Item -LiteralPath $Bundle -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force (Join-Path $Bundle 'build'), (Join-Path $Bundle 'models_ort') | Out-Null
Copy-Item -LiteralPath $Binary -Destination (Join-Path $Bundle 'build/vigil_ort_inference')
Get-ChildItem -LiteralPath $Models -Filter '*.ort' | Copy-Item -Destination (Join-Path $Bundle 'models_ort')
Copy-Item -LiteralPath (Join-Path $Root 'run_demo.sh') -Destination $Bundle
@'
This update runs all three models on the QNX Pi through ONNX Runtime.

Copy build/vigil_ort_inference and models_ort into the existing qnx_bundle.
Replace run_demo.sh with this copy. Then, on QNX:

  cd /data/home/qnxuser/qnx_bundle
  chmod 755 build/vigil_ort_inference run_demo.sh
  export VIGIL_LOCAL_INFERENCE=1
  export VIGIL_INFERENCE_RUNTIME=ort
  export VIGIL_GPS_DEVICE=/dev/null
  sh ./run_demo.sh

The log must show runtime=onnxruntime-1.18.1 and INFERENCE frame=... lines.
'@ | Set-Content -Encoding ascii (Join-Path $Bundle 'README.txt')
Compress-Archive -LiteralPath $Bundle -DestinationPath (Join-Path $PSScriptRoot 'ort_pipeline_update.zip') -Force
Write-Output "Update bundle: $(Join-Path $PSScriptRoot 'ort_pipeline_update.zip')"
