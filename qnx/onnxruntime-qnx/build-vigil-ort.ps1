$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Ort = Join-Path $Root 'third_party/onnxruntime'
$Build = Join-Path $Ort 'build-qnx-minimal'
if (-not $env:QNX_HOST) { throw 'Run qnxsdp-env.bat first.' }
if (-not (Test-Path (Join-Path $Build 'libonnxruntime_session.a'))) {
    throw 'Build ONNX Runtime before building the pipeline executable.'
}
$Compiler = Join-Path $env:QNX_HOST 'usr/bin/aarch64-unknown-nto-qnx8.0.0-g++.exe'
$Source = Join-Path $Root 'src/vigil_ort_inference.cpp'
$Output = Join-Path $Root 'build/vigil_ort_inference'
$Response = Join-Path $PSScriptRoot 'vigil-ort-link.rsp'
$Libraries = Get-ChildItem -LiteralPath $Build -Recurse -Filter '*.a' |
    Where-Object { $_.FullName -notmatch 'CMakeScratch' } | Sort-Object FullName
New-Item -ItemType Directory -Force (Split-Path -Parent $Output) | Out-Null
$Arguments = @('-std=c++17', '-D_QNX_SOURCE', '-O2', '-Wall', '-Wextra',
    ('-I"' + (Join-Path $Root 'include').Replace('\','/') + '"'),
    ('-I"' + (Join-Path $Ort 'include/onnxruntime/core/session').Replace('\','/') + '"'),
    ('"' + $Source.Replace('\','/') + '"'), '-o', ('"' + $Output.Replace('\','/') + '"'),
    '-Wl,--start-group')
$Arguments += $Libraries | ForEach-Object { '"' + $_.FullName.Replace('\','/') + '"' }
$Arguments += @('-Wl,--end-group', '-lm', '-lsocket')
$Arguments | Set-Content -Encoding ascii $Response
& $Compiler "@$Response"
exit $LASTEXITCODE
