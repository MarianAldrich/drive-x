$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Ort = Join-Path $Root 'third_party/onnxruntime'
$Build = Join-Path $Ort 'build-qnx-minimal'
if (-not $env:QNX_HOST) { throw 'Run qnxsdp-env.bat first.' }
$Compiler = Join-Path $env:QNX_HOST 'usr/bin/aarch64-unknown-nto-qnx8.0.0-g++.exe'
$Output = Join-Path $PSScriptRoot 'ort_smoke'
$Response = Join-Path $PSScriptRoot 'smoke-link.rsp'
$Libraries = Get-ChildItem -LiteralPath $Build -Recurse -Filter '*.a' |
    Where-Object { $_.FullName -notmatch 'CMakeScratch' } |
    Sort-Object FullName
if (-not (Test-Path (Join-Path $Build 'libonnxruntime_session.a'))) {
    throw 'Build ONNX Runtime successfully before linking the smoke test.'
}
$Arguments = @('-std=c++17', '-D_QNX_SOURCE', '-O2', '-g',
    ('-I"' + (Join-Path $Ort 'include/onnxruntime/core/session').Replace('\','/') + '"'),
    ('"' + (Join-Path $PSScriptRoot 'ort_smoke.cpp').Replace('\','/') + '"'),
    '-Wl,--start-group')
$Arguments += $Libraries | ForEach-Object { '"' + $_.FullName.Replace('\','/') + '"' }
$Arguments += @('-Wl,--end-group', '-lm', '-lsocket', '-o', ('"' + $Output.Replace('\','/') + '"'))
$Arguments | Set-Content -Encoding ascii $Response
& $Compiler "@$Response"
exit $LASTEXITCODE
