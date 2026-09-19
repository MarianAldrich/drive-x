$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Binary = Join-Path $PSScriptRoot 'ort_smoke'
$Models = Join-Path $Root 'models_ort'
$Bundle = Join-Path $PSScriptRoot 'ort_trial'
if (-not (Test-Path $Binary)) { throw 'The QNX smoke-test executable has not been linked.' }
foreach ($Model in @('eye_state', 'yawn_frame', 'face_yunet')) {
    foreach ($Suffix in @('.ort', '.input.f32', '.output0.f32')) {
        if (-not (Test-Path (Join-Path $Models ($Model + $Suffix)))) {
            throw "Missing test asset: $Model$Suffix"
        }
    }
}
New-Item -ItemType Directory -Force $Bundle | Out-Null
Copy-Item -LiteralPath $Binary -Destination $Bundle -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run-smoke.sh') -Destination $Bundle -Force
Copy-Item -LiteralPath $Models -Destination $Bundle -Recurse -Force
Get-ChildItem -LiteralPath $Bundle -File -Recurse | ForEach-Object {
    $Relative = $_.FullName.Substring($Bundle.Length + 1).Replace('\', '/')
    if ($Relative -ne 'SHA256SUMS') {
        (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + $Relative
    }
} | Set-Content -Encoding ascii (Join-Path $Bundle 'SHA256SUMS')
Write-Output "Trial bundle: $Bundle"
