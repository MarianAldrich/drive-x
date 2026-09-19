param(
    [string]$QnxHost = '',
    [int]$QnxPort = 45555,
    [switch]$UdpDisabled
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python scripts/download_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Face detector setup failed.' }
Write-Host 'Open http://127.0.0.1:8765 in Chrome or Edge.'
$serverArgs = @('app/server.py', '--qnx-port', $QnxPort)
if ($UdpDisabled) {
    $serverArgs += '--udp-disabled'
} elseif ($QnxHost) {
    $serverArgs += @('--qnx-host', $QnxHost)
}
python @serverArgs
