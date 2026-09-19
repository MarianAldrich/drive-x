$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$OrtRoot = Join-Path $Root 'third_party\onnxruntime'
$BuildDir = Join-Path $OrtRoot 'build-qnx-minimal'
$Cmake = 'C:\Program Files\CodeBlocks\MinGW\bin\cmake.exe'

if (-not $env:QNX_HOST -or -not $env:QNX_TARGET) {
    throw 'QNX_HOST/QNX_TARGET are missing. Run qnxsdp-env.bat before this script.'
}

# Let CMake supply build flags instead of inheriting SDP Make defaults.
Remove-Item Env:MAKEFLAGS -ErrorAction SilentlyContinue

$Toolchain = Join-Path $PSScriptRoot 'aarch64-qnx.toolchain.cmake'
$QnxMake = Join-Path $env:QNX_HOST 'usr\bin\make.exe'
$Eigen = (Join-Path $Root 'third_party/eigen-ort').Replace('\', '/')
$CaBundle = (Join-Path $PSScriptRoot 'windows-ca-bundle.pem').Replace('\', '/')
if (-not (Test-Path $CaBundle)) {
    $Certificates = Get-ChildItem Cert:\LocalMachine\Root,Cert:\CurrentUser\Root
    $Pem = foreach ($Certificate in $Certificates) {
        '-----BEGIN CERTIFICATE-----'
        [Convert]::ToBase64String($Certificate.RawData, [Base64FormattingOptions]::InsertLineBreaks)
        '-----END CERTIFICATE-----'
    }
    $Pem | Set-Content -Encoding ascii $CaBundle
}

& $Cmake -S (Join-Path $OrtRoot 'cmake') -B $BuildDir -G 'Unix Makefiles' `
    "-DCMAKE_MAKE_PROGRAM=$QnxMake" `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    "-DCMAKE_TLS_CAINFO=$CaBundle" `
    -Donnxruntime_USE_PREINSTALLED_EIGEN=ON `
    "-Deigen_SOURCE_PATH=$Eigen" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_FLAGS=-D_QNX_SOURCE `
    -DCMAKE_CXX_FLAGS=-D_QNX_SOURCE `
    -Donnxruntime_MINIMAL_BUILD=ON `
    -Donnxruntime_EXTENDED_MINIMAL_BUILD=OFF `
    -Donnxruntime_BUILD_SHARED_LIB=OFF `
    -Donnxruntime_BUILD_UNIT_TESTS=OFF `
    -Donnxruntime_BUILD_BENCHMARKS=OFF `
    -Donnxruntime_ENABLE_PYTHON=OFF `
    -Donnxruntime_DEV_MODE=OFF `
    -Donnxruntime_DISABLE_EXCEPTIONS=ON `
    -Donnxruntime_DISABLE_RTTI=ON `
    -Donnxruntime_ENABLE_LTO=OFF `
    -Donnxruntime_ENABLE_CPUINFO=OFF `
    -DFLATBUFFERS_BUILD_FLATC=OFF `
    -Donnxruntime_USE_XNNPACK=OFF `
    -Donnxruntime_USE_FULL_PROTOBUF=OFF

exit $LASTEXITCODE
