# install_windows.ps1 — one-command setup on Windows (D3D12 path).
# Locates Visual Studio via vswhere, configures + builds, generates the demo
# model and runs the smoke test.
#
# Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot

Write-Host "== OpenDLSS-NR MetalFX — Windows installer =="

# 1. locate Visual Studio / Build Tools
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere not found; install Visual Studio 2022 or the Build Tools (C++ workload)"
}
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { Write-Error "no Visual Studio with the C++ toolset found" }
Write-Host "  Visual Studio: $vsPath"

# 2. cmake (bundled with VS, else PATH)
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    $bundled = Get-ChildItem "$vsPath\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -ErrorAction SilentlyContinue
    if ($bundled) { $cmakeCmd = $bundled.FullName } else { Write-Error "cmake not found" }
} else { $cmakeCmd = $cmake.Source }
Write-Host "  cmake: $cmakeCmd"

# 3. build
Set-Location $Here
& $cmakeCmd -B build -DCMAKE_BUILD_TYPE=Release
& $cmakeCmd --build build --config Release -j $env:NUMBER_OF_PROCESSORS

# 4. demo model
python tools\make_demo_weights.py --out models\demo-nr

# 5. smoke test
& build\opendlss-nr.exe info
ctest --test-dir build -C Release --output-on-failure

Write-Host ""
Write-Host "done. try:"
Write-Host "  build\opendlss-nr.exe demo --width 480 --height 270 --scale 2 --model models\demo-nr"
