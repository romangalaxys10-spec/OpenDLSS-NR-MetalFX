# Fetch the portable toolchain into tools/ (git-ignored): glslang (release zip), Vulkan-Headers (tagged clone),
# volk (tagged clone), CMake and Ninja (release zips, for Filament and the demo), and the npm modules of the scene
# converter (-Npm). Re-running skips what is already there. No Vulkan SDK is needed: everything loads the driver's
# vulkan-1 through volk (the tool, the demo) or bluevk (Filament).
#   powershell -File scripts\fetch_tools.ps1 [-Npm]
param([switch]$Npm)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tools = Join-Path $root "tools"
New-Item -ItemType Directory -Force $tools | Out-Null

$glslangVersion = "16.6.0"
$vulkanHeadersTag = "v1.4.363"
$volkTag = "vulkan-sdk-1.4.357.0"

# glslang: the Windows x64 release zip (bin/glslang.exe)
if (-not (Test-Path (Join-Path $tools "glslang\bin\glslang.exe"))) {
  $zip = Join-Path $tools "glslang.zip"
  $url = "https://github.com/KhronosGroup/glslang/releases/download/$glslangVersion/glslang-$glslangVersion-windows-x86_64-release.zip"
  Write-Host "downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $zip
  $stage = Join-Path $tools "glslang-unzip"
  Expand-Archive -Path $zip -DestinationPath $stage -Force
  $bin = Get-ChildItem -Path $stage -Recurse -Filter glslang.exe | Select-Object -First 1
  if (-not $bin) { throw "glslang.exe not found in $zip" }
  $top = Split-Path -Parent $bin.DirectoryName   # the zip's root (bin/, include/, lib/), possibly nested one level
  Move-Item -Path $top -Destination (Join-Path $tools "glslang")
  if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
  Remove-Item -Force $zip
}
& (Join-Path $tools "glslang\bin\glslang.exe") --version | Select-Object -First 1

# Vulkan-Headers and volk: shallow tagged clones (only the include tree and volk.c / volk.h are used)
if (-not (Test-Path (Join-Path $tools "Vulkan-Headers\include\vulkan\vulkan.h"))) {
  git clone --depth 1 --branch $vulkanHeadersTag https://github.com/KhronosGroup/Vulkan-Headers.git (Join-Path $tools "Vulkan-Headers")
  if ($LASTEXITCODE -ne 0) { throw "git clone Vulkan-Headers failed" }
}
if (-not (Test-Path (Join-Path $tools "volk\volk.c"))) {
  git clone --depth 1 --branch $volkTag https://github.com/zeux/volk.git (Join-Path $tools "volk")
  if ($LASTEXITCODE -ne 0) { throw "git clone volk failed" }
}
Write-Host ("Vulkan-Headers " + (Select-String -Path (Join-Path $tools "Vulkan-Headers\include\vulkan\vulkan_core.h") -Pattern "#define VK_HEADER_VERSION " | Select-Object -First 1).Line)

# CMake + Ninja (portable zips): the demo renderer (Filament) and the demo itself build with CMake
$cmakeVersion = "3.31.12"
$ninjaVersion = "1.13.2"
if (-not (Test-Path (Join-Path $tools "cmake\bin\cmake.exe"))) {
  $zip = Join-Path $tools "cmake.zip"
  $url = "https://github.com/Kitware/CMake/releases/download/v$cmakeVersion/cmake-$cmakeVersion-windows-x86_64.zip"
  Write-Host "downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $zip
  $stage = Join-Path $tools "cmake-unzip"
  Expand-Archive -Path $zip -DestinationPath $stage -Force
  $exe = Get-ChildItem -Path $stage -Recurse -Filter cmake.exe | Select-Object -First 1
  Move-Item -Path (Split-Path -Parent $exe.DirectoryName) -Destination (Join-Path $tools "cmake")
  if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
  Remove-Item -Force $zip
}
if (-not (Test-Path (Join-Path $tools "ninja\ninja.exe"))) {
  $zip = Join-Path $tools "ninja.zip"
  $url = "https://github.com/ninja-build/ninja/releases/download/v$ninjaVersion/ninja-win.zip"
  Write-Host "downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $zip
  Expand-Archive -Path $zip -DestinationPath (Join-Path $tools "ninja") -Force
  Remove-Item -Force $zip
}
& (Join-Path $tools "cmake\bin\cmake.exe") --version | Select-Object -First 1
Write-Host ("ninja " + (& (Join-Path $tools "ninja\ninja.exe") --version))

# Scene converter modules (optional): tools/gltf/node_modules
if ($Npm) {
  $gltf = Join-Path $tools "gltf"
  New-Item -ItemType Directory -Force $gltf | Out-Null
  Push-Location $gltf
  npm install --no-audit --no-fund --prefix . "@gltf-transform/core@4" "@gltf-transform/extensions@4" "@gltf-transform/functions@4" meshoptimizer draco3dgltf
  Pop-Location
  if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
}
Write-Host "tools ready in $tools"
