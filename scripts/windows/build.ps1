# build.ps1 - Windows build (Vulkan backend, the original bit-exact route).
# Requires: Visual Studio 2022 (or Build Tools), git, CMake. tools/ via fetch_tools.ps1.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
Set-Location $root
if (-not (Test-Path "tools/Vulkan-Headers")) { & "$root\scripts\windows\fetch_tools.ps1" }
cmake -B build -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j 8
New-Item -ItemType Directory -Force -Path build\shaders | Out-Null
Get-ChildItem shaders\vulkan-glsl\*.comp | ForEach-Object {
  & tools\glslang\bin\glslangValidator.exe -V $_.FullName -o ("build\shaders\" + $_.BaseName + ".spv")
}
Write-Host "build complete: build\ (opendlss.exe, dlss5vk.exe, opendlss-tests.exe)"
