# Print the path of vcvars64.bat: $env:VCVARS64 if set, else the newest Visual Studio (2022 or later, any
# edition, Build Tools included) that vswhere reports with the C++ x64 toolset.
$ErrorActionPreference = "Stop"
if ($env:VCVARS64 -and (Test-Path $env:VCVARS64)) { $env:VCVARS64; exit 0 }
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
  $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if ($install) {
    $candidate = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $candidate) { $candidate; exit 0 }
  }
}
foreach ($edition in "Enterprise", "Professional", "Community", "BuildTools") {
  foreach ($year in "2022", "2026") {
    $candidate = "C:\Program Files\Microsoft Visual Studio\$year\$edition\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $candidate) { $candidate; exit 0 }
  }
}
throw "vcvars64.bat not found: install Visual Studio 2022 (or the Build Tools) with the C++ x64 toolset, or set VCVARS64"
