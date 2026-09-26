; Inno Setup script for OpenDLSS-NR-MetalFX (Windows x64)
[Setup]
AppName=OpenDLSS-NR-MetalFX
AppVersion=1.0.0
DefaultDirName={autopf}\OpenDLSS
DefaultGroupName=OpenDLSS
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=opendlss-windows-x64-setup
Compression=lzma2
SolidCompression=yes

[Files]
Source: "build\Release\opendlss.exe"; DestDir: "{app}\bin"
Source: "build\Release\dlss5vk.exe"; DestDir: "{app}\bin"
Source: "build\Release\opendlss-tests.exe"; DestDir: "{app}\bin"
Source: "build\shaders\*.spv"; DestDir: "{app}\share\opendlss\shaders"
Source: "include\opendlss\opendlss.h"; DestDir: "{app}\include"

[Icons]
Name: "{group}\OpenDLSS CLI"; Filename: "{app}\bin\opendlss.exe"
