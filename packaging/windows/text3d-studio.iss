; Inno Setup script for the Windows installer: installs the Nuitka build and registers the .t3d project type.
; Built by the package workflow: iscc /DAppVersion=X.Y.Z /O<dir> /F<name> packaging\windows\text3d-studio.iss
; Paths are relative to this file. Installs per user by default, so no administrator rights are needed.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppExe "text3d-studio.exe"
#define ProgId "Text3DStudio.Project"

[Setup]
AppId={{AD83B697-4ECA-42AD-B091-8456B24D0B4D}
AppName=Text3D Studio
AppVersion={#AppVersion}
AppPublisher=Text3D Studio
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\Text3D Studio
DefaultGroupName=Text3D Studio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes
SetupIconFile=..\..\text3d_studio\resources\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
WizardStyle=modern
Compression=lzma2
SolidCompression=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\build\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Text3D Studio"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\Text3D Studio"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.t3d"; ValueType: string; ValueName: ""; ValueData: "{#ProgId}"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.t3d\OpenWithProgids"; ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#ProgId}"; ValueType: string; ValueName: ""; ValueData: "Text3D Studio Project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#ProgId}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe},0"
Root: HKA; Subkey: "Software\Classes\{#ProgId}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExe}\SupportedTypes"; ValueType: string; ValueName: ".t3d"; ValueData: ""; Flags: uninsdeletekey

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,Text3D Studio}"; Flags: nowait postinstall skipifsilent
