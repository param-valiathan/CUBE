@echo off
rem Shared launcher used by "Launch CUBE.bat", "Launch Analyser.bat" and "Launch Video Explorer.bat".
rem   _cube_launch.bat <script.py> [console]
rem Finds the CUBE Python environment (no hard-coded paths), starts the script minimised, and shows a
rem message box if it closes with an error. With "console" the window stays open so errors can be read.
setlocal EnableExtensions
cd /d "%~dp0"

set "SCRIPT=%~1"
if "%SCRIPT%"=="" set "SCRIPT=cube.py"
set "MODE=%~2"

rem Re-start minimised (normal mode only) so no big black window sits in front of the app.
if /i not "%MODE%"=="console" if not defined CUBE_LAUNCH_MIN (
    set "CUBE_LAUNCH_MIN=1"
    start "CUBE" /min "%~f0" %*
    exit /b 0
)

call :find
if not defined CUBE_PY goto :noenv

if /i "%MODE%"=="console" (
    "%CUBE_PY%" "%SCRIPT%"
    echo.
    echo CUBE has closed. Press any key to close this window.
    pause >nul
    exit /b 0
)

if not exist "CUBE_logs" mkdir "CUBE_logs"
"%CUBE_PY%" "%SCRIPT%" 2> "CUBE_logs\last_launch_errors.txt"
if errorlevel 1 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName PresentationFramework; [void][System.Windows.MessageBox]::Show('CUBE closed with an error.' + [Environment]::NewLine + [Environment]::NewLine + 'Details were saved to:' + [Environment]::NewLine + (Join-Path (Get-Location) 'CUBE_logs\last_launch_errors.txt') + [Environment]::NewLine + [Environment]::NewLine + 'To repair the installation, run install_cube.bat.','CUBE')"
)
exit /b 0

:noenv
echo.
echo  CUBE has not been installed on this computer yet (or its environment was removed).
echo.
echo  Please double-click  install_cube.bat  in this folder, wait for it to finish, then try again.
echo.
pause
exit /b 1

:find
set "CUBE_PY="
if exist "cube_env_path.txt" set /p CUBE_PY=<"cube_env_path.txt"
if defined CUBE_PY if exist "%CUBE_PY%" exit /b 0
set "CUBE_PY="
for %%R in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\mambaforge" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\miniforge3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "C:\miniforge3" "C:\anaconda3" "C:\miniconda3") do (
    for %%E in (CUBE cube) do (
        if not defined CUBE_PY if exist "%%~R\envs\%%E\python.exe" set "CUBE_PY=%%~R\envs\%%E\python.exe"
    )
)
exit /b 0
