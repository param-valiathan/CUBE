@echo off
title CUBE installer
cd /d "%~dp0"
echo.
echo  CUBE installer - this sets up everything CUBE needs. It can take 10-30 minutes
echo  the first time and downloads several GB. You can leave this window open and wait.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_cube.ps1" %*
echo.
echo  Press any key to close this window.
pause >nul
