@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Restore.ps1"
if errorlevel 1 echo Restoration stopped. See the message above.
pause
