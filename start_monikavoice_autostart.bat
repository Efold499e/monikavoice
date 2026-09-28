@echo off
rem monikavoice: boot-autostart launch (same as normal; window-less pythonw).
cd /d "%~dp0"
if exist "D:\anaconda3\pythonw.exe" (
  start "monikavoice" /MIN "D:\anaconda3\pythonw.exe" monikavoice.py
) else (
  start "monikavoice" /MIN pythonw.exe monikavoice.py
)
