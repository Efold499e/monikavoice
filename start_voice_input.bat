@echo off
rem monikavoice: normal launch (console-free). Falls back to PATH pythonw when anaconda is absent.
cd /d "%~dp0"
if exist "%LocalAppData%\Programs\Python" (
  where pythonw.exe >nul 2>nul && start "monikavoice" pythonw.exe monikavoice.py && goto :eof
)
if exist "D:\anaconda3\pythonw.exe" (
  start "monikavoice" "D:\anaconda3\pythonw.exe" monikavoice.py
) else (
  start "monikavoice" pythonw.exe monikavoice.py
)
