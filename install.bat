@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD="

for %%P in (
    "%~dp0..\..\python_embeded\python.exe"
    "%~dp0..\..\python_embedded\python.exe"
    "%~dp0..\..\..\python_embeded\python.exe"
    "%~dp0..\..\..\python_embedded\python.exe"
    "%~dp0..\..\.venv\Scripts\python.exe"
    "%~dp0..\..\venv\Scripts\python.exe"
    "%~dp0.venv\Scripts\python.exe"
) do (
    if not defined PYTHON_CMD if exist "%%~fP" (
        set "PYTHON_CMD=%%~fP"
    )
)

if defined PYTHON_CMD (
    "%PYTHON_CMD%" install.py --with-flux-assets %*
    exit /b %errorlevel%
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 install.py --with-flux-assets %*
    exit /b %errorlevel%
)

where python >nul 2>nul
if %errorlevel%==0 (
    python install.py --with-flux-assets %*
    exit /b %errorlevel%
)

echo Python was not found in PATH.
echo Install Python or use the Python environment that runs ComfyUI, then try again.
exit /b 1