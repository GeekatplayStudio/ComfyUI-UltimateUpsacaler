@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD="

if exist "%~dp0..\..\python_embeded\python.exe" (
    set "PYTHON_CMD=%~dp0..\..\python_embeded\python.exe"
) else if exist "%~dp0..\..\python_embedded\python.exe" (
    set "PYTHON_CMD=%~dp0..\..\python_embedded\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
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