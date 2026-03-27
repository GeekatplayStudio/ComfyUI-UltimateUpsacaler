@echo off
setlocal

cd /d "%~dp0"

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