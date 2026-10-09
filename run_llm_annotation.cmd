@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv is not installed or is not available on PATH.
    echo Install uv, reopen the terminal, and run this command again.
    exit /b 1
)

if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%CD%\.uv-cache"
if not defined UV_PYTHON_INSTALL_DIR set "UV_PYTHON_INSTALL_DIR=%CD%\.uv-python"

echo Ensuring Python 3.12 is available through uv...
uv python install 3.12 --no-bin --no-registry
if errorlevel 1 exit /b %errorlevel%

echo Starting resumable Ollama consensus annotation...
uv run --python 3.12 python -m corpus.annotate_with_ollama %*
exit /b %errorlevel%
