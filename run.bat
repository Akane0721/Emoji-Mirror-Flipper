@echo off
rem ===========================================================
rem  Emoji-Mirror-Flipper launcher
rem  Double-click this file. It sets up the venv on first run,
rem  then starts the server and opens your browser.
rem
rem  NOTE: messages here are ASCII on purpose -- .bat files are
rem  decoded using the console code page, so non-ASCII text
rem  would show up as garbage on some machines. All the Chinese
rem  output comes from app.py instead.
rem ===========================================================

setlocal
cd /d "%~dp0"

rem Do NOT add "chcp" here: changing the code page mid-script makes cmd
rem resume reading this file at a wrong byte offset and execute garbage.
rem Python already writes Unicode to the console correctly on its own.

set "VPY=.venv\Scripts\python.exe"

if exist "%VPY%" goto check_deps

echo [setup] Creating virtual environment...
where uv >nul 2>nul
if errorlevel 1 goto venv_python

uv venv .venv
if errorlevel 1 goto fail_venv
goto install

:venv_python
py -3 -m venv .venv
if not errorlevel 1 goto install
python -m venv .venv
if errorlevel 1 goto fail_venv
goto install

:check_deps
"%VPY%" -c "import flask, PIL, numpy" >nul 2>nul
if errorlevel 1 goto install
goto run

:install
echo [setup] Installing dependencies...
where uv >nul 2>nul
if errorlevel 1 goto install_pip
uv pip install --python "%VPY%" -r requirements.txt
if errorlevel 1 goto fail_deps
goto run

:install_pip
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail_deps
goto run

:run
"%VPY%" app.py
goto end

:fail_venv
echo.
echo [ERROR] Could not create a virtual environment.
echo         Install Python 3.9+ and make sure it is on PATH:
echo         https://www.python.org/downloads/
goto end

:fail_deps
echo.
echo [ERROR] Dependency install failed. Check your network connection.
goto end

:end
echo.
pause
