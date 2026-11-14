@echo off
rem One-time setup: creates .venv in this folder and installs the packages.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 -m venv .venv) else (python -m venv .venv)
if not exist ".venv\Scripts\python.exe" (
  echo Could not create the virtual environment. Is Python 3.9 or newer installed?
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo.
echo Training the show-up model and replaying 2016...
".venv\Scripts\python.exe" train_show_model.py train.csv
".venv\Scripts\python.exe" backtest.py
echo.
echo Done. To use the environment in a terminal:  .venv\Scripts\activate
pause
