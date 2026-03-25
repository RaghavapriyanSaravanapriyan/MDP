@echo off
echo ===========================================
echo Smart Door Camera - Windows Build Script
echo ===========================================

:: Check if python is in path
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH. Please install Python.
    pause
    exit /b
)

echo [1/4] Creating Virtual Environment...
python -m venv .venv

echo [2/4] Installing Dependencies...
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

echo [3/4] Building Portable Executable (SmartDoorCamera.exe)...
pyinstaller --name "SmartDoorCamera" ^
            --onefile ^
            --add-data "frontend;frontend" ^
            --hidden-import "cv2" ^
            --hidden-import "numpy" ^
            backend/app.py

echo [4/4] Done!
echo The portable file is in "dist/SmartDoorCamera.exe"
pause
