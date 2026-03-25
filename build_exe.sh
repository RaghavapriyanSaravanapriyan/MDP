#!/bin/bash
echo "Building Standalone Executable for Linux..."

source .venv/bin/activate

# Clean previous builds
rm -rf build dist

# Run PyInstaller
pyinstaller --name SmartDoorCamera_Linux \
            --onefile \
            --add-data "frontend:frontend" \
            --hidden-import "cv2" \
            --hidden-import "numpy" \
            backend/app.py

echo "======================================================"
echo "Build complete! The executable is located at dist/SmartDoorCamera_Linux"
echo "You can run it directly: ./dist/SmartDoorCamera_Linux"
echo "======================================================"
