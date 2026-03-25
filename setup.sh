#!/bin/bash

echo "Starting Smart Door Camera Setup..."

# Check if .venv exists, if not create it
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        echo "If you are on Ubuntu/Debian, please run: sudo apt install python3-venv"
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

# Activate and install
echo "Installing dependencies from requirements.txt..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo "To start the application, run:"
echo "source .venv/bin/activate"
echo "python backend/app.py"
