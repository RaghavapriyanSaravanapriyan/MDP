# Smart Door Camera

A complete, end-to-end Smart Door Camera web application with a sleek, hacker-esque black and white aesthetic.

## Features
- **Frontend**: A sleek single-page Vanilla JS application that captures webcam frames directly in the browser and sends them to the backend for real-time model training and inference. Includes smooth view transitions, a custom CSS scanner animation for the live feed, and highly polished visual elements.
- **Backend**: A robust Flask API providing endpoints for image registration, model training, and frame recognition.
- **Computer Vision**: Utilizes OpenCV's built-in `LBPHFaceRecognizer` for lightning-fast dynamic face registration and training.

## 🚀 Windows Deployment (Handoff)

To provide the client with a single **Portable `.exe`**, follow these simple steps on any Windows machine:

1. **Copy this folder** to the Windows machine.
2. **Double-click `build_windows.bat`**.
3. A single file called **`SmartDoorCamera.exe`** will be generated in the `dist/` folder.
4. **Give that file to the client.** When they open it, it will automatically launch the server and the web dashboard.

---

## 🛠 Manual Setup (Linux)
If you wish to run the project directly on Linux without an executable:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python backend/app.py
```

## User Guide

### 1. Registration (Setup Phase)
- Enter an authorized person's name into the input field.
- Click the circular shutter button to capture a photo.
- **Tip**: Click the shutter button multiple times (e.g., 5-10 times) while moving your head slightly to provide the model with multiple angles.
- Change the name to register additional authorized users.

### 2. Training Phase
- Once all authorized people are captured, click **START TRAINING**.
- A loading screen with a simulated progress bar will appear while the backend compiles the photos and trains the model.

### 3. Demo / Live Recognition Phase
- Step into the frame.
- If you are recognized, the right sidebar will display **UNLOCKED** in neon green, and your name will update in the "Last Visited" field.
- If an unrecognized or unauthorized face is on camera, it will flip to **LOCKED** in vivid red.

### 4. Team Members
To update the team members list shown on the Demo page, edit `data/team.txt` and refresh the page.
