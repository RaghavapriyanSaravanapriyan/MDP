# Smart Door Camera

A complete, end-to-end Smart Door Camera web application with a sleek, hacker-esque black and white aesthetic.

## Features
- **Frontend**: A sleek single-page Vanilla JS application that captures webcam frames directly in the browser and sends them to the backend for real-time model training and inference. Includes smooth view transitions, a custom CSS scanner animation for the live feed, and highly polished visual elements.
- **Backend**: A robust Flask API providing endpoints for image registration, model training, and frame recognition.
- **Computer Vision**: Utilizes OpenCV's built-in `LBPHFaceRecognizer` for lightning-fast dynamic face registration and training.

## Installation & Setup

Linux systems often enforce PEP 668 to protect system packages, which means running `pip install` globally will throw an **externally managed environment** error. To resolve this safely, we use a Python Virtual Environment (`venv`).

1. **Create the Virtual Environment**
   Open a terminal in the project directory and run:
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the Virtual Environment & Install Dependencies**
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Start the Server**
   Make sure your virtual environment is still activated, then run the Flask app:
   ```bash
   python backend/app.py
   ```
   
4. **Open the Application**
   Navigate to [http://localhost:5000](http://localhost:5000) in your web browser. When prompted, **allow camera permissions**.

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
