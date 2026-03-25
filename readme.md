# Smart Door Camera

A complete, end-to-end Smart Door Camera web application with a sleek, hacker-esque black and white aesthetic for a college submission.

## Features
- **Frontend**: A sleek single-page Vanilla JS application that captures webcam frames directly in the browser. Features smooth view transitions, a custom CSS scanner animation, and highly polished visual elements.
- **Backend**: A robust Flask API providing endpoints for image registration, model training, and frame recognition.
- **Computer Vision**: Utilizes OpenCV's built-in `LBPHFaceRecognizer` for lightning-fast dynamic face registration and training.

## Installation & Setup

1. **Create the Virtual Environment**
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the Virtual Environment & Install Dependencies**
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Start the Server**
   ```bash
   python backend/app.py
   ```
   
4. **Open the Application**
   Navigate to [http://localhost:5000](http://localhost:5000) in your web browser.

## User Guide

### 1. Registration (Setup Phase)
- Enter an authorized person's name into the input field.
- Click the circular shutter button to capture a photo. Use multiple angles for better accuracy.

### 2. Training Phase
- Click **START TRAINING**. A loading screen with a progress bar will appear while the backend processes the images.

### 3. Demo / Live Recognition Phase
- Step into the frame. If recognized, the system will flip to **UNLOCKED** in green. If unauthorized, it remains **LOCKED** in red.

### 4. Team Members
To update the team members list, edit `data/team.txt` and refresh the page.
