# Smart Door Camera System

A robust facial recognition security system designed for seamless access control and identity management. This project leverages contemporary web technologies and computer vision algorithms to provide a reliable, end-to-end smart door solution.

## Project Overview

The Smart Door Camera System provides an intuitive interface for managing authorized personnel. Key capabilities include:
- **Live Recognition**: Real-time face detection and matching against a trained database.
- **Dynamic Registration**: On-the-fly capture and enrollment of new users via a web interface.
- **Automated Training**: Efficient model updates using the LBPH (Local Binary Patterns Histograms) algorithm.
- **Responsive Interface**: A modern, high-performance web frontend with integrated feedback and status indicators.

## Technical Stack

- **Backend**: Flask-based REST API
- **Computer Vision**: OpenCV (LBPHFaceRecognizer)
- **Frontend**: Vanilla JavaScript and CSS
- **Data Management**: File-system-based image storage and training data persistence

## Getting Started

Follow these steps to set up and run the application locally:

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Environment Setup
Create and activate a virtual environment to manage dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installation
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Running the Application
Start the Flask server:
```bash
python backend/app.py
```
The application will be accessible at `http://localhost:5000`.

## Operational Workflow

### Phase 1: Registration
Input the name of the individual to be authorized and capture several clear photos from different angles using the interface's capture button.

### Phase 2: Training
Initiate the training process via the dashboard. The system will process the newly captured images and update the recognition model.

### Phase 3: Recognition
Once trained, the system will monitor the camera feed. Authorized personnel will trigger an "UNLOCKED" state, while unrecognized individuals will result in a "LOCKED" state.

## Project Team

- **S.Raghavapriyan**
- **Utkarsh chahal**
- **Ashish paila**
- **Adhisriram RK**
- **Nishanth**
