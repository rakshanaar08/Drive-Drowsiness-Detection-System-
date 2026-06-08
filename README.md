
# Driver Drowsiness Detection System 🚗💤

A professional-grade, real-time Python prototype for monitoring driver fatigue and alertness using computer vision. This system uses its own webcam to analyze facial landmarks and detect signs of drowsiness, yawning, and lack of attention.

## ✨ Key Features

- **Personalized EAR (Eye Aspect Ratio)**: Adaptive calibration that handles different eye shapes (including small-eye/East-Asian adaptation).
- **PERCLOS Monitoring**: Tracks the percentage of eye closure over a sliding time window (60s).
- **Advanced Yawn Detection (MAR)**: Monitors vertical mouth enlargement, invariant to horizontal stretching (smiling).
- **3D Head Pose tracking**: Detects head tilting (pitch/roll) and nodding patterns using 3D pose estimation.
- **Blink-Based Liveness Detection**: Prevents spoofing attempts using static photos by monitoring natural eye transitions.
- **Interactive Configuration Panel**: Adjust thresholds, alarm intervals, and mute settings live via a Tkinter interface.
- **Audio-Visual Alerts**: Immediate red-border HUD warnings and audible beeps (with `winsound` fallback for Windows).

## 🛠️ Technology Stack

- **MediaPipe (FaceMesh)**: For robust 468-point 3D landmark extraction.
- **OpenCV**: For video capture and real-time HUD rendering.
- **Pygame**: For multi-platform audible alerts.
- **Tkinter**: For the graphical configuration panel.
- **NumPy & SciPy**: For numerical analysis and signal variance calculations.

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed (Python 3.12 is fully supported).

### 2. Install Dependencies
```bash
pip install opencv-python mediapipe numpy scipy pygame
```

### 3. Setup Model
The system uses the modern MediaPipe Tasks API. Ensure `face_landmarker.task` is in the project directory (the script will look for it automatically).

## 🎮 How to Use

1. **Run the Script**:
   ```bash
   python drowsiness_detector.py
   ```
2. **Calibration (Crucial)**:
   On startup, keep your eyes open and look directly at the camera for **5 seconds**. The system will calibrate your baseline eye opening state.
3. **Indicators**:
   - **Green**: Safe state.
   - **Yellow**: Approaching threshold (warning).
   - **Red**: Alert triggered.
4. **Commands**:
   - `Q`: Quit the application.
   - `C`: Re-trigger manual EAR calibration.

## 🔍 Alerts Explained

- **EAR Alert**: Triggered if eyes stay closed for more than 1 second.
- **PERCLOS Alert**: Triggered if the driver's eyes are closed for more than 20% of the last minute.
- **Yawn (MAR) Alert**: Triggered if the mouth opens vertically past the threshold.
- **Pose Alert**: Triggered if the head tilts or nods (gated by eye drowsiness to reduce false positives).
- **NO BLINK Alert**: Anti-spoofing mechanism that triggers if no eye transitions (blinks) are detected for 20 seconds.

## ⚖️ Disclaimer
This system is a prototype designed for research and educational purposes. It should not be used as a primary safety device in real driving conditions. Always stay alert and take breaks when tired.
