# Real-time Face Analysis System

This Python application provides real-time face detection, eye tracking, and head pose estimation using OpenCV and MediaPipe. It's designed to work robustly in various lighting conditions and can handle partial occlusions.

## Features

- Real-time face detection
- Precise eye location tracking
- Head pose estimation (pitch, yaw, roll angles)
- Facial landmark visualization
- Works in challenging conditions (low light, partial occlusions)

## Requirements

- Python 3.8+
- Webcam
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python face_analyzer.py
```

- Press 'q' to quit the application
- The application will show:
  - Facial mesh overlay
  - Head pose direction (green line)
  - Real-time measurements for pitch, yaw, and roll angles

## Technical Details

The system uses:
- MediaPipe Face Mesh for robust facial landmark detection
- OpenCV for image processing and pose estimation
- 3D model points for accurate head pose calculation
- Camera matrix estimation for proper 3D-2D point projection

## Notes

- The camera matrix is estimated using default values. For better accuracy, you can calibrate your specific camera.
- The system is optimized for single-face detection and tracking.
- Performance may vary depending on your hardware capabilities. 