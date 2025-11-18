# DMS Ultra - Advanced Drowsiness Detection System

This project is a real-time driver safety system that detects drowsiness and yawning using computer vision.

## Features
* **Real-time Detection:** Uses MediaPipe FaceMesh to detect Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
* **Alerts:** Plays an alarm sound (`.wav`) when drowsiness or yawning is detected.
* **Cloud Integration:** Sends data to ThingSpeak for remote monitoring.
* **Recording:** Auto-records video clips when alerts are triggered.
* **Dark/Light Mode:** Custom UI built with CustomTkinter.

## How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the script:
    ```bash
    python dms_ultra_advanced.py
    ```