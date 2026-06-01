# Vehicle Detection, Classification, and Tracking with YOLO26

This project is a computer vision system for analyzing traffic video. It detects vehicles in a video stream, classifies them into vehicle categories, tracks them across frames, and visualizes their movement with bounding boxes, labels, and motion trails.

It's an expansion of my earlier final-year project on the vehicle counting and classification system. The earlier version focused on identifying and counting vehicles in traffic footage. This updated version shifts the focus towards detection, classification, and tracking, which makes the system more useful for understanding driver intention and behaviour, and how vehicles move through a scene over time.

The project is built with Python, OpenCV, NumPy, and the Ultralytics YOLO framework.

---

## How It Works

The progam works as follows:

1. A video file is opened with OpenCV.
2. Each frame is sent to the YOLO model.
3. The model detects vehicles and assigns class labels.
4. The tracking system assigns an ID to each detected object.
5. The project stores the center point of each tracked vehicle over time.
6. Those saved points are drawn as a trajectory trail.
7. The annotated frame is shown on screen and saved to a new video file.

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/9a2609bf-4924-4800-8a45-6e85dd814931" />


---

## Technologies Used

- **Python**
- **OpenCV**
- **NumPy**
- **Ultralytics YOLO26**
- **PyTorch**

---

## Requirements

Before running the project, please install the required libraries.
