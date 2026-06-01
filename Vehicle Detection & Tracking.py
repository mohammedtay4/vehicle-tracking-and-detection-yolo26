from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# Path to input video
VIDEO_PATH = r""

# Path to output video
OUTPUT_PATH = r""

# YOLO model weight
MODEL_PATH = "yolo26n.pt"

#vehicle classes
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# Detection and tracking settings
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.5
TRACKER_CFG = "bytetrack.yaml"

# How many recent points to keep for each vehicle trail
TRAIL_LENGTH = 30


def main():
    # Load the YOLO model
    model = YOLO(MODEL_PATH)

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    # Read video properties to save the output correctly
    fps = cap.get(cv2.CAP_PROP_FPS)
    

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up the output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # Store trail points for each tracked object ID
    track_history = defaultdict(list)

    print("Press Q in the video window to stop.")

    while True:
        # Read the next frame
        success, frame = cap.read()
        if not success:
            break

        #tracking
        results = model.track(
            frame,
            persist=True,
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
            tracker=TRACKER_CFG,
            verbose=False
        )[0]

        # Draw the model's own boxes and labels
        annotated = results.plot()

        # Make sure there are detected boxes and track IDs
        if results.boxes is not None and results.boxes.id is not None:
            # Get bounding boxes in center-x, center-y, width, height format
            boxes_xywh = results.boxes.xywh.cpu().numpy()

            # Get track IDs for each detected object
            track_ids = results.boxes.id.int().cpu().tolist()

            # Get class IDs for each detected object
            class_ids = results.boxes.cls.int().cpu().tolist()

            # Get class name mapping
            names = results.names

            # Process each tracked object
            for box, track_id, class_id in zip(boxes_xywh, track_ids, class_ids):
                label = names[class_id]

                # Ignore anything that is not a vehicle
                if label not in VEHICLE_CLASSES:
                    continue

                # Box values
                x, y, w, h = box

                # Store the center point for this object's trail
                track = track_history[track_id]
                track.append((float(x), float(y)))

                # Keep trail length limited
                if len(track) > TRAIL_LENGTH:
                    track.pop(0)

                # Draw the motion trail if we have at least two points
                if len(track) > 1:
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [points], False, (255, 255, 255), 2)

                # Draw the label and ID above the box
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                cv2.putText(
                    annotated,
                    f"{label} ID:{track_id}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )


        cv2.imshow("Vehicle Detection and Tracking", annotated)


        writer.write(annotated)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Saved output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()