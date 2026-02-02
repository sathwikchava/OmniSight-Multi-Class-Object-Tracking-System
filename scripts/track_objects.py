import torch
import cv2
import os
from ultralytics import YOLO

# Define paths
# Define paths
MODEL_PATH = os.path.join("models", "yolov8n.pt")  # Ensure this path is correct
VIDEO_PATH = os.path.join("videos", "sample3.mp4")  # Change this to the uploaded video file
OUTPUT_PATH = os.path.join("outputs", "sample2_output.mp4")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load YOLOv8 model
model = YOLO(MODEL_PATH)
# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Video file not found or cannot be opened: {VIDEO_PATH}")

# Get video properties
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        
        break
    # Perform object detection
    results = model(frame)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = result.names[int(box.cls[0].item())]  # Class label
            
            label = f"{cls} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save frame to output video
    out.write(frame)

    # Show frame (optional)
    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracking completed! Output saved at: {OUTPUT_PATH}")
