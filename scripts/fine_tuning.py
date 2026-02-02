from ultralytics import YOLO
import cv2
import os
import random

# Define paths
# Define paths (assuming script is run from project root)
MODEL_PATH = os.path.join("models", "fine_tuned_model.pt")
VIDEO_PATH = os.path.join("videos", "sample2.mp4")
OUTPUT_PATH = os.path.join("outputs", "sample44_output.mp4")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load the fine-tuned YOLOv8 model
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

# Function to generate a random color
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Initialize a dictionary to store random colors for each class
class_colors = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection using the fine-tuned model
    results = model(frame)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = result.names[int(box.cls[0].item())]  # Class label
            
            # Assign a random color for each class if not already assigned
            if cls not in class_colors:
                class_colors[cls] = random_color()
            
            color = class_colors[cls]  # Get the random color for the class
            
            label = f"{cls} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
