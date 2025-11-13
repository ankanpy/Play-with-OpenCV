import cv2
from ultralytics import YOLO

# --- Configuration ---
# MODEL_NAME = 'yolo11n.pt'  # Fastest, good for basic webcam
MODEL_NAME = 'yolo11m.pt'  # Balanced speed and accuracy (recommended)
# MODEL_NAME = 'yolo11l.pt'  # Slower, but highest accuracy

# Set the classes you want to track: 0 is 'person', 2 is 'car'
CLASSES_TO_TRACK = [0, 2] 

# Set a confidence threshold
CONF_THRESHOLD = 0.3
# --- End Configuration ---


# Load the YOLOv11 model
model = YOLO(MODEL_NAME)

# Open a connection to the local webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if not success:
        print("Error: Failed to capture frame.")
        break

    # Run YOLOv11 tracking on the frame
    # We pass our custom tracker config and filter classes
    results = model.track(
        frame,
        persist=True,                # Remember tracks between frames
        # classes=CLASSES_TO_TRACK,    # Filter for people and cars
        conf=CONF_THRESHOLD ,
        tracker= 'botsort.yaml',  # Use BoTSORT for high accuracy
    )

    # Get the annotated frame with bounding boxes and track IDs
    # results[0].plot() returns the frame with all annotations drawn
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv11 Detection & Tracking", annotated_frame)

    # Check for the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()