import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random

# Load YOLOv11 model
model = YOLO("models/best.pt")  # Replace with your actual model path

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Class labels (adjust based on your training)
CLASS_NAMES = {0: "Ball", 2: "Player", 3: "Referee"}

# Track ID to color mapping
track_colors = {}

# Load input video
video_path = "D:/New folder (2)/15sec_input_720p.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()
else:
    print("✅ Video opened successfully.")

while True:
    success, frame = cap.read()
    if not success:
        break

    print("📹 Reading frame...")

    # Run YOLO detection
    results = model(frame)
    detections = results[0].boxes

    detections_for_sort = []
    if detections is not None:
        for box in detections:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())

            if cls in [2, 3] and conf > 0.4:  # Filter: only players and referees
                detections_for_sort.append((xyxy, conf, CLASS_NAMES.get(cls, str(cls))))

    # Deep SORT tracking update
    tracks = tracker.update_tracks(detections_for_sort, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        class_name = track.get_det_class() or "Unknown"

        # Assign a unique color to each ID
        if track_id not in track_colors:
            track_colors[track_id] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )

        color = track_colors[track_id]
        label = f"ID {track_id}: {class_name}"

        # Draw bounding box and label
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show result live
    cv2.imshow("Tracked Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting... Done.")
