from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Live tracking started.")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # YOLO detection + tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    person_count = 0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            # Person class
            class_id = int(box.cls[0])

            if class_id != 0:
                continue

            person_count += 1

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display ID
            cv2.putText(
                frame,
                f"Person ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display current count
    cv2.putText(
        frame,
        f"People: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Crowd Intelligence - Live Tracking",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Tracking stopped.")