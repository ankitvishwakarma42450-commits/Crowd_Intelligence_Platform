from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Live camera started.")
print("Press Q to stop.")

while True:

    # Read frame from camera
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    person_count = 0

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            # COCO class 0 = person
            if class_id == 0:

                person_count += 1

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    # Display crowd count
    cv2.putText(
        frame,
        f"People: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show live camera
    cv2.imshow(
        "Crowd Intelligence - Live Camera",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")