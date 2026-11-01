from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Path to your video
video_path = "input/videos/crowd.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("ERROR: Could not open the video.")
    print("Make sure your video is located at:")
    print("input/videos/crowd.mp4")
    exit()

print("Video started successfully.")
print("Press Q to stop the video.")

while True:

    # Read one frame
    ret, frame = cap.read()

    # Stop when video ends
    if not ret:
        print("Video finished.")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Start person count
    person_count = 0

    # Process detections
    for result in results:

        for box in result.boxes:

            # Get detected object's class
            class_id = int(box.cls[0])

            # COCO class ID 0 = person
            if class_id == 0:
                person_count += 1

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    # Display number of people
    cv2.putText(
        frame,
        f"People: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show video
    cv2.imshow(
        "Crowd Intelligence - Person Counter",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Program stopped.")