from ultralytics import YOLO
import cv2

from density import calculate_density


# ==============================
# SETTINGS
# ==============================

MODEL_PATH = "yolo11n.pt"

# Maximum people allowed inside the zone
MAX_CAPACITY = 10

# Approximate zone area
ZONE_AREA = 20

# Monitoring zone coordinates
# x1, y1, x2, y2
ZONE = (100, 80, 550, 400)


# ==============================
# LOAD MODEL
# ==============================

model = YOLO(MODEL_PATH)


# ==============================
# OPEN CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Live Crowd Intelligence started.")
print("Press Q to stop.")


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    person_count = 0
    zone_count = 0

    # Zone coordinates
    zx1, zy1, zx2, zy2 = ZONE

    # Draw monitoring zone
    cv2.rectangle(
        frame,
        (zx1, zy1),
        (zx2, zy2),
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "MONITORING ZONE",
        (zx1, zy1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    # ==============================
    # PROCESS DETECTIONS
    # ==============================

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            # Only detect people
            if class_id != 0:
                continue

            person_count += 1

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Center of person
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check if person is inside zone
            if (
                zx1 <= center_x <= zx2
                and
                zy1 <= center_y <= zy2
            ):
                zone_count += 1

                box_color = (0, 0, 255)

            else:
                box_color = (0, 255, 0)

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
                box_color,
                2
            )

            # Display ID
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2
            )

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                box_color,
                -1
            )


    # ==============================
    # CALCULATE DENSITY
    # ==============================

    density, status = calculate_density(
        zone_count,
        ZONE_AREA,
        MAX_CAPACITY
    )


    # ==============================
    # DISPLAY INFORMATION
    # ==============================

    cv2.putText(
        frame,
        f"Total People: {person_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Zone People: {zone_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Density: {density:.2f}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ==============================
    # SHOW CAMERA
    # ==============================

    cv2.imshow(
        "Crowd Intelligence - Live Analysis",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()

print("Live analysis stopped.")