from ultralytics import YOLO
import cv2

# Load fire and smoke model
model = YOLO("fire_smoke.pt")

# Check model classes
print("Model Classes:", model.names)

# Test an image
image_path = "test.jpg"

results = model.predict(
    source=image_path,
    conf=0.25,
    verbose=False
)

# Display detection results
for result in results:
    print("Detected Objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            model.names[class_id],
            round(confidence * 100, 2),
            "%"
        )

    annotated_image = result.plot()

    cv2.imshow("Fire and Smoke Detection", annotated_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()