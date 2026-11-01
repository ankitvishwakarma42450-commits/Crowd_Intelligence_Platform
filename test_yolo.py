from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt")

image_path = Path(__file__).with_name("x4.jpeg")
results = model(str(image_path))

for result in results:
    result.show()