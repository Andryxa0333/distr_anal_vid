import cv2 as cv
from ultralytics import YOLO

try:
    model = YOLO("yolo11n.pt")
    device = 'cpu'
    model.to(device)
except Exception as e:
    raise RuntimeError(f"Error loading YOLO model: {e}")

image = cv.imread("/home/andr/antonov_proj/input/maxresdefault.jpg")
print(image.shape)
image = cv.resize(image, (640,360),interpolation=cv.INTER_AREA)
print(image.shape)

results = model(image, show=True)

detections = []
for result in results:
    for box in result.boxes:
        detection = {
            "class": int(box.cls),
            "confidence": float(box.conf),
            "box": box.xyxy.tolist()
        }
        detections.append(detection)

print(detections)