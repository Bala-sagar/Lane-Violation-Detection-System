from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to("cuda")

    def detect(self, frame):
        results = self.model(frame, conf=0.4, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])

                detections.append((x1, y1, x2, y2, cls))

        return detections

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to("cuda")

    def detect(self, image):
        results = self.model(image, conf=0.4, verbose=False)

        for r in results:
            for box in r.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                return px1, py1, px2, py2

        return None
