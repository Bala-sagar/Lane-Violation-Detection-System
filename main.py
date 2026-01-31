import cv2
import os
import numpy as np
from datetime import datetime

from detector import VehicleDetector, PlateDetector
from config import SMALL_VEHICLES, BIG_VEHICLES, CLASS_NAMES, VIDEO_SOURCE
import easyocr


# ================= OCR =================
ocr_reader = easyocr.Reader(['en'], gpu=False)

def read_plate_text(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    results = ocr_reader.readtext(gray)
    if not results:
        return None

    best = max(results, key=lambda x: x[2])
    return "".join(c for c in best[1] if c.isalnum())


# ================= ADAPTIVE LANE =================
def get_lane_center_x(y, frame_w, frame_h):
    """
    Perspective-aware center lane calculation.
    """

    # 🔧 tune once per camera
    top_y = int(frame_h * 0.45)
    bottom_y = int(frame_h * 0.95)

    top_center_x = frame_w // 2
    bottom_center_x = frame_w // 2

    # clamp above / below road
    if y <= top_y:
        return top_center_x
    if y >= bottom_y:
        return bottom_center_x

    alpha = (y - top_y) / (bottom_y - top_y)
    return int(top_center_x * (1 - alpha) + bottom_center_x * alpha)


def draw_lane(frame):
    h, w, _ = frame.shape

    prev_pt = None
    for y in range(0, h, 8):   # FULL HEIGHT
        x = get_lane_center_x(y, w, h)
        curr_pt = (x, y)

        if prev_pt is not None:
            cv2.line(frame, prev_pt, curr_pt, (255, 255, 0), 2)

        prev_pt = curr_pt


# ================= INIT =================
vehicle_detector = VehicleDetector("models/vehicle.pt")
plate_detector = PlateDetector("models/plate.pt")

cap = cv2.VideoCapture(VIDEO_SOURCE)

os.makedirs("violations/small_in_big_lane", exist_ok=True)
os.makedirs("violations/big_in_small_lane", exist_ok=True)


# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    draw_lane(frame)  # 👈 full-height adaptive divider

    detections = vehicle_detector.detect(frame)

    for x1, y1, x2, y2, cls_id in detections:
        vehicle_type = CLASS_NAMES[cls_id]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        lane_x = get_lane_center_x(cy, w, h)
        lane = "left" if cx < lane_x else "right"

        violation_type = None
        if lane == "left" and vehicle_type in BIG_VEHICLES:
            violation_type = "big_in_small_lane"
        elif lane == "right" and vehicle_type in SMALL_VEHICLES:
            violation_type = "small_in_big_lane"

        if violation_type:
            vehicle_crop = frame[y1:y2, x1:x2]
            plate_bbox = plate_detector.detect(vehicle_crop)

            if plate_bbox:
                px1, py1, px2, py2 = plate_bbox
                plate_img = vehicle_crop[py1:py2, px1:px2]

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                plate_text = read_plate_text(plate_img) or "UNKNOWN"

                filename = f"{vehicle_type}_{plate_text}_{ts}.jpg"
                cv2.imwrite(
                    f"violations/{violation_type}/{filename}",
                    plate_img
                )

                with open(
                    f"violations/{violation_type}/{vehicle_type}_{ts}.txt",
                    "w"
                ) as f:
                    f.write(f"plate: {plate_text}\n")
                    f.write(f"lane_violation: {violation_type}\n")
                    f.write(f"time: {ts}\n")

        color = (0, 0, 255) if violation_type else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{vehicle_type}-{lane}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Lane Violation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
