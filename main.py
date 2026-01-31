import cv2
import os
import numpy as np
from datetime import datetime

from detector import VehicleDetector, PlateDetector
from config import CLASS_NAMES, LEFT_LANE_ALLOWED, RIGHT_LANE_ALLOWED, VIDEO_SOURCE
from db import init_db, insert_violation
import easyocr

# ================= OCR =================
ocr_reader = easyocr.Reader(['en'], gpu=True)

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

# ================= LANE CONFIG =================
# Adjust these two points ONCE for your camera
# LANE_LINE = [
#     (300, 720),   # bottom point (x, y)
#     (640, 400)    # top point (x, y)
# ]

def get_center_lane_line(frame_w, frame_h):
    x = frame_w // 2
    return [(x, frame_h), (x, 0)]

def get_lane(cx, cy, line):
    (x1, y1), (x2, y2) = line
    val = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
    return "left" if val > 0 else "right"



# ================= INIT =================
seen_violations = set()
BASE_VIOLATION_DIR = "violations"

vehicle_detector = VehicleDetector("models/vehicle.pt")
plate_detector = PlateDetector("models/plate.pt")
init_db()

cap = cv2.VideoCapture(VIDEO_SOURCE)

# os.makedirs("violations/small_in_big_lane", exist_ok=True)
# os.makedirs("violations/big_in_small_lane", exist_ok=True)


# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    LANE_LINE = get_center_lane_line(w, h)


    detections = vehicle_detector.detect(frame)

    for x1, y1, x2, y2, cls_id in detections:
        vehicle_type = CLASS_NAMES[cls_id]

        cx = (x1 + x2) // 2
        cy = y2  # bottom of vehicle (touching road)


        lane = get_lane(cx, cy, LANE_LINE)

        violation_type = None

        if lane == "left" and vehicle_type not in LEFT_LANE_ALLOWED:
            violation_type = "heavy_in_normal_lane"

        elif lane == "right" and vehicle_type not in RIGHT_LANE_ALLOWED:
            violation_type = "normal_in_heavy_lane"


        if violation_type:
            # simple vehicle identity (fast, no tracker)
            vehicle_id = f"{cls_id}_{cx//50}_{cy//50}"

            if vehicle_id not in seen_violations:
                seen_violations.add(vehicle_id)

                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                event_dir = os.path.join(BASE_VIOLATION_DIR, ts)
                os.makedirs(event_dir, exist_ok=True)

                # 1️⃣ Save full frame
                cv2.imwrite(f"{event_dir}/frame.jpg", frame)

                # 2️⃣ Save vehicle crop
                vehicle_crop = frame[y1:y2, x1:x2]
                cv2.imwrite(f"{event_dir}/vehicle.jpg", vehicle_crop)

                # 3️⃣ Plate detection + OCR
                plate_text = "UNKNOWN"
                plate_bbox = plate_detector.detect(vehicle_crop)

                if plate_bbox:
                    px1, py1, px2, py2 = plate_bbox
                    plate_img = vehicle_crop[py1:py2, px1:px2]
                    cv2.imwrite(f"{event_dir}/plate.jpg", plate_img)
                    plate_text = read_plate_text(plate_img) or "UNKNOWN"

                # 4️⃣ Metadata
                with open(f"{event_dir}/metadata.txt", "w") as f:
                    f.write(f"date_time: {ts}\n")
                    f.write(f"vehicle_type: {vehicle_type}\n")
                    f.write(f"lane: {lane}\n")
                    f.write(f"violation: {violation_type}\n")
                    f.write(f"plate_number: {plate_text}\n")
                
                insert_violation(
                    timestamp=ts,
                    vehicle_type=vehicle_type,
                    lane=lane,
                    violation_type=violation_type,
                    plate_number=plate_text,
                    event_dir=event_dir
                )



        color = (0, 0, 255) if violation_type else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{vehicle_type} | {lane.upper()}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.line(frame, LANE_LINE[0], LANE_LINE[1], (255, 255, 0), 3)

    cv2.imshow("Lane Violation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
