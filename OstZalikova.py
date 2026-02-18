import cv2
import time
import csv
import os
import torch
import numpy as np
from ultralytics import YOLO
from streamlink import Streamlink


ZONE_POLYGON = np.array([(900, 400), (1600, 500), (1300, 1000), (50, 750)], np.int32)
PARKING_THRESHOLD = 5.0
YOUTUBE_URL = 'https://www.youtube.com/watch?v=Lxqcg1qt0XU'
CSV_PATH = 'parking_violations.csv'
SCREENSHOTS_DIR = 'violations_screens'

if not os.path.exists(SCREENSHOTS_DIR):
    os.makedirs(SCREENSHOTS_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


mouse_coords = (0, 0)
temp_points = []
drawing_mode = False


def update_mouse(event, x, y, flags, param):
    global mouse_coords, temp_points, ZONE_POLYGON, drawing_mode

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coords = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN and drawing_mode:

        real_x = int(x * (orig_w / 1280))
        real_y = int(y * (orig_h / 720))
        temp_points.append([real_x, real_y])
        if len(temp_points) == 4:
            ZONE_POLYGON = np.array(temp_points, np.int32)
            temp_points = []
            drawing_mode = False


try:
    session = Streamlink()
    streams = session.streams(YOUTUBE_URL)
    stream_url = streams['best'].url
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception as e:
    print(f"Помилка підключення: {e}")
    exit()

model = YOLO("yolov8n.pt").to(device)
zone_entry_times, violation_logged = {}, set()
total_violations_count = 0

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['ID', 'Time', 'Duration_sec', 'Status', 'Screenshot_File'])

cv2.namedWindow('Parking Violation Radar')
cv2.setMouseCallback('Parking Violation Radar', update_mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]


    for _ in range(2):
        cap.grab()


    grid_step = 200
    for x in range(0, orig_w, grid_step):
        cv2.line(frame, (x, 0), (x, orig_h), (128, 128, 128), 2)
        cv2.putText(frame, str(x), (x + 10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 5)
        cv2.putText(frame, str(x), (x + 10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
    for y in range(0, orig_h, grid_step):
        cv2.line(frame, (0, y), (orig_w, y), (128, 128, 128), 2)
        cv2.putText(frame, str(y), (10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 5)
        cv2.putText(frame, str(y), (10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)

 
    results = model.track(frame, conf=0.3, imgsz=640, persist=True, classes=[2, 3, 5, 7], verbose=False, device=device)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = int((x1 + x2) / 2), int(y2)
            is_inside = cv2.pointPolygonTest(ZONE_POLYGON, (float(cx), float(cy)), False) >= 0

            box_color = (0, 255, 0)
            status_text = "Moving"

            if is_inside:
                if track_id not in zone_entry_times:
                    zone_entry_times[track_id] = time.time()
                stay_duration = time.time() - zone_entry_times[track_id]

                if stay_duration > PARKING_THRESHOLD:
                    box_color = (0, 0, 255)
                    status_text = f"STOPPED: {int(stay_duration)}s"

                    if track_id not in violation_logged:
                        total_violations_count += 1
                        timestamp = time.strftime("%H-%M-%S")
                        file_name = f"violation_id_{track_id}_{timestamp}.jpg"

                        crop_img = frame[max(0, y1 - 15):min(orig_h, y2 + 15), max(0, x1 - 15):min(orig_w, x2 + 15)]
                        cv2.imwrite(os.path.join(SCREENSHOTS_DIR, file_name), crop_img)

                        try:
                            with open(CSV_PATH, 'a', newline='') as f:
                                csv.writer(f).writerow(
                                    [track_id, time.strftime("%H:%M:%S"), round(stay_duration, 1), 'Violation',
                                     file_name])
                        except PermissionError:
                            pass
                        violation_logged.add(track_id)
                else:
                    box_color = (0, 255, 255)
                    status_text = f"In Zone: {int(stay_duration)}s"
            else:
                zone_entry_times.pop(track_id, None)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(frame, f"ID:{track_id} {status_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color,
                        2)


    overlay = frame.copy()
    cv2.fillPoly(overlay, [ZONE_POLYGON], (0, 0, 255))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.polylines(frame, [ZONE_POLYGON], True, (0, 0, 255), 4)


    for pt in temp_points:
        cv2.circle(frame, (pt[0], pt[1]), 15, (255, 0, 0), -1)


    real_x = int(mouse_coords[0] * (orig_w / 1280))
    real_y = int(mouse_coords[1] * (orig_h / 720))
    cv2.putText(frame, f"REAL X:{real_x} Y:{real_y}", (real_x + 20, real_y - 20), 0, 1.2, (255, 255, 0), 2)


    cv2.rectangle(frame, (20, 20), (500, 180), (0, 0, 0), -1)
    cv2.putText(frame, f"Violations: {total_violations_count}", (40, 75), 0, 1.5, (255, 255, 255), 3)
    mode_txt = "DRAWING" if drawing_mode else "LIVE"
    cv2.putText(frame, f"MODE: {mode_txt}", (40, 120), 0, 1, (0, 255, 255), 2)
    cv2.putText(frame, "S: Set Zone | R: Reset", (40, 160), 0, 0.8, (200, 200, 200), 1)

    show_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Parking Violation Radar', show_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        drawing_mode = True
        temp_points = []
    elif key == ord('r'):
        drawing_mode = False
        temp_points = []

cap.release()

cv2.destroyAllWindows()
