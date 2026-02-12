import cv2
import time
import csv
import os
import torch
from ultralytics import YOLO
from streamlink import Streamlink


LINE_START = [(950, 430), (1800, 450)]
LINE_FINISH = [(200, 650), (1200, 850)]
DISTANCE_METERS = 15.0

YOUTUBE_URL = 'https://www.youtube.com/watch?v=Lxqcg1qt0XU'
CSV_PATH = 'traffic_speeds.csv'


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def is_below_line(pos, line):
    (x_p, y_p) = pos
    (x1, y1), (x2, y2) = line
    value = (y2 - y1) * (x_p - x1) - (x2 - x1) * (y_p - y1)
    return value < 0


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

entry_times = {}
entry_times_reverse = {}
current_speeds = {}
processed_ids = set()
total_cars_count = 0

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Time', 'Speed_kmh', 'Direction'])

while True:
    ret, frame = cap.read()
    if not ret:
        break


    for _ in range(2):
        cap.grab()

    results = model.track(frame, conf=0.3, imgsz=640, persist=True, classes=[2, 3, 5, 7], verbose=False, device=device)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = int((x1 + x2) / 2), y2
            pos = (cx, cy)


            if is_below_line(pos, LINE_START) and not is_below_line(pos, LINE_FINISH):
                if track_id not in entry_times and track_id not in processed_ids:
                    entry_times[track_id] = time.time()

            if is_below_line(pos, LINE_FINISH) and track_id in entry_times:
                dt = time.time() - entry_times.pop(track_id)
                if dt > 0.3:
                    speed_kmh = round((DISTANCE_METERS / dt) * 3.6, 1)
                    current_speeds[track_id] = speed_kmh
                    processed_ids.add(track_id)
                    total_cars_count += 1
                    with open(CSV_PATH, 'a', newline='') as f:
                        csv.writer(f).writerow([track_id, time.strftime("%H:%M:%S"), speed_kmh, 'Forward'])


            if is_below_line(pos,
                             LINE_FINISH) and track_id not in entry_times_reverse and track_id not in processed_ids:
                entry_times_reverse[track_id] = time.time()

            if not is_below_line(pos, LINE_START) and track_id in entry_times_reverse:
                dt = time.time() - entry_times_reverse.pop(track_id)
                if dt > 0.3:
                    speed_kmh = round((DISTANCE_METERS / dt) * 3.6, 1)
                    current_speeds[track_id] = speed_kmh
                    processed_ids.add(track_id)
                    total_cars_count += 1
                    with open(CSV_PATH, 'a', newline='') as f:
                        csv.writer(f).writerow([track_id, time.strftime("%H:%M:%S"), speed_kmh, 'Reverse'])


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


            display_text = f"ID:{track_id}"
            if track_id in current_speeds:
                display_text += f" | {current_speeds[track_id]} km/h"


            cv2.rectangle(frame, (x1, y1 - 25), (x1 + 220, y1), (0, 255, 0), -1)
            cv2.putText(frame, display_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


    cv2.line(frame, LINE_START[0], LINE_START[1], (255, 255, 0), 3)
    cv2.line(frame, LINE_FINISH[0], LINE_FINISH[1], (0, 0, 255), 3)
    cv2.rectangle(frame, (20, 20), (350, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Total Cars: {total_cars_count}", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    show_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Traffic Radar (Speed on Screen)', show_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()