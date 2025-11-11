import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Дані для навчання
X = []
y = []

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "pink": (255, 0, 255),

    "cyan": (255, 255, 0),
    "white": (255, 255, 255)
}

for color_name, bgr in colors.items():
    for i in range(40):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr) + noise, 0, 255)
        X.append(sample)
        y.append(color_name)

# 2. Навчання моделі
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 3. Відеопотік
cap = cv2.VideoCapture(0)

# Буфер для згладження кольору
mean_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (20, 50, 50), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y + h, x:x + w]

            mean_color = cv2.mean(roi)[:3]
            mean_buffer.append(mean_color)
            if len(mean_buffer) > 5:
                mean_buffer.pop(0)

            smoothed_color = np.mean(mean_buffer, axis=0).reshape(1, -1)
            label = model.predict(smoothed_color)[0]


            probs = model.predict_proba(smoothed_color)[0]
            confidence = np.max(probs) * 100

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, f"{label.upper()} {confidence:}%", (x, y - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("color", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
