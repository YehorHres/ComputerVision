import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



#1
#список ознак
X =[]
#список міток
y = []

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}



for color_name, bgr in colors.items():
    for i in range(20):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr) + noise, 0, 255)
        X.append(sample)
        y.append(color_name)
#3 розділяємо дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)#X_train - ознаки для навчання  X_test - для перевірки так само y

#4 навчаємо модель
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (20, 50, 50), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))#прибрали дрібні точки
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y + h, x:x + w]

            mean_color = cv2.mean(roi)[:3]
            mean_color = np.array(mean_color).reshape(1, -1)


            label = model.predict(mean_color)[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, label.upper(), (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("color", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()