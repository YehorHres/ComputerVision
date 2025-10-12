import cv2
import numpy as np


img = cv2.imread("images/figury.jpg")
img_copy = img.copy()
img = cv2.GaussianBlur(img, (5, 5), 5)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([113, 24, 0])
upper = np.array([179, 255, 255])
mask = cv2.inRange(img, lower, upper)
img = cv2.bitwise_and(img, img, mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        perimetr = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)


        if M["m00"] != 0:

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2)
        compatcness = round((4 * np.pi * area) / (perimetr ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetr, True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Quadratic"
        elif len(approx) > 5:
            shape = "Star"
        elif len(approx) > 6:
            shape = "Circle"
        else:
            shape = "inshe"
        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 1)
        cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(img_copy, f'{shape}', (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(img_copy, f'S:{int(area)}, P:{int(perimetr)}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 0), 1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compatcness}', (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 0), 1)
cv2.imshow("UNCE", img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()