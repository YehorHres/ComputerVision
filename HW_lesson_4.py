import cv2
import numpy as np

img = cv2.imread("images/1.png")



img_copy = img.copy()
img_copy_color = img.copy()
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 4)
img_copy = cv2.equalizeHist(img_copy)#підсилення контрату
img_copy = cv2.Canny(img_copy, 50, 300)

contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#external -крайні та зовнішні контури(останні) CHAIN_APPROX_SIMPLE - точки по яким ми можемо визначитьи контури елементу

#малювання контурів прямокутнику та тексту
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 120: #фільтр шуму
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), ( 0, 255, 0), 2)
        text_y  = y - 5 if y -5 >10 else y + 15
        text = f'x:{x}, y:{y}, S:{int(area)}'
        cv2.putText(img_copy_color, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# cv2.imshow("men", img)
cv2.imshow("men_copy", img_copy)
cv2.imshow("men_copy_color", img_copy_color)
cv2.waitKey(0)
cv2.destroyAllWindows()