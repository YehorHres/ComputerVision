import cv2
import numpy as np

img = np.zeros((400, 600, 3), np.uint8)
img[:] = 200, 230, 255
cv2.rectangle(img, (10, 10), (590, 390), (161, 161, 161), 3)
qr = cv2.imread("images/qrcode.png")
qr = cv2.resize(qr, (100, 100))
img[210:310, 430:530] = qr
photo = cv2.imread("images/img.jpg")
photo = cv2.resize(photo, (120, 120))
size = cv2.resize(photo, (120, 120))
cv2.putText(img, "Yehor Hres", (185, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
img[40:160, 40:160] = photo

cv2.putText(img, "Computer vision student", (185, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (161, 161, 161), 2)


cv2.putText(img, "Email: yehorhres07@gmail.com", (185, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(img, "Phone: +380688061030", (185, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


cv2.putText(img, "30/12/1999", (185, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.putText(img, "OpenCV Business Card", (125, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

cv2.imshow("businesscard", img)
cv2.imwrite("business_card.png", img)

cv2.waitKey(0)
cv2.destroyAllWindows()