import cv2
import numpy as np


img = cv2.imread('images/img.jpg')
size = cv2.resize(img, (612, 512))

cv2.rectangle(size, (220, 180), (320, 355), (123, 232, 145), 3)
cv2.putText(size, "Yehor Hres", (225, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


cv2.imshow('Face', size)





cv2.waitKey(0)
cv2.destroyAllWindows()