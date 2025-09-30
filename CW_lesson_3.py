import cv2
import numpy as np


img = np.zeros((512, 512, 3), np.uint8)
#rgb = bgr
# img[:] = 123, 232, 145 залити все

# img[100:150, 200:280] = 123, 232, 145#прописати відступи y:x

cv2.rectangle(img, (100, 100), (200, 200), (123, 232, 145), 3)


cv2.line(img, (100, 100), (200, 200), (123, 232, 145), 3)
print(img.shape)
cv2.line(img, (0, img.shape[0]//2), (img.shape[1], img.shape[0]//2), (123, 232, 145))
cv2.line(img, (img.shape[1]//2,0), (img.shape[1]//2, img.shape[0]), (123, 232, 145))

cv2.circle(img, (200,200), 20, (123, 232, 145), 3)

cv2.putText(img, "Yehor Hres", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (123, 232, 145))

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()