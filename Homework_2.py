import cv2
img = cv2.imread('images/img.jpg')
size1 = cv2.resize(img, (700,500))
gray = cv2.cvtColor(size1, cv2.COLOR_BGR2GRAY)
c = cv2.Canny(gray, 100,100)
img1 = cv2.imread('images/mail.jpg')
size2 = cv2.resize(img1, (700,500))
gray1 = cv2.cvtColor(size2, cv2.COLOR_BGR2GRAY)
c1 = cv2.Canny(gray1, 100,100)
cv2.imshow('me', c)
cv2.imshow('mail', c1)

cv2.waitKey(0)
cv2.destroyAllWindows()