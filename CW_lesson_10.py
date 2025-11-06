import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



#1 створимо іункцію для генерації прстих фігур
def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

#список ознак
X =[]
#список міток
y = []

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "pink": (203, 192, 255),
    "purple": (255, 0, 255),
    "orange": (0, 165, 255),
    "brown": (19, 69, 139)
}


shapes = ["circle", "square", "triangle"]

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3] #повертає (b, g, r, alpha), але без alpha, бо [:3]
            features = [mean_color[0], mean_color[1], mean_color[2]]
            X.append(features)
            y.append(f'{color_name}_{shape}')

#3 розділяємо дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)#X_train - ознаки для навчання  X_test - для перевірки так само y

#4 навчаємо модель
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
#5 перевіряємо точність
accuracy = model.score(X_test, y_test)
print(f'Точність моделі: {round(accuracy * 100, 2)}%')

test_image = generate_image((19, 69, 139), "square")
mean_color = cv2.mean(test_image)[:3]
prediction = model.predict([mean_color])
print(f'Передбачення: {prediction[0]}')

cv2.imshow("img", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
