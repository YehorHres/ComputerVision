import cv2

net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt", "data/MobileNet/mobilenet.caffemodel")#завантажуємо моделі
classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f: #зчитуємо списокт назв
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

image = cv2.imread("images/MobileNet/cat.jpg")

blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))#адаптуємо зображення під модель

net.setInput(blob) #вкладення в мережу підготовлених файлів

preds = net.forward() #вектор імовірності для класів




# знаходемо індекс класу з найбільшою імовірністю
idx = preds[0].argmax()

label = classes[idx] if idx < len(classes) else "Unknown"
conf = float(preds[0][idx]) * 100

print("Class:", label)
print("Likelihood:", conf)


#підписуємо зображення
text = f'{label}: {int(conf)}%'


cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

