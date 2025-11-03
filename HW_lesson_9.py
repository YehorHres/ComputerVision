import cv2
import os

# Завантаження моделі
net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt",
                               "data/MobileNet/mobilenet.caffemodel")


classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)


folder = "images/MobileNet"


class_count = {}


for file in os.listdir(folder):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(folder, file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Помилка з файлом: {file}")
        continue

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)),
                                 1.0 / 127.5, (224, 224),
                                 (127.5, 127.5, 127.5))
    net.setInput(blob)
    preds = net.forward()

    idx = preds[0].argmax()
    label = classes[idx]
    conf = float(preds[0][idx]) * 100


    if label in class_count:
        class_count[label] += 1
    else:
        class_count[label] = 1

    text = f'{label}: {int(conf)}%'
    cv2.putText(image, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

    print(f"{file}: {label} ({conf:.2f}%)")
    cv2.imshow("Image", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


print("таблиця")
for label, count in class_count.items():
    print(f"{label} : {count}")
