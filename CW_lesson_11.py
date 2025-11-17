import pandas as pd #робота з Csv таблицями
import numpy as np #математичі операції
import tensorflow as tf #нейронка
from tensorflow import keras #розширерння до tensorflow
from tensorflow.keras import layers #створення шарів в нейронці
from sklearn.preprocessing import LabelEncoder #текстові мітки в числа
import matplotlib.pyplot as plt #графіки будувати


#2 працюємо з csv файлами
df = pd.read_csv('data/figures.csv')
# print(df.head())

encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

#3 мітки й ознаки, визначаємо стовпчики
X = df[["area", "perimeter", "corners"]]
y = df['label_enc']

#4 створюємо модель
model = keras.Sequential([
    layers.Input(shape=(3, )),
    layers.Dense(8, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(8, activation='softmax'),
])

#5 компіляція моделей
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X, y, epochs=300, verbose=0)

#6 візуалізація навчання
plt.plot(history.history['loss'], label = "Втрати")
plt.plot(history.history['accuracy'], label = "Точність")
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title("Процес начання моделі")
plt.legend()
plt.show()

#7 тестування
test = np.array([[25, 20, 0]])
pred = model.predict(test)
print(f'імовірність кожного класу {pred}')
print(f'Модель визначила {encoder.inverse_transform([np.argmax(pred)])}')