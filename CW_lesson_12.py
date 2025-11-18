import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.preprocessing import image

#1 завантажуємо файли
train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train',
                                                              image_size = (128, 128),
                                                              batch_size = 32,
                                                              label_mode = 'categorical')

test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test',
                                                             image_size = (128, 128),
                                                             batch_size = 32,
                                                             label_mode = 'categorical')
#2 нормалізація зображень
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x,y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x,y: (normalization_layer(x), y))

model = models.Sequential()

# прості ознаки
model.add(layers.Conv2D(
    filters=32,                # кількість фільтрів
    kernel_size=(3, 3),        # розмір фільтра
    activation='relu',         # функція активації
    input_shape=(128, 128, 3)  # форма вхідного зображення (RGB)
))

model.add(layers.MaxPooling2D((2, 2)))   # зменшуємо карту ознак у 2 рази


#глибші ознаки
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, (3, 3), activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

#компіляція
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#навчання моделі
history = model.fit(train_ds, epochs = 30, validation_data=test_ds)

test_loss, tess_acc = model.evaluate(test_ds)
print('Test loss:', test_loss)
print('Test accuracy:', tess_acc)

class_name = ["cars", "cats", "dogs"]

img = image.load_img('images/ghy.jpg', target_size=(128, 128))

image_array = image.img_to_array(img)
image_array = image_array/255.0
image_array = np.expand_dims(image_array, axis=0)
predictions = model.predict(image_array)

predicted_index = np.argmax(predictions[0])


print(f"Ймовірності по класах: {predictions[0]}", )
print(f"Модель визначила:, {class_name[predicted_index]}")





