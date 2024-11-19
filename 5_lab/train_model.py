import tensorflow as tf
import pandas as pd
from keras import layers, models
from matplotlib import pyplot as plt
import numpy as np


train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

X_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values
X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0


# def add_noise(images, noise_factor=0.3):
#     noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
#     noisy_images = np.clip(noisy_images, 0., 1.)
#     return noisy_images
#
#
# X_train_noisy = add_noise(X_train)

# Определение модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Добавление Dropout для предотвращения переобучения
    layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


def augment(image, label):
    # Случайный поворот
    image = tf.image.random_flip_left_right(image)
    # Случайное изменение яркости
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Случайное изменение контраста
    # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(augment).batch(64).shuffle(1000)

history = model.fit(train_dataset, epochs=20, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)

plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.legend()
plt.show()

model.save('digit_recognition_model_from_csv_with_noise.h5')
