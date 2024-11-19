import numpy as np
from keras import layers, models, datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def add_noise(images, noise_factor=0.3):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


x_train_noisy = add_noise(x_train)


model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    x_train_noisy, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=64
)


test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'\nТочность на тестовых данных: {test_accuracy:.2f}')


plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.legend()
plt.title("Точность на обучении и валидации")
plt.show()


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)


cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))


disp.plot(cmap='viridis', xticks_rotation='vertical')
disp.ax_.set_title("Confusion Matrix")
plt.show()

model.save('improved_digit_recognition_model.h5')
