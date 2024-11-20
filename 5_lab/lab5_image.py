import numpy as np
from keras import models
from PIL import Image, ImageOps


model = models.load_model('digit_recognition_model_split.h5')


def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("L")  # Преобразование в градации серого
        image = image.resize((28, 28))
        image = ImageOps.invert(image)

        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28)

        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        return predicted_digit, confidence
    except Exception as e:
        return f"Ошибка {image_path}: {e}", None


digit, conf = predict_image("zero.bmp")
print(f"Цифра: {digit}, Уверенность: {conf:.2f}%")
