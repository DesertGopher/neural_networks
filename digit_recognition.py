import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps

model = tf.keras.models.load_model('5_lab/digit_recognition_model_split.h5')


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание цифр")

        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas.grid(row=1, column=0, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = tk.Button(self.root, text="OK", command=self.predict_digit, font=("Arial", 12))
        self.predict_button.grid(row=2, column=0, pady=5)

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas, font=("Arial", 12))
        self.clear_button.grid(row=3, column=0, pady=5)

        self.image = Image.new("L", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        digit_image = self.image.resize((28, 28))
        digit_image = ImageOps.invert(digit_image)
        digit_array = np.array(digit_image) / 255.0
        digit_array = digit_array.reshape(1, 28, 28)

        prediction = model.predict(digit_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        messagebox.showinfo("Результат", f"Распознанная цифра: {predicted_digit}\n"
                                         f"Уверенность: {confidence:.2f}%")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
