from keras import models
import numpy as np
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps

model = models.load_model('digit_recognition_model_split.h5')


class DigitRecognize:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторная 5")
        window_width = 500
        window_height = 500
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        self.root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=20, pady=20)

        self.label = tk.Label(self.frame, text="Нарисуйте цифру", font=("Arial", 18, "bold"))
        self.label.grid(row=0, column=0, columnspan=2, pady=10)

        self.canvas = tk.Canvas(self.frame, width=280, height=280, bg="black", relief="solid", bd=2)
        self.canvas.grid(row=1, column=0, columnspan=2, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = tk.Button(self.frame, text="Определить", command=self.predict_digit, font=("Arial", 14), width=20, bg="#4CAF50", fg="white")
        self.predict_button.grid(row=2, column=0, pady=5)

        self.clear_button = tk.Button(self.frame, text="Очистка", command=self.clear_canvas, font=("Arial", 14), width=20, bg="#f44336", fg="white")
        self.clear_button.grid(row=2, column=1, pady=5)

        self.quit_button = tk.Button(self.frame, text="Выход", command=self.root.quit, font=("Arial", 14), width=20, bg="#9E9E9E", fg="white")
        self.quit_button.grid(row=3, column=0, columnspan=2, pady=10)

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

        # Предсказание с использованием модели
        prediction = model.predict(digit_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        messagebox.showinfo("Результат", f"Цифра: {predicted_digit}\n"
                                         f"Уверенность: {confidence:.2f}%")


if __name__ == "__main__":
    import tkinter as tk
    root = tk.Tk()
    app = DigitRecognize(root)
    root.mainloop()
