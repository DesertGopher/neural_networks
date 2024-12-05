import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# Имя модели (можно поменять на вашу)
model_name = "Helsinki-NLP/opus-mt-en-ru"
print("Загрузка модели и токенайзера...")
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Модель и токенайзер загружены.")

# Подготовка данных
print("Подготовка данных...")
texts = [
    "Hello, how are you?",
    "The weather is great today.",
    "I love programming in Python!"
]
labels = [
    "Привет, как дела?",
    "Сегодня отличная погода.",
    "Я люблю программировать на Python!"
]
print("Исходные тексты и метки подготовлены.")

# Токенизация данных
print("Токенизация данных...")
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
outputs = tokenizer(labels, padding=True, truncation=True, return_tensors="tf")
print(f"Токены входных текстов: {inputs['input_ids']}")
print(f"Токены меток: {outputs['input_ids']}")

# Создание входных данных для модели
print("Создание датасета...")
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": outputs["input_ids"][:, :-1]  # Сдвиг для обучения
    },
    outputs["input_ids"][:, 1:]  # Целевые данные
)).batch(2)
print("Датасет создан.")

# Компиляция модели
print("Компиляция модели...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5))
print("Модель скомпилирована.")

# Обучение
print("Начало обучения...")
model.fit(train_dataset, epochs=5)
print("Обучение завершено.")

# Сохранение модели и токенизатора
save_dir = "my_translation_model"
print(f"Сохранение модели и токенизатора в папку: {save_dir}...")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Модель и токенизатор успешно сохранены.")


# Пример генерации текста с сохраненной моделью
def translate_text(input_text):
    print(f"Перевод текста: {input_text}")

    # Загрузка модели и токенизатора
    print("Загрузка сохраненной модели и токенизатора...")
    loaded_model = TFAutoModelForSeq2SeqLM.from_pretrained(save_dir)
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
    print("Модель и токенизатор загружены.")

    # Генерация текста
    print("Генерация перевода...")
    input_ids = loaded_tokenizer.encode(input_text, return_tensors="tf")
    output_ids = loaded_model.generate(input_ids)
    translation = loaded_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Перевод завершен: {translation}")
    return translation


# Проверка перевода
input_text = "I am learning to build neural networks."
output_text = translate_text(input_text)
print(f"Input: {input_text}")
print(f"Output: {output_text}")
