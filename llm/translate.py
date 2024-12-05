from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# Загрузка модели и токенизатора
model_name = "my_translation_model"
print("Загрузка сохраненной модели и токенайзера...")
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Модель и токенайзер загружены.")


def translate_text(text):
    """Перевод текста с использованием загруженной модели."""
    print(f"Перевод текста: {text}")

    # Токенизация входного текста
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)

    # Генерация перевода
    print("Генерация перевода...")
    outputs = model.generate(input_ids=inputs["input_ids"])

    # Декодирование перевода
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Перевод завершен.")
    return translated_text


if __name__ == "__main__":
    translated = translate_text(input("Введите текст на английском: "))
    print(f"Перевод: {translated}")
