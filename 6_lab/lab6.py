import pandas as pd
import numpy as np
from keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Загрузка данных
data = pd.read_csv("MPG.txt", sep="\t")

# Преобразование категориальных данных (если есть)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
countries_encoded = encoder.fit_transform(data[['Country']])
countries_df = pd.DataFrame(countries_encoded,
                            columns=encoder.get_feature_names_out(['Country']),
                            index=data.index)
data = pd.concat([data, countries_df], axis=1).drop(columns=['Country'])

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['MPG'])
y = data['MPG']

# Нормализация числовых признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Создание функциональной модели (с увеличенной глубиной)
input_layer = layers.Input(shape=(X_train.shape[1],))
hidden1 = layers.Dense(128, activation='relu')(input_layer)
hidden2 = layers.Dense(64, activation='relu')(hidden1)
hidden3 = layers.Dense(32, activation='relu')(hidden2)
output_layer = layers.Dense(1)(hidden3)  # Один выход для регрессии

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=100, validation_split=0.15, batch_size=64, verbose=1)

# Оценка модели
loss, mae = model.evaluate(X_test, y_test, verbose=2)
print(f"Средняя абсолютная ошибка на тестовых данных: {mae:.2f}")

# Предсказание
y_pred = model.predict(X_test)

# Построение графика предсказаний
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Предсказания')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Идеал')  # Линия идеального совпадения
plt.xlabel('Истинное значение MPG')
plt.ylabel('Предсказанное значение MPG')
plt.title('Сравнение истинных и предсказанных значений')
plt.legend()
plt.show()

# Анализ ошибок (ошибка = истина - предсказание)
errors = y_test - y_pred.flatten()

# График плотности ошибок
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Ошибка (Истинное значение - Предсказание)')
plt.ylabel('Частота')
plt.title('Гистограмма ошибок')
plt.grid()
plt.show()

# Вывод метрик
print(f"Средняя ошибка: {np.mean(errors):.2f}")
print(f"Медианная ошибка: {np.median(errors):.2f}")
print(f"Стандартное отклонение ошибки: {np.std(errors):.2f}")
