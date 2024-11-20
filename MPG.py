import pandas as pd
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Загрузка данных
data = pd.read_csv("6_lab/MPG.txt", sep="\t")

# Проверяем данные
print(data.head())

# Преобразование категориальных данных (если есть)
encoder = OneHotEncoder(sparse_output=False)
countries_encoded = encoder.fit_transform(data[['Country']])
countries_df = pd.DataFrame(countries_encoded, columns=encoder.get_feature_names_out(['Country']))
data = pd.concat([data, countries_df], axis=1).drop(columns=['Country'])

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['MPG'])
y = data['MPG']

# Нормализация числовых признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=10)

# Создание модели нейросети
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Один выход для регрессии
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=70, validation_split=0.3, batch_size=64, verbose=1)

# Оценка модели
loss, mae = model.evaluate(X_test, y_test, verbose=2)
print(f"Средняя абсолютная ошибка на тестовых данных: {mae:.2f}")

# Предсказание и визуализация
y_pred = model.predict(X_test)

# Построение графика
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Линия идеального совпадения
plt.xlabel('Истинное значение MPG')
plt.ylabel('Предсказанное значение MPG')
plt.title('Сравнение истинных и предсказанных значений')
plt.show()

# График обучения
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss на обучении')
plt.plot(history.history['val_loss'], label='Loss на валидации')
plt.xlabel('Эпохи')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('График обучения модели')
plt.show()
