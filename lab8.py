import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Загрузка данных
dataset = pd.read_csv('Position_Salaries.csv')

# Разделение данных на матрицу признаков и зависимую переменную
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Вывод данных
print("Матрица признаков:")
print(X[:5])
print("Зависимая переменная:")
print(y[:5])

# Обучение линейной регрессии
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Полиномиальные признаки
degree = 10
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Обучение полиномиальной регрессии
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Предсказания для новых данных
X_new = [[6.5]]
y_pred_linear = linear_reg.predict(X_new)
y_pred_poly = poly_reg.predict(poly_features.transform(X_new))
print("Прогноз линейной регрессии:", y_pred_linear)
print("Прогноз полиномиальной регрессии:", y_pred_poly)

# Визуализация результатов
plt.figure(1)
plt.scatter(X, y, color='red')
plt.plot(X, linear_reg.predict(X), color='blue')
plt.title('Зависимость зарплаты от уровня')
plt.xlabel('Уровень')
plt.ylabel('Зарплата')

plt.figure(2)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, poly_reg.predict(poly_features.transform(X_grid)), color='blue')
plt.title('Полиномиальная регрессия')
plt.xlabel('Уровень')
plt.ylabel('Зарплата')

plt.show()
