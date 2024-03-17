import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Загрузка данных
dataset = pd.read_csv('Data5.csv')

# Отображение первых строк данных
dataset.head()

# Определение матрицы признаков и зависимой переменной
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Матрица признаков:")
print(X)
print("Зависимая переменная:")
print(y)

# Импьютация пропущенных значений с помощью SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

print("Матрица признаков без пропущенных значений:")
print(X)

# Кодирование категориальной зависимой переменной
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print("Зависимая переменная после кодирования:")
print(y)

# Создание копии "грязного" объекта с пропусками и не закодированными категориями
X_dirty = X.copy()
print("Копия 'грязного' объекта:")
print(X_dirty)

# Преобразование признаков с помощью ColumnTransformer
transformers = [
    ('onehot', OneHotEncoder(), [0]),
    ('imp', SimpleImputer(), [1, 2])
]

ct = ColumnTransformer(transformers)

X_transformed = ct.fit_transform(X_dirty)
print("Размер преобразованных данных:")
print(X_transformed.shape)
print("Преобразованные данные:")
print(X_transformed)

# Преобразование многомерного массива в DataFrame
X_data = pd.DataFrame(
    X_transformed,
    columns=['C1', 'C2', 'C3', 'Age', 'Salary']
)
print("Преобразованные данные в DataFrame:")
print(X_data)
