import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных
dataset = pd.read_csv('Mall_Customers.csv')

# Вывод первых строк данных для проверки
print(dataset.head())

# Выбор признаков для кластеризации
X = dataset.iloc[:, [3, 4]].values

# Использование elbow method для поиска оптимального количества кластеров
from sklearn.cluster import KMeans

# Вычисление Within Cluster Sum of Squares (WCSS) для разного количества кластеров
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Визуализация метода локтя
plt.plot(range(1, 11), wcss)
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()

# Обучение модели K-Means на данных
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Визуализация кластеров
plt.figure(figsize=(8, 6))
for cluster_label in range(5):
    plt.scatter(X[y_kmeans == cluster_label, 0], X[y_kmeans == cluster_label, 1], label=f'Cluster {cluster_label+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Кластеры потребителей')
plt.xlabel('Ежегодный доход (k$)')
plt.ylabel('Баллы (1-100)')
plt.legend()
plt.show()
