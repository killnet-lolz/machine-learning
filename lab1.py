import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
dt = np.dtype("f8,f8,f8,f8,U30")
data2 = np.genfromtxt("iris.data", delimiter=",", dtype=dt)

# Извлечение данных по признакам
sepal_length = data2['f0']
sepal_width = data2['f1']
petal_length = data2['f2']
petal_width = data2['f3']

# Визуализация данных
plt.figure(1)
plt.plot(sepal_length[:50], sepal_width[:50], 'ro', label='Setosa')
plt.plot(sepal_length[50:100], sepal_width[50:100], 'g^', label='Versicolor')
plt.plot(sepal_length[100:150], sepal_width[100:150], 'bs', label='Virginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.figure(2)
plt.plot(sepal_length[:50], petal_length[:50], 'ro', label='Setosa')
plt.plot(sepal_length[50:100], petal_length[50:100], 'g^', label='Versicolor')
plt.plot(sepal_length[100:150], petal_length[100:150], 'bs', label='Virginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')

plt.figure(3)
plt.plot(sepal_length[:50], petal_width[:50], 'ro', label='Setosa')
plt.plot(sepal_length[50:100], petal_width[50:100], 'g^', label='Versicolor')
plt.plot(sepal_length[100:150], petal_width[100:150], 'bs', label='Virginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')

plt.figure(4)
plt.plot(petal_length[:50], petal_width[:50], 'ro', label='Setosa')
plt.plot(petal_length[50:100], petal_width[50:100], 'g^', label='Versicolor')
plt.plot(petal_length[100:150], petal_width[100:150], 'bs', label='Virginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.show()
