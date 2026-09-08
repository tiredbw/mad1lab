# Импорт необходимых библиотек с псевдонимом
import pandas as pd
import numpy as np

# Импорт модуля из библиотеки
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка внешнего вида графиков в Jupyter Notebook (если используется)
# %matplotlib inline
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)

# Загрузка данных из файла
# sep=';' указывает, что столбцы разделены точкой с запятой (часто используется в CSV)
# encoding="cp1251" указывает кодировку для корректного отображения русского языка
data = pd.read_csv("./v_6.csv", sep=';', encoding="cp1251")

# Преобразуем данные в таблицу:
data = pd.DataFrame(data)
# Выводим первые 5 строк таблицы, чтобы увидеть данные
print("Первые 5 строк данных:")
print(data.head())

# Выводим основную информацию о таблице: типы данных, количество ненулевых значений (non-null) в каждом столбце:
print("\nИнформация о структуре данных:")
print(data.info())

# Для адресации к конкретному столбцу указывается имя столбца или индекс:
print("Информация о признаке «возраст»:")
print(data['Age'].info())

# Обращение к столбцу номер 3 (признак «возраст»).
# ОБРАТИТЕ ВНИМАНИЕ! Нумерация строк и столбцов начинается с нуля.
print(data.iloc[:,3].info())

# Выводим названия всех столбцов
print("\nНазвания столбцов:")
print(data.columns.tolist())