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
# Новые имена столбцов

new_column_names = ['номер', 'группа', 'пол', 'возраст', 'стаж', 'процент_разработок', 'ошибки', 'удовл(очки)', 'качество']

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

# Присваиваем новый список имен

data.columns = new_column_names
print("\nТаблица после переименования:")
print(data)

# Список категориальных переменных для преобразования

categorical_columns = ["группа", "пол", "удовл(очки)","качество"]

# Преобразуем каждый столбец в тип 'category'

for col in categorical_columns:
    if col in data.columns:
        data[col] = data[col].astype("category")
        print(f"Столбец '{col}' преобразован в категориальный.")

# Проверяем результат

print("\nТипы данных после преобразования:")
print(data.dtypes)

# Удаление колонки "№п/п", если она существует
# inplace=True означает, что изменения применяются к исходному DataFrame `data`

if "номер" in data.columns:
    data.drop(columns=["номер"], inplace=True)
    print("Столбец 'номер' удален.")

# Рассчитываем количество пропущенных значений в каждом столбце

print("Количество пропущенных значений до обработки:")
print(data.isnull().sum())

# Удаляем строки, в которых есть хотя бы один пропуск

data_cleaned = data.dropna()
# Сравниваем размеры до и после очистки

print(f"\nРазмер исходного набора данных: {data.shape}")
print(f"Размер набора данных после удаления пропусков: {data_cleaned.shape}")

# Если разница невелика, продолжаем работать с очищенным набором

data = data_cleaned

# Создание подмножеств данных по группам (значения 1 и 2 в столбце "группа")

group1 = data[data["группа"] == 1]
group2 = data[data["группа"] == 2]

# Создание подмножеств данных по полу (значения 1 и 2 в столбце "пол")

gender1 = data[data["пол"] == 1]
gender2 = data[data["пол"] == 2]

# Проверяем размеры подвыборок

print(f"Количество наблюдений в группе 1: {len(group1)}")
print(f"Количество наблюдений в группе 2: {len(group2)}")
print(f"Количество наблюдений среди мужчин (пол=1): {len(gender1)}")
print(f"Количество наблюдений среди женщин (пол=2): {len(gender2)}")

# Выводим основные описательные статистики для количественных признаков

print("\nОсновные описательные статистики (для числовых столбцов):")
print(data.describe())

# Статистические характеристики для 4-го столбца

print(data.iloc[:,5].describe())

def calculate_statistics(df, group_name=""):
    """
    Рассчитывает основные описательные статистики
    для количественных переменных.
    """

    def find_mode(series):
        """Возвращает первую моду признака."""
        mode_result = series.mode()

        if not mode_result.empty:
            return mode_result.iloc[0]

        return None

    print("\n" + "-" * 50)
    print(f"СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ: {group_name.upper()}")
    print("-" * 50)

    # Выбираем только количественные столбцы
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        print(f"\n--- {column} ---")

        # Удаляем пропуски только из анализируемого столбца
        column_data = df[column].dropna()

        print(f"Количество наблюдений: {len(column_data)}")
        print(f"Минимальное значение: {column_data.min():.2f}")
        print(f"Максимальное значение: {column_data.max():.2f}")
        print(f"Среднее значение: {column_data.mean():.2f}")
        print(f"Стандартное отклонение: {column_data.std():.2f}")
        print(f"Первый квартиль (Q1): {column_data.quantile(0.25):.2f}")
        print(f"Медиана (Q2): {column_data.median():.2f}")
        print(f"Третий квартиль (Q3): {column_data.quantile(0.75):.2f}")
        print(f"Мода: {find_mode(column_data)}")

        skewness = stats.skew(column_data)
        kurtosis = stats.kurtosis(column_data)

        print(f"Асимметрия: {skewness:.4f}")
        print(f"Эксцесс: {kurtosis:.4f}")

        # Интерпретация асимметрии
        if abs(skewness) < 0.5:
            skew_text = (
                "слабая асимметрия, распределение близко к симметричному"
            )
        elif abs(skewness) < 1:
            skew_text = "умеренная асимметрия"
        else:
            skew_text = "сильная асимметрия"

        # Интерпретация эксцесса
        if kurtosis > 0:
            kurtosis_text = "распределение островершинное"
        elif kurtosis < 0:
            kurtosis_text = "распределение плосковершинное"
        else:
            kurtosis_text = (
                "нулевой эксцесс, как у нормального распределения"
            )

        print(f"Интерпретация: {skew_text}; {kurtosis_text}")

# Рассчитываем статистические характеристики для всей выборки
calculate_statistics(data, "Вся выборка")

# Рассчитываем статистические характеристики для первой группы
calculate_statistics(group1, "Группа 1")

# Рассчитываем статистические характеристики для второй группы
calculate_statistics(group2, "Группа 2")

# Дополнительно: сравнение по полу
calculate_statistics(gender1, "Мужчины (пол=1)")
calculate_statistics(gender2, "Женщины (пол=2)")
