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

categorical_columns = ["группа", "пол", "качество"]

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

# Диаграмма рассеяния: Возраст и Стаж работы

plt.figure(figsize=(10, 6)) # задается полотно размером 10 на 6
sns.scatterplot(        #   инициализация диаграммы
    data=data,          #   источник данных
    x="возраст",        #   параметр признак
    y="стаж",    #   параметр признак
    hue="группа",       #   параметр разделения точек (по цвету)
    palette="viridis",  #   Настройка палитры
    s=50,               #   Размер точек графика
    alpha=0.7           #   прозрачность точек графика
)

plt.title("Диаграмма рассеяния: Возраст и Стаж работы") # Название диаграммы
plt.xlabel("Возраст")   # Подпись оси абсцисс
plt.ylabel("Стаж работы")   # Подпись оси ординат
plt.grid(True)  #   Включение сетки
plt.show()  #   Отображение графики

# Радиальная диаграмма для "качество документирования"

plt.figure(figsize=(8, 8))  # задается полотно размером 8 на 8
counts = data["качество"].value_counts() # Извлекает данные из столбца, получается Series, где индексы - это категории качества, а значения - их частоты
print(counts)
plt.pie(    # Строит круговую диаграмму
    counts, # данные для построения (количества по категориям)
    labels=counts.index,    # подписи для секторов (названия категорий)
    autopct="%1.1f%%",      # отображает проценты на диаграмме с одним знаком после запятой
    startangle=90,          # начинает построение с 90 градусов (сверху)
    colors=sns.color_palette("Blues", len(counts))  # задает цветовую палитру "Blues"
)

plt.title("Качество документирования (радиальная диаграмма)") # Название диаграммы
plt.show()  #   Отображение графики

# Категориальная радиальная диаграмма по "качество документирования" в зависимости от группы

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))   # Создает полотно с двумя подграфиками рядом друг с другом (1 строка, 2 столбца)
# subplot_kw=dict(polar=True) - ключевой параметр: создает радиальные (полярные) координаты вместо декартовых
# fig - общее полотно, axes - массив из двух осей для радиальных графиков

# Для группы 1 и группы 2

for i, group in enumerate([group1, group2], start=1):
    counts = group["качество"].value_counts().sort_index()  # Подсчитывает частоты уникальных значений в столбце "качество" для текущей группы
    axes[i-1].bar(                  # Строит радиальные столбцы на соответствующем подграфике
        x=range(len(counts)),       # позиции столбцов по кругу (в радианах)
        height=counts,              # высота столбцов (значения частот)
        width=0.6,                  # ширина столбцов в радианах
        bottom=0.0,                 # основание столбцов (в центре диаграммы)
        color=sns.color_palette("viridis", len(counts)),    # цветовая палитра "viridis"
        alpha=0.7                   # прозрачность
    )
    axes[i-1].set_title(f"Группа {i}: Качество документирования")   # Устанавливает заголовок для каждого подграфика с номером группы
    # Настраивает метки на оси: позиции и подписи (категории качества документирования)
    axes[i-1].set_xticks(range(len(counts)))
    axes[i-1].set_xticklabels(counts.index)
    axes[i-1].grid(True)    # Добавляет сетку на радиальную диаграмму

plt.tight_layout()  # Автоматически регулирует отступы между подграфиками
plt.show()  #   Отображение графика

# Столбиковая диаграмма: Средняя степень удовлетворенности заказчика (балльная оценка)

# Создаем DataFrame для средних значений степени удовлетворенности заказчика

quality_data = pd.DataFrame([
    {
        "Категория": "Группа 1",
        "Средняя степень": group1["удовл(очки)"].mean()
    },
    {
        "Категория": "Группа 2",
        "Средняя степень": group2["удовл(очки)"].mean()
    },
    {
        "Категория": "Пол м",
        "Средняя степень": gender1["удовл(очки)"].mean()
    },
    {
        "Категория": "Пол ж",
        "Средняя степень": gender2["удовл(очки)"].mean()
    }
])

# Столбиковая диаграмма: Средняя степень удовлетворенности заказчика (балльная оценка)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=quality_data,  # Данные
    x="Категория",      # Подпись
    y="Средняя степень",# Подпись
    hue="Категория",    # Цвет столбцов по категориям
    palette="viridis",  # Настройка палитры
    legend=False
)
plt.title("Средняя степень удовлетворенности заказчика (балльная оценка)")
plt.xlabel("")
plt.ylabel("Средняя степень удовлетворенности")
plt.grid(True)
plt.show()

# Диаграмма размаха: % выполнения разработок в срок

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=data,
    x="группа",
    y="процент_разработок",
    hue="группа",
    palette="viridis",
    legend=False
)
plt.title("Диаграмма размаха: % выполнения разработок в срок")
plt.xlabel("Группа")
plt.ylabel("% выполнения разработок в срок")
plt.grid(True)
plt.show()

# Гистограммы для всех количественных признаков

# Список количественных столбцов, которые требуется исследовать

numeric_columns = ["возраст", "стаж", "процент_разработок", "ошибки", "удовл(очки)"]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))    # Создает полотно с 6 подграфиками в виде сетки 3×2
axes = axes.flatten()                               # Выравнивает массив осей из формата 3×2 в одномерный массив из 6 элементов

for i, column in enumerate(numeric_columns):        # Организует цикл по всем числовым столбцам
    sns.histplot(data[column], ax=axes[i], kde=True, color="skyblue")   # Строит гистограмму для текущего числового столбца
    # data[column] - данные для построения
    # ax=axes[i] - указывает, на каком подграфике строить
    # kde=True - добавляет Kernel Density Estimation (гладкую линию оценки плотности)
    # color="skyblue" - задает голубой цвет гистограммы
    axes[i].set_title(f"Гистограмма: {column}") # Устанавливает заголовок для каждого подграфика с названием переменной
    axes[i].set_xlabel("")                      # Убирает подпись оси X для более компактного вида
    axes[i].grid(True)                           # Добавляет сетку на график

# Скрывает последний пустой подграфик, так как количественных признаков пять

axes[-1].set_visible(False)
plt.tight_layout()
plt.show()

# Матричный график по всем количественным признакам

# Создает матричный график (pairplot) - сетку графиков "каждый с каждым"
# data=data[numeric_columns] - использует только числовые столбцы из DataFrame
# diag_kind="kde" - на диагональных графиках использует гладкие кривые распределения
# plot_kws задает прозрачность, размер и обводку точек
# height=2.5 - высота каждого отдельного подграфика в дюймах

sns.pairplot(
    data=data[numeric_columns],
    diag_kind="kde",
    plot_kws={"alpha": 0.6, "s": 50, "edgecolor": "k"},
    height=2.5
)
plt.suptitle("Матричный график", y=1.02)
# y=1.02 - немного поднимает заголовок выше для лучшего визуального размещения
plt.show()

# Проверка распределения признака "стаж" на соответствие нормальному закону

# Проверяем первую и вторую группы с помощью критериев Шапиро-Уилка и Андерсона-Дарлинга

for group, group_name in [(group1, "Группа 1"), (group2, "Группа 2")]:
    experience_data = group["стаж"].dropna()

    print(f"\nПРОВЕРКА НОРМАЛЬНОСТИ: {group_name.upper()}")

    # Критерий Шапиро-Уилка

    shapiro_stat, shapiro_p = stats.shapiro(experience_data)
    print("Критерий Шапиро-Уилка:")
    print(f"  Статистика = {shapiro_stat:.4f}, p-value = {shapiro_p:.6f}")

    if shapiro_p > 0.05:
        print("  -> Вывод: Не отвергаем нулевую гипотезу. Данные согласуются с нормальным законом.")
    else:
        print("  -> Вывод: Отвергаем нулевую гипотезу. Данные НЕ согласуются с нормальным законом.")

    print("-" * 40)

    # Критерий Андерсона-Дарлинга

    anderson_result = stats.anderson(experience_data, dist="norm")
    print("Критерий Андерсона-Дарлинга:")
    print(f"  Статистика = {anderson_result.statistic:.4f}")
    print("  Критические значения для уровня значимости alpha:")

    for i in range(len(anderson_result.critical_values)):
        significance_level = anderson_result.significance_level[i]
        critical_value = anderson_result.critical_values[i]

        if anderson_result.statistic < critical_value:
            conclusion = "согласуются"
        else:
            conclusion = "НЕ согласуются"

        print(
            f"  {significance_level}%: {critical_value:.3f} | "
            f"Данные {conclusion} с нормальным законом"
        )

    print("-" * 40)

# Гистограммы и графики Q-Q для первой и второй групп

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(group1["стаж"], kde=True, color="green", bins=8, ax=axes[0, 0])
axes[0, 0].set_title("Группа 1: гистограмма стажа")

stats.probplot(group1["стаж"], plot=axes[0, 1])
axes[0, 1].set_title("Группа 1: график Q-Q")

sns.histplot(group2["стаж"], kde=True, color="blue", bins=8, ax=axes[1, 0])
axes[1, 0].set_title("Группа 2: гистограмма стажа")

stats.probplot(group2["стаж"], plot=axes[1, 1])
axes[1, 1].set_title("Группа 2: график Q-Q")

plt.tight_layout()
plt.show()

# Корреляционный анализ количественных переменных для первой группы

numeric_data_1 = group1[numeric_columns]
print(f"\nКоличественные переменные: {list(numeric_data_1.columns)}")

corr_pearson_1 = numeric_data_1.corr(method="pearson")
corr_spearman_1 = numeric_data_1.corr(method="spearman")
corr_kendall_1 = numeric_data_1.corr(method="kendall")

print("\nГРУППА 1")
print("\nМатрица корреляций Пирсона:")
print(corr_pearson_1.round(3))
print("\nМатрица корреляций Спирмена:")
print(corr_spearman_1.round(3))
print("\nМатрица корреляций Кендалла:")
print(corr_kendall_1.round(3))

# Корреляционный анализ количественных переменных для второй группы

numeric_data_2 = group2[numeric_columns]

corr_pearson_2 = numeric_data_2.corr(method="pearson")
corr_spearman_2 = numeric_data_2.corr(method="spearman")
corr_kendall_2 = numeric_data_2.corr(method="kendall")

print("\nГРУППА 2")
print("\nМатрица корреляций Пирсона:")
print(corr_pearson_2.round(3))
print("\nМатрица корреляций Спирмена:")
print(corr_spearman_2.round(3))
print("\nМатрица корреляций Кендалла:")
print(corr_kendall_2.round(3))

# Тепловые карты коэффициентов корреляции для первой группы

sns.set_theme(style="white", font_scale=1.2)

# Тепловая карта для корреляции Пирсона

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_pearson_1,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f",
    linewidths=0.5
)
plt.title("Тепловая карта: Корреляция Пирсона, группа 1")
plt.tight_layout()
plt.show()

# Тепловая карта для корреляции Спирмена

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_spearman_1,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f",
    linewidths=0.5
)
plt.title("Тепловая карта: Корреляция Спирмена, группа 1")
plt.tight_layout()
plt.show()

# Тепловая карта для корреляции Кендалла

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_kendall_1,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f",
    linewidths=0.5
)
plt.title("Тепловая карта: Корреляция Кендалла, группа 1")
plt.tight_layout()
plt.show()
