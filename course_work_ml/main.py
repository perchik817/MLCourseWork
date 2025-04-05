import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import warnings

warnings.filterwarnings('ignore')


# --- 1. Загрузка и очистка данных ---
def load_and_clean_data(file_path):
    """Загрузка и предобработка данных"""
    print("Загрузка данных...")
    df = pd.read_excel(file_path, sheet_name="PSS 1", header=13)
    df = df.dropna(how='all').reset_index(drop=True)

    # Фикс заголовков
    if 'Unnamed: 0' in df.columns:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

    # Переименование столбцов
    df.columns = [
        "Sex", "Year", "Experience_of_violence", "Type_of_violence",
        "Sex_of_perpetrator", "Relationship_with_perpetrator",
        "Unit", "Value", "RSE", "95%_MoE", "Lower_CI", "Upper_CI", "Data_flag"
    ]

    # Очистка числовых значений
    def clean_value(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace(",", "").replace("*", "").replace("^", "").strip()
            if x.lower() in ["n.a.", "n.p.", "—", "nan", ""]:
                return np.nan
            try:
                return float(x)
            except:
                return np.nan
        return float(x)

    for col in ["Value", "RSE", "95%_MoE", "Lower_CI", "Upper_CI"]:
        df[col] = df[col].apply(clean_value)

    # Обработка года
    df['Year'] = df['Year'].astype(str).apply(
        lambda x: int(x.split('-')[0]) if '-' in x else int(x) if x.isdigit() else np.nan
    )

    # Удаление строк без целевой переменной
    initial_rows = len(df)
    df = df.dropna(subset=['Value', 'Year'])
    final_rows = len(df)
    print(f"Удалено {initial_rows - final_rows} строк с отсутствующими значениями Value или Year")
    print(f"Итоговый размер датасета: {final_rows} строк")

    return df


# --- 2. Визуализация данных ---
def plot_data_distributions(df):
    """Графики распределения данных"""
    print("\nАнализ распределения данных:")

    # Основные статистики
    print("\nОсновные статистики числовых полей:")
    print(df[['Value', 'Year', 'RSE', '95%_MoE']].describe().round(2))

    # Категориальные переменные
    print("\nУникальные значения категориальных переменных:")
    cat_cols = ['Sex', 'Experience_of_violence', 'Type_of_violence',
                'Sex_of_perpetrator', 'Relationship_with_perpetrator', 'Unit']
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(10))

    plt.figure(figsize=(20, 15))

    # Распределение по полу
    plt.subplot(3, 3, 1)
    sns.countplot(data=df, x='Sex')
    plt.title('Распределение по полу')

    # Распределение значений
    plt.subplot(3, 3, 2)
    sns.histplot(df['Value'], bins=30, kde=True)
    plt.title('Распределение значений Value')

    # Топ типов насилия
    plt.subplot(3, 3, 3)
    top_types = df['Type_of_violence'].value_counts().head(10)
    sns.barplot(x=top_types.values, y=top_types.index)
    plt.title('Топ-10 типов насилия')
    plt.xlabel('Количество')

    # Boxplot по полу
    plt.subplot(3, 3, 4)
    sns.boxplot(data=df, x='Sex', y='Value')
    plt.title('Распределение Value по полу')

    # Распределение по годам
    plt.subplot(3, 3, 5)
    sns.histplot(df['Year'], bins=30, kde=True)
    plt.title('Распределение по годам')

    # Отношения с агрессором
    plt.subplot(3, 3, 6)
    top_relations = df['Relationship_with_perpetrator'].value_counts().head(10)
    sns.barplot(x=top_relations.values, y=top_relations.index)
    plt.title('Топ-10 отношений с агрессором')
    plt.xlabel('Количество')

    # Корреляционная матрица для числовых данных
    plt.subplot(3, 3, 7)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Корреляционная матрица')
    else:
        plt.text(0.5, 0.5, 'Нет числовых данных для корреляции', ha='center')

    # Опыт насилия
    plt.subplot(3, 3, 8)
    experience_counts = df['Experience_of_violence'].value_counts()
    sns.barplot(x=experience_counts.values, y=experience_counts.index)
    plt.title('Опыт насилия')
    plt.xlabel('Количество')

    # Пол агрессора
    plt.subplot(3, 3, 9)
    perpetrator_counts = df['Sex_of_perpetrator'].value_counts().head(10)
    sns.barplot(x=perpetrator_counts.values, y=perpetrator_counts.index)
    plt.title('Пол агрессора')
    plt.xlabel('Количество')

    plt.tight_layout()
    plt.show()


# --- 3. Подготовка данных для ML ---
def prepare_ml_data(df):
    """Подготовка данных для машинного обучения"""
    print("\nПодготовка данных для ML...")

    # Удаление строк с NaN в целевой переменной
    initial_size = len(df)
    df = df.dropna(subset=['Value'])
    final_size = len(df)
    print(f"Удалено {initial_size - final_size} строк с отсутствующими значениями Value")

    # Разделение признаков и целевой переменной
    X = df.drop(columns=['Value'])
    y = df['Value']

    # Категориальные и числовые признаки
    categorical_cols = ['Sex', 'Experience_of_violence', 'Type_of_violence',
                        'Sex_of_perpetrator', 'Relationship_with_perpetrator', 'Unit']
    numeric_cols = ['Year']

    # Пайплайн для предобработки
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Применение преобразований
    X_processed = preprocessor.fit_transform(X)

    print(f"\nРазмерность данных после обработки:")
    print(f"Признаки: {X_processed.shape}")
    print(f"Целевая переменная: {y.shape}")

    # Анализ целевой переменной
    print("\nАнализ целевой переменной (Value):")
    print(f"Минимум: {y.min():.2f}")
    print(f"Максимум: {y.max():.2f}")
    print(f"Среднее: {y.mean():.2f}")
    print(f"Медиана: {y.median():.2f}")
    print(f"Стандартное отклонение: {y.std():.2f}")

    return X_processed, y, preprocessor


# --- 4. Обучение и оценка моделей ---
def train_and_evaluate(X, y):
    """Обучение и сравнение моделей"""
    print("\nОбучение моделей...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")

    # 1. RandomForest с оптимизацией
    print("\nОптимизация RandomForest...")
    rf = RandomForestRegressor(random_state=42)
    rf_params = {
        'n_estimators': randint(50, 300),
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=20, cv=3,
                                   scoring='neg_mean_squared_error', random_state=42)
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_test)
    print(f"Лучшие параметры RandomForest: {rf_search.best_params_}")

    # 2. XGBoost с оптимизацией
    print("\nОптимизация XGBoost...")
    xgb = XGBRegressor(random_state=42)
    xgb_params = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=20, cv=3,
                                    scoring='neg_mean_squared_error', random_state=42)
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_
    xgb_pred = xgb_best.predict(X_test)
    print(f"Лучшие параметры XGBoost: {xgb_search.best_params_}")

    # 3. Stacking (RF + XGBoost)
    print("\nОбучение Stacking модели...")
    estimators = [('rf', rf_best), ('xgb', xgb_best)]
    stack_model = StackingRegressor(estimators=estimators,
                                    final_estimator=XGBRegressor(random_state=42))
    stack_model.fit(X_train, y_train)
    stack_pred = stack_model.predict(X_test)

    # Оценка моделей
    def evaluate(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"\n{name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2: {r2:.4f}")
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print("\nОценка моделей на тестовых данных:")
    metrics = {
        'RandomForest': evaluate("RandomForest", y_test, rf_pred),
        'XGBoost': evaluate("XGBoost", y_test, xgb_pred),
        'Stacking': evaluate("Stacking", y_test, stack_pred)
    }

    # Визуализация результатов
    plt.figure(figsize=(15, 5))

    # График реальных vs предсказанных значений
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, rf_pred, alpha=0.3, label='RandomForest')
    plt.scatter(y_test, xgb_pred, alpha=0.3, label='XGBoost')
    plt.scatter(y_test, stack_pred, alpha=0.3, label='Stacking')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title('Реальные vs Предсказанные')
    plt.legend()

    # График важности признаков для RandomForest
    if hasattr(rf_best, 'feature_importances_'):
        plt.subplot(1, 3, 2)
        importances = rf_best.feature_importances_
        indices = np.argsort(importances)[-10:]  # Топ-10 признаков
        plt.title('Важность признаков (RandomForest)')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
        plt.xlabel('Относительная важность')

    # График метрик
    plt.subplot(1, 3, 3)
    pd.DataFrame(metrics).T.plot(kind='bar', ax=plt.gca())
    plt.title('Сравнение метрик моделей')
    plt.xticks(rotation=45)
    plt.ylabel('Значение метрики')

    plt.tight_layout()
    plt.show()

    return rf_best, xgb_best, stack_model, metrics


# --- 5. Основной пайплайн ---
def main(file_path):
    # Загрузка данных
    print("=" * 50)
    print("Начало анализа данных о насилии")
    print("=" * 50)
    df = load_and_clean_data(file_path)

    print("\nПервые 5 строк данных:")
    print(df.head())
    print("\nИнформация о данных:")
    print(df.info())

    # Визуализация
    plot_data_distributions(df)

    # Подготовка данных для ML
    X, y, preprocessor = prepare_ml_data(df)

    # Обучение моделей
    rf_model, xgb_model, stack_model, metrics = train_and_evaluate(X, y)

    print("\n" + "=" * 50)
    print("Итоговые результаты:")
    print("=" * 50)
    print("\nИтоговые метрики моделей:")
    for model_name, scores in metrics.items():
        print(f"\n{model_name}:")
        print(f"  MAE: {scores['MAE']:.4f}")
        print(f"  RMSE: {scores['RMSE']:.4f}")
        print(f"  R2: {scores['R2']:.4f}")

    # Сравнение моделей
    best_model = min(metrics.items(), key=lambda x: x[1]['RMSE'])
    print(f"\nЛучшая модель: {best_model[0]} с RMSE = {best_model[1]['RMSE']:.4f}")

    return {
        'data': df,
        'models': {
            'RandomForest': rf_model,
            'XGBoost': xgb_model,
            'Stacking': stack_model
        },
        'metrics': metrics,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    file_path = r"data/AIHW-FDSV-all-data-download.xlsx"
    results = main(file_path)