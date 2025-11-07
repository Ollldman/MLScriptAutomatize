import pytest
import pandas as pd
import numpy as np
from ModelForge.modules.data_analysis.data_statistics import compute_descriptive_statistics
from ModelForge.modules.data_loading_to_dataFrame.load_from_static import load_from_csv
from ModelForge.settings import Settings


@pytest.fixture
def diabetes_df(app_settings: Settings):
    # Загружаем реальный датасет diabetes.csv
    df = load_from_csv(app_settings.TEST_DATASETS_PATH+"diabetes.csv")
    if df is None:
        raise FileNotFoundError("diabetes.csv не найден или не загружен через load_from_csv")
    return df


def test_compute_descriptive_statistics_all_numeric(diabetes_df):
    # Тестируем вычисление статистик для всех числовых колонок
    stats = compute_descriptive_statistics(diabetes_df)
    numeric_cols = diabetes_df.select_dtypes(include=[np.number]).columns.tolist()

    assert set(stats.keys()) == set(numeric_cols)
    for col in numeric_cols:
        assert "mean" in stats[col]
        assert "std" in stats[col]
        assert "percentiles" in stats[col]
        assert "50%" in stats[col]["percentiles"]  # медиана


def test_compute_descriptive_statistics_subset(diabetes_df):
    # Тестируем вычисление статистик для подмножества колонок
    subset_cols = ["Age", "BMI", "Diabetes_012"]
    stats = compute_descriptive_statistics(diabetes_df, columns=subset_cols)

    assert set(stats.keys()) == set(subset_cols)
    for col in subset_cols:
        assert "mean" in stats[col]
        assert "std" in stats[col]
        assert "percentiles" in stats[col]


def test_compute_descriptive_statistics_custom_percentiles(diabetes_df):
    # Тестируем с кастомными процентилями
    percentiles = [0.05, 0.5, 0.95]
    stats = compute_descriptive_statistics(diabetes_df, columns=["BMI"], percentiles=percentiles)

    assert "5%" in stats["BMI"]["percentiles"]
    assert "50%" in stats["BMI"]["percentiles"]
    assert "95%" in stats["BMI"]["percentiles"]


def test_compute_descriptive_statistics_exclude_mode(diabetes_df):
    # Тестируем отключение вычисления mode
    stats = compute_descriptive_statistics(diabetes_df, columns=["Diabetes_012"], include_all=False)

    assert "mode" not in stats["Diabetes_012"]


def test_compute_descriptive_statistics_compare_with_pandas_describe(diabetes_df):
    # Сравниваем mean, std, min, max, percentiles с результатом df.describe()
    col = "BMI"
    stats = compute_descriptive_statistics(diabetes_df, columns=[col], percentiles=[0.25, 0.5, 0.75])
    desc = diabetes_df[[col]].describe(percentiles=[0.25, 0.5, 0.75])

    assert stats[col]["mean"] == pytest.approx(desc[col]["mean"])
    assert stats[col]["std"] == pytest.approx(desc[col]["std"])
    assert stats[col]["min"] == pytest.approx(desc[col]["min"])
    assert stats[col]["max"] == pytest.approx(desc[col]["max"])
    assert stats[col]["percentiles"]["50%"] == pytest.approx(desc[col]["50%"])


def test_compute_descriptive_statistics_no_numeric_columns():
    # Тестируем ошибку при отсутствии числовых колонок
    df = pd.DataFrame({
        'cat1': ['A', 'B', 'C'],
        'cat2': ['X', 'Y', 'Z']
    })
    with pytest.raises(ValueError, match="No numeric columns"):
        compute_descriptive_statistics(df)


def test_compute_descriptive_statistics_invalid_column():
    # Тестируем ошибку при указании несуществующей колонки
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    with pytest.raises(ValueError, match="not found"):
        compute_descriptive_statistics(df, columns=['A', 'nonexistent'])


def test_compute_descriptive_statistics_with_nan():
    # Тестируем корректную обработку NaN
    df = pd.DataFrame({
        'values': [1.0, 2.0, np.nan, 4.0, 5.0]
    })
    stats = compute_descriptive_statistics(df)

    expected_mean = (1 + 2 + 4 + 5) / 4  # 3.0
    assert stats['values']['mean'] == pytest.approx(expected_mean)
    assert stats['values']['count'] == 4  # без NaN


def test_compute_descriptive_statistics_single_value():
    # Тестируем колонку с одним значением
    df = pd.DataFrame({
        'single': [42.0]
    })
    stats = compute_descriptive_statistics(df)

    assert stats['single']['mean'] == 42.0
    assert stats['single']['std'] == 0.0  # std от одного значения = 0
    assert stats['single']['min'] == 42.0
    assert stats['single']['max'] == 42.0


def test_compute_descriptive_statistics_empty_df():
    # Тестируем пустой DataFrame
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="No columns"):
        compute_descriptive_statistics(df)