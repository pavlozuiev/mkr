import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

# Налаштування підключення
DB_USER = "student"
DB_PASSWORD = "student"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "meteo"

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_observations(retries: int = 12, delay: float = 2.5) -> pd.DataFrame:
    url = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url)
    for attempt in range(1, retries + 1):
        try:
            df = pd.read_sql("SELECT * FROM observations", engine)
            print(f"Підключено до MySQL з {attempt}-ї спроби. Рядків: {len(df)}")
            return df
        except OperationalError:
            if attempt == retries:
                raise
            print(f"  MySQL ще не готова (спроба {attempt}/{retries})...")
            time.sleep(delay)
    raise RuntimeError("Unreachable")


def block_1_numpy(df_raw: pd.DataFrame) -> None:
    section("БЛОК 1. NumPy")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        T = np.array(df_raw['temperature_c'], dtype=float)
        RH = np.array(df_raw['humidity_pct'], dtype=float)
        wind = np.array(df_raw['wind_speed_ms'], dtype=float)

        apparent = T - (100.0 - RH) / 5.0
        print(f"1) T_app: len={len(apparent)}, min={np.nanmin(apparent):.2f}, max={np.nanmax(apparent):.2f}")

        t_outliers_mask = (T > 60) | (T < -60)
        wind_outliers_mask = wind > 100

        temperature_clean = np.where(t_outliers_mask, np.nan, T)
        wind_clean = np.where(wind_outliers_mask, np.nan, wind)

        print(f"2) Викидів температури замінено: {np.sum(t_outliers_mask)}")
        print(f"   Викидів вітру замінено:       {np.sum(wind_outliers_mask)}")

        mean_t = np.nanmean(temperature_clean)
        median_t = np.nanmedian(temperature_clean)
        std_t = np.nanstd(temperature_clean)
        print(f"3) mean={mean_t:.3f}  median={median_t:.3f}  std={std_t:.3f}")

        n_frost = np.nansum(temperature_clean < 0)
        n_hot = np.nansum(temperature_clean > 30)
        print(f"4) морозних: {n_frost}    жарких: {n_hot}")

        idx_max = np.nanargmax(temperature_clean)
        idx_min = np.nanargmin(temperature_clean)

        obs_id_max = df_raw['obs_id'].iloc[idx_max]
        dt_max = df_raw['datetime'].iloc[idx_max]
        t_max_val = temperature_clean[idx_max]

        obs_id_min = df_raw['obs_id'].iloc[idx_min]
        dt_min = df_raw['datetime'].iloc[idx_min]
        t_min_val = temperature_clean[idx_min]

        print(f"5) Максимальна T: {t_max_val:.1f}°C (obs_id={obs_id_max}, час={dt_max})")
        print(f"   Мінімальна T:  {t_min_val:.1f}°C (obs_id={obs_id_min}, час={dt_min})")


def block_2_cleaning(df_raw: pd.DataFrame) -> pd.DataFrame:
    section("БЛОК 2. Pandas — очищення")

    rows_before = len(df_raw)
    df = df_raw.copy()

    print("1) df.info():")
    df.info()

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    df.drop_duplicates(inplace=True)
    n_dups = rows_before - len(df)
    print(f"2) drop_duplicates: видалено {n_dups}")

    df['month'] = df.index.month
    humidity_nan_before = df['humidity_pct'].isna().sum()

    df['humidity_pct'] = df.groupby(['city', 'month'])['humidity_pct'].transform(
        lambda s: s.fillna(s.median()) if not s.dropna().empty else s
    )
    n_filled = humidity_nan_before - df['humidity_pct'].isna().sum()
    print(f"3) Заповнено NaN humidity_pct: {n_filled}")

    len_before_outliers = len(df)
    mask_temp = (df['temperature_c'] >= -60) & (df['temperature_c'] <= 60)
    mask_wind = df['wind_speed_ms'].isna() | ((df['wind_speed_ms'] >= 0) & (df['wind_speed_ms'] <= 60))
    df = df[mask_temp & mask_wind].copy()
    n_outliers = len_before_outliers - len(df)
    print(f"4) Видалено фізичних викидів: {n_outliers}")

    print(f"\n   Звіт: {rows_before} → {len(df)} рядків")
    return df


def block_3_analytics(df: pd.DataFrame) -> dict:
    section("БЛОК 3. Pandas — аналітика")

    by_city_temp = df.groupby('city')['temperature_c'].mean().sort_values()
    print("1) Середня T по містах:")
    print(by_city_temp.round(2).to_string())

    by_city_precip = df.groupby('city')['precipitation_mm'].sum().sort_values(ascending=False)
    print("\n2) Сумарні опади по містах:")
    print(by_city_precip.round(1).to_string())

    try:
        monthly_mean = df['temperature_c'].resample('ME').mean()
    except ValueError:
        monthly_mean = df['temperature_c'].resample('M').mean()

    print(f"\n3) Місячна середня T:")
    print(monthly_mean.round(2).to_string())

    pivot = df.pivot_table(values='temperature_c', index='city', columns='month', aggfunc='mean')
    print("\n4) Pivot місто × місяць:")
    print(pivot.round(1).to_string())

    daily_precip = df.groupby(['city', df.index.date])['precipitation_mm'].sum()
    rainy_days = daily_precip[daily_precip > 5].groupby('city').count()
    print("\n5) Дні з опадами > 5 мм:")
    print(rainy_days.to_string())

    df['year'] = df.index.year
    norm_by_month = df.groupby('month')['temperature_c'].mean()
    actual_by_month = df.groupby(['year', 'month'])['temperature_c'].mean()

    deviations = actual_by_month.sub(norm_by_month, level='month')
    anomaly_idx = deviations.abs().idxmax()

    anomaly_month = f"{anomaly_idx[0]}-{anomaly_idx[1]:02d}"
    anomaly_dev = deviations.loc[anomaly_idx]

    print(f"\n6) Аномальний місяць: {anomaly_month}  відхилення = {anomaly_dev:+.2f}°C")

    return {
        "by_city_temp": by_city_temp,
        "by_city_precip": by_city_precip,
        "monthly_mean": monthly_mean,
        "pivot": pivot,
    }


def block_4_plots(df: pd.DataFrame, analytics: dict) -> None:
    section("БЛОК 4. Matplotlib")

    # 1. Monthly Lines
    fig, ax = plt.subplots(figsize=(11, 5))
    cities = df['city'].unique()[:3]
    for city in cities:
        try:
            city_data = df[df['city'] == city]['temperature_c'].resample('ME').mean()
        except ValueError:
            city_data = df[df['city'] == city]['temperature_c'].resample('M').mean()
        ax.plot(city_data.index, city_data.values, marker='o', label=city)
    ax.set_title("Динаміка температури")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Температура (°C)")
    ax.legend()
    fig.savefig(PLOTS_DIR / "01_monthly_temperature_lines.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. Precipitation Bar
    fig, ax = plt.subplots(figsize=(8, 5))
    precip = analytics["by_city_precip"]
    ax.bar(precip.index, precip.values, color='skyblue')
    ax.set_title("Сумарні опади по містах")
    ax.set_xlabel("Місто")
    ax.set_ylabel("Опади (мм)")
    fig.savefig(PLOTS_DIR / "02_precipitation_by_city.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3. Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    temp_data = df['temperature_c'].dropna()
    ax.hist(temp_data, bins=30, color='lightgreen', edgecolor='black')
    mean_val = temp_data.mean()
    median_val = temp_data.median()
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='blue', linestyle='solid', linewidth=2, label=f'Median: {median_val:.1f}')
    ax.set_title("Розподіл температур")
    ax.set_xlabel("Температура (°C)")
    ax.set_ylabel("Частота")
    ax.legend()
    fig.savefig(PLOTS_DIR / "03_temperature_histogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 4. Heatmap
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = analytics["pivot"]
    cax = ax.imshow(pivot.values, cmap='coolwarm', aspect='auto')
    fig.colorbar(cax, ax=ax, label='Температура (°C)')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Середня температура: Місто / Місяць")
    ax.set_xlabel("Місяць")
    ax.set_ylabel("Місто")
    fig.savefig(PLOTS_DIR / "04_city_month_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"4 графіки збережені в {PLOTS_DIR}/")


def main() -> None:
    df_raw = load_observations()
    print(f"Завантажено: shape={df_raw.shape}")

    block_1_numpy(df_raw)
    df_clean = block_2_cleaning(df_raw)
    analytics = block_3_analytics(df_clean)
    block_4_plots(df_clean, analytics)


if __name__ == "__main__":
    main()
"""
ВИСНОВКИ.

Аналіз даних показав, що найтеплішим містом за період спостережень є Дніпро середня температура 12.59°C, 
тоді як найхолоднішим виявилася Одеса 8.01°C, що в межах даного датасету може вказувати на специфічні локальні 
умови розташування метеостанцій. Найбільша кількість опадів фіксується у Києві понад 730 мм, а найпосушливішим 
містом є Одеса. Сезонність температури має яскраво виражений характер із піком у літні місяці та суттєвим спадом 
нижче нуля у грудні-січні. Під час дослідження було виявлено аномальний місяць — травень 2023 року,
який характеризувався значним відхиленням температури на -4.34°C від норми, що свідчить про сильну нетипову хвилю 
холоду наприкінці весни. На основі отриманих результатів рекомендується враховувати ці кліматичні особливості при 
плануванні логістики: наприклад, у Дніпрі варто закладати більші витрати на промислове кондиціонування влітку, 
тоді як у Києві необхідна посилена увага до гідроізоляції складських приміщень через високу кількість опадів.
"""