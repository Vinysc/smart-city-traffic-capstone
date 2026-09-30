"""Add features for traffic data. Run: python feature_engineering.py"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_PATH = Path("data/cleaned_traffic.csv")
FEAT_PATH = Path("data/features_traffic.csv")
logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for file and stream outputs."""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # File handler records everything down to DEBUG
    fh = logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Console handler records normal operation (INFO and above)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Log shape before feature engineering
    logger.info(
        "Feature engineering start: %s rows, %s columns", df.shape[0], df.shape[1]
    )
    df = df.copy()
    # --- Time Features ---
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding for time of day (24-hour cycle)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

   # --- WEATHER FEATURES ---
    # Label encoding for categorical weather variable (weather_main)
    if "weather_main" in df.columns:
        weather_map = {
            w: i
            for i, w in enumerate(sorted(df["weather_main"].dropna().unique()))
        }
        df["weather_main_code"] = df["weather_main"].map(weather_map)
        # Log mapping dictionary at DEBUG level so it doesn't clutter console
        logger.debug("weather_main_map=%s", weather_map)

    # Derived boolean flags for rain and snow
    if "rain_1h" in df.columns:
        df["is_rainy"] = (df["rain_1h"] > 0).astype(int)

    if "snow_1h" in df.columns:
        df["is_snowy"] = (df["snow_1h"] > 0).astype(int)

    # --- NUMERICAL FEATURES ---
    # Min-Max Scaling for continuous weather variables (normalizes values to [0, 1])
    scale_cols = [c for c in ["temp", "rain_1h", "clouds_all"] if c in df.columns]
    for col in scale_cols:
        mn, mx = df[col].min(), df[col].max()
        df[col + "_scaled"] = (
            (df[col] - mn) / (mx - mn) if mx != mn else 0.0
        )
        # Log intermediate min/max values at DEBUG level
        logger.debug("%s min=%.4f max=%.4f", col, mn, mx)

    # --- TRAFFIC TARGET CATEGORY ---
    # Create data-driven congestion target based on traffic volume quartiles
    if "traffic_volume" in df.columns:
        q1, q2, q3 = (
            df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
        )
        # Log quartile thresholds at DEBUG level
        logger.debug(
            "traffic_volume quartiles q1=%.1f q2=%.1f q3=%.1f", q1, q2, q3
        )

        # Logic to map traffic volume into four congestion levels
        def congestion_label(v):
            if v <= q1:
                return "Low"
            if v <= q2:
                return "Moderate"
            if v <= q3:
                return "Heavy"
            return "Severe"

        labels = []
        for v in df["traffic_volume"]:
            labels.append(congestion_label(v))

        df["congestion_category"] = labels

    # 2. Log final dataset shape after feature engineering (INFO level)
    logger.info(
        "Feature engineering end: %s rows, %s columns",
        df.shape[0],
        df.shape[1],
    )
    return df


def main():
    setup_logging()
    try:
        # Load cleaned traffic dataset
        df = pd.read_csv(CLEAN_PATH)
        logger.info("Loaded cleaned data from %s", CLEAN_PATH)

        # Apply feature engineering
        df = add_features(df)

        # Save output dataset
        df.to_csv(FEAT_PATH, index=False)
        logger.info("Saved features to %s", FEAT_PATH)

    except FileNotFoundError:
        logger.error(
            "Cleaned file missing (%s). Ensure clean_traffic.csv exists.",
            CLEAN_PATH,
            exc_info=True,
        )
        sys.exit(1)
    except Exception:
        logger.error("Feature engineering failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
