"""Clean Metro Interstate Traffic Volume CSV. Run: python pipeline.py"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Expected schema for Metro Interstate Traffic Volume Dataset
EXPECTED_COLS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]

RAW_PATH = Path("data/Metro_Interstate_Traffic_Volume.csv")
CLEAN_PATH = Path("data/cleaned_traffic.csv")

# Setup module-level logger
logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging handlers, levels and formatting."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

# Define standard log message format
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler: Output detailed DEBUG logs to pipeline.log
    fh = logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Stream Handloer: Output high-level INFO logs to the terminal/console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # Avoid duplicate handlers if setup_logging is called multiple times
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)

# Task 1: Load the raw CSV file with exception handling
def load_raw(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("CSV not found: %s", path, exc_info=True)
        sys.exit(1)
    except pd.errors.ParserError:
        logger.error("Could not parse CSV: %s", path, exc_info=True)
        sys.exit(1)
    logger.info("Loaded raw data: %s rows, %s columns", df.shape[0], df.shape[1])
    return df

# Task 2: Validate data schema
def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)
    logger.info("Schema validation passed")

# Task 3 & 4: Data cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

# standardise categorical text fields
    before_h = df["holiday"].astype(str)
    df["holiday"] = before_h.str.strip().str.title()
    n_h = int((df["holiday"] != before_h).sum())
    if n_h:
        logger.warning("Standardised holiday casing for %s rows", n_h)
    else:
        logger.info("holiday already consistent")
    before_wm = df["weather_main"].astype(str)
    df["weather_main"] = before_wm.str.strip().str.title()
    n_wm = int((df["weather_main"] != before_wm).sum())
    if n_wm:
        logger.warning("Standardised weather_main casing for %s rows", n_wm)
    else:
        logger.info("weather_main already consistent")

    before_wd = df["weather_description"].astype(str)
    df["weather_description"] = before_wd.str.strip().str.lower()
    n_wd = int((df["weather_description"] != before_wd).sum())
    if n_wd:
        logger.warning("Standardised weather_description casing for %s rows", n_wd)
    else:
        logger.info("weather_description already consistent")

# Parse and validate date_time
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    bad = int(df["date_time"].isna().sum())
    if bad:
        df = df.dropna(subset=["date_time"]).copy()
        logger.warning("Dropped %s rows with invalid date_time", bad)
    else:
        logger.info("All date_time values parsed OK")

# Remove duplication
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates().copy()
        logger.warning("Removed %s duplicate rows", n_dup)
    else:
        logger.info("No duplicate rows found")

    df["month"] = df["date_time"].dt.month

# sensor error: temp <= 0 Kelvin (-273.15 °C) is physically impossible
    bad_t = df["temp"] <= 0
    n_t = int(bad_t.sum())
    if n_t:
        for m in df.loc[bad_t, "month"].unique():
            med = df.loc[(df["month"] == m) & (~bad_t), "temp"].median()
            df.loc[bad_t & (df["month"] == m), "temp"] = med
        logger.warning("Imputed %s impossible temp values with monthly median", n_t)
    else:
        logger.info("No impossible temp values found")

# rain_1h > 9000 mm is physically implausible sensor failure (e.g. 9831 mm)
    bad_r = df["rain_1h"] > 9000
    n_r = int(bad_r.sum())
    if n_r:
        for m in df.loc[bad_r, "month"].unique():
            med = df.loc[(df["month"] == m) & (~bad_r), "rain_1h"].median()
            df.loc[bad_r & (df["month"] == m), "rain_1h"] = med
        logger.warning("Imputed %s extreme rain_1h values with monthly median", n_r)
    else:
        logger.info("No extreme rain_1h values found")

# Remove month column created for outlier imputation by month
    df = df.drop(columns=["month"])
    logger.info("Cleaning finished: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def main():
    setup_logging()
    logger.info("Pipeline started")
    try:
        df = load_raw(RAW_PATH)
        validate_schema(df)
        df = clean_data(df)
        df.to_csv(CLEAN_PATH, index=False)
        logger.info("Saved cleaned data to %s", CLEAN_PATH)
        logger.info("Pipeline completed successfully")
    except Exception:
        logger.error("Pipeline failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()