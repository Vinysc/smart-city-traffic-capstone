"""
CLI for Mini Traffic Analytics Application.

Usage examples:
    python app.py summary
    python app.py by-datetime --datetime "2012-10-02 09:00:00"
    python app.py high-traffic --threshold 5000
    python app.py compare-days
    python app.py recommend --day weekend
"""

# Standard library imports for handling CLI arguments, logging, system exits, and file paths
import argparse
import logging
import sys
from pathlib import Path

# Data analysis library for tabular data manipulation
import pandas as pd

# Define the relative path to the preprocessed traffic dataset CSV
FEAT_PATH = Path("data/features_traffic.csv")

# Create a logger specific to this module (__name__ resolves to "__main__" when run directly)
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """
    Configures the dual-logging system (Console Stream + File output).
    Ensures logs are written to both a file and printed to the terminal.
    """
    root = logging.getLogger()
    
    # Avoid adding duplicate handlers if setup_logging() is called multiple times
    if root.handlers:
        return
        
    # Set the base logging level for the root logger to capture all DEBUG messages
    root.setLevel(logging.DEBUG)

    # Format string defining: Timestamp | Log Level | Logger Name | Message
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler 1: FileHandler - Appends detailed DEBUG-level logs to 'pipeline.log'
    fh = logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Handler 2: StreamHandler - Prints clean INFO-level logs directly to the terminal
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # Attach both handlers to the root logger
    root.addHandler(fh)
    root.addHandler(ch)


def load_data() -> pd.DataFrame:
    """
    Safely loads the CSV dataset into a pandas DataFrame and converts date column.
    Exits gracefully with a log entry if the file does not exist or fails to parse.
    """
    try:
        # Read dataset into DataFrame
        df = pd.read_csv(FEAT_PATH)
        # Parse the string 'date_time' column into actual pandas datetime objects
        df["date_time"] = pd.to_datetime(df["date_time"])
        return df
    except FileNotFoundError:
        # Log missing file error with stack trace and terminate the program safely
        logger.error("Traffic features file not found: %s", FEAT_PATH, exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Log unexpected data parsing errors and terminate
        logger.error("Failed to parse data: %s", e, exc_info=True)
        sys.exit(1)


def cmd_summary(df: pd.DataFrame) -> None:
    """
    Command 1: Displays overall dataset statistics including record counts,
    date ranges, average traffic volume, temperature, and congestion category breakdown.
    """
    print("Total Records:", len(df))
    print("Date range:", df["date_time"].min(), "to", df["date_time"].max())
    print("Avg traffic volume:", round(df["traffic_volume"].mean(), 1))
    print("Avg temperature:", round(df["temp"].mean(), 1))
    
    # Check if target variable 'congestion_category' exists before displaying count distribution
    if "congestion_category" in df.columns:
        print("\nCongestion Categories:")
        print(df["congestion_category"].value_counts().to_string())


def cmd_by_datetime(df: pd.DataFrame, datetime_str: str) -> None:
    """
    Command 2: Queries specific traffic and weather info for a given timestamp.
    Validates user input and outputs an informative error instead of throwing a traceback.
    """
    try:
        # Validate and parse input string to pandas Timestamp
        dt = pd.to_datetime(datetime_str)
    except Exception:
        # Catch malformed date inputs, log an error, and alert the user cleanly
        logger.error("Invalid datetime format received: %s", datetime_str)
        print("Error: Invalid date format. Please use 'YYYY-MM-DD HH:MM:SS'.")
        return

    # Filter DataFrame for exact timestamp match
    sub = df[df["date_time"] == dt]
    if sub.empty:
        print(f"No traffic records found for: {datetime_str}")
        return

    # Extract matching row and display details
    row = sub.iloc[0]
    print(f"\n--- Traffic & Weather Details for {dt} ---")
    print(f"Traffic Volume : {row['traffic_volume']}")
    print(f"Weather        : {row.get('weather_main', 'N/A')} ({row.get('weather_description', 'N/A')})")
    print(f"Temperature    : {row.get('temp', 'N/A')}")
    print(f"Rain (1h)      : {row.get('rain_1h', 0)}")
    print(f"Snow (1h)      : {row.get('snow_1h', 0)}")


def cmd_high_traffic(df: pd.DataFrame, threshold: int) -> None:
    """
    Command 3: Identifies periods where traffic volume exceeds the user-defined threshold.
    Prints total counts and displays details for the top 5 peak hours.
    """
    if threshold < 0:
        logger.error("Invalid traffic volume threshold: %s", threshold)
        print("Error: Threshold volume must be a non-negative integer.")
        return

    # Filter rows meeting or exceeding the volume cutoff
    high_df = df[df["traffic_volume"] >= threshold]
    print(f"\nFound {len(high_df)} records exceeding {threshold} volume threshold.")
    
    if not high_df.empty:
        print("\nTop 5 Busiest Hours Recorded:")
        # Sort by volume descending to get peak congestion hours
        top = high_df.sort_values(by="traffic_volume", ascending=False).head(5)
        for _, row in top.iterrows():
            print(f" - {row['date_time']}: {row['traffic_volume']} volume | Weather: {row.get('weather_main', 'N/A')}")


def cmd_compare_days(df: pd.DataFrame) -> None:
    """
    Command 4: Calculates and compares average traffic volume between weekdays and weekends.
    """
    # Filter using binary flag (0 = Weekday, 1 = Weekend)
    weekday_avg = df[df["is_weekend"] == 0]["traffic_volume"].mean()
    weekend_avg = df[df["is_weekend"] == 1]["traffic_volume"].mean()

    print("\n--- Traffic Comparison: Weekday vs Weekend ---")
    print(f"Avg Weekday Traffic Volume : {weekday_avg:.1f}")
    print(f"Avg Weekend Traffic Volume : {weekend_avg:.1f}")
    
    diff = weekday_avg - weekend_avg
    print(f"Difference                 : Weekday is {abs(diff):.1f} {'higher' if diff > 0 else 'lower'}")


def cmd_recommend(df: pd.DataFrame, day: str) -> None:
    """
    Command 5: Calculates average hourly traffic for weekdays or weekends
    and identifies the top 3 best times (lowest traffic volume) to travel.
    """
    day_clean = day.lower().strip()
    # Validate user argument
    if day_clean not in ("weekday", "weekend"):
        logger.error("Invalid day type received: %s", day)
        print("Error: day must be 'weekday' or 'weekend'")
        return

    # Map day type to flag: weekend -> 1, weekday -> 0
    is_wknd = 1 if day_clean == "weekend" else 0
    sub = df[df["is_weekend"] == is_wknd]

    # Group by 24-hour integer, average traffic volume, and sort ascending (lowest volume first)
    by_hour = sub.groupby("hour")["traffic_volume"].mean().sort_values(ascending=True)
    best_hours = by_hour.head(3)

    print(f"\n--- Recommended Travel Hours ({day_clean.capitalize()}) ---")
    print("Lowest average traffic volume hours:")
    for h, v in best_hours.items():
        print(f"  {int(h):02d}:00 - avg volume: {v:.0f}")

    best_h = int(best_hours.index[0])
    print(f"Suggestion: Best time to travel on a {day_clean} is around {best_h:02d}:00.")


def main() -> None:
    """
    Main entry point: Sets up logging, parses CLI arguments, and routes commands.
    """
    # Initialize loggers
    setup_logging()

    # Configure command-line parser
    parser = argparse.ArgumentParser(description="Mini Traffic Analytics CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Command 1 setup: summary
    subparsers.add_parser("summary", help="Print dataset overview and average traffic stats")

    # Command 2 setup: by-datetime --datetime <STRING>
    p_dt = subparsers.add_parser("by-datetime", help="Query traffic and weather for a given date/time")
    p_dt.add_argument("--datetime", type=str, required=True, help="Format: 'YYYY-MM-DD HH:MM:SS'")

    # Command 3 setup: high-traffic --threshold <INT>
    p_ht = subparsers.add_parser("high-traffic", help="List periods exceeding a traffic threshold")
    p_ht.add_argument("--threshold", type=int, default=4000, help="Volume threshold limit")

    # Command 4 setup: compare-days
    subparsers.add_parser("compare-days", help="Compare average traffic between weekdays and weekends")

    # Command 5 setup: recommend --day <weekday|weekend>
    p_rec = subparsers.add_parser("recommend", help="Recommend optimal travel times")
    p_rec.add_argument("--day", type=str, required=True, help="'weekday' or 'weekend'")

    # Parse arguments provided at the command line
    args = parser.parse_args()

    # If user ran `python app.py` with no subcommand, display help menu and exit
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Logging requirement: Record command name and all supplied arguments at INFO level
    logger.info("CLI command=%s args=%s", args.command, vars(args))

    # Load data into DataFrame
    df = load_data()

    # Execute corresponding function based on parsed subcommand
    if args.command == "summary":
        cmd_summary(df)
    elif args.command == "by-datetime":
        cmd_by_datetime(df, args.datetime)
    elif args.command == "high-traffic":
        cmd_high_traffic(df, args.threshold)
    elif args.command == "compare-days":
        cmd_compare_days(df)
    elif args.command == "recommend":
        cmd_recommend(df, args.day)


# Ensures code inside main() runs only when file is executed directly (not when imported)
if __name__ == "__main__":
    main()