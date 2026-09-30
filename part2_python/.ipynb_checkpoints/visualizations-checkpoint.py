"""Plots for FreshBasket traffic patterns. Run: python visualizations.py"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Define paths for input data and output directory for figures
FEAT_PATH = Path("data/features_traffic.csv")
FIG_DIR = Path("figures")

# Create a logger instance for this module
logger = logging.getLogger(__name__)


def setup_logging():
    """Configures logging handlers to log messages to both a file (pipeline.log) and standard output console."""
    root = logging.getLogger()

    # Prevent adding duplicate handlers if logging is already set up
    if root.handlers:
        return

    # Set root logger level to capture DEBUG and higher level logs
    root.setLevel(logging.DEBUG)

    # Format string detailing timestamp, log level, logger name, and log message
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler: Appends detailed DEBUG logs to pipeline.log
    fh = logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Stream Handler: Prints clean INFO logs to the console window
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # Attach both handlers to the root logger
    root.addHandler(fh)
    root.addHandler(ch)


def main():
    # Initialize logging configuration
    setup_logging()

    # Ensure output directory for figures exists
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Load the feature-engineered traffic dataset
        df = pd.read_csv(FEAT_PATH)
        logger.info("Loaded features: %s rows", len(df))

        # Determine actual column names dynamically or fall back
        traffic_col = "traffic_volume" if "traffic_volume" in df.columns else "traffic"
        temp_col = "temp" if "temp" in df.columns else "temperature"

        # =========================================================================
        # Visualisation 1: Traffic Demand by Hour (Line Plot)
        # Purpose: Identifies peak congestion hours to optimize delivery timing.
        # =========================================================================

        g1 = df.groupby("hour")[traffic_col].mean()
        p1 = FIG_DIR / "traffic_demand_by_hour.png"

        plt.figure(figsize=(8, 4))
        plt.plot(g1.index, g1.values, marker="o", color="teal", linewidth=2)
        plt.xlabel("Hour of Day")
        plt.ylabel("Avg Traffic Volume")
        plt.title("Average Traffic Demand by Hour")
        plt.xticks(range(0, 24))  # Ensure all 24 hours are labeled on x-axis
        plt.grid(True, linestyle="--", alpha=0.5)  # Add faint grid lines
        plt.tight_layout()  # Adjust layout to prevent label clipping

        # Save plot to file and log confirmation
        plt.savefig(p1)
        plt.close()  # Free memory resources
        logger.info("Saved figure: %s", p1)

        # =========================================================================
        # Visualisation 2: Weekday vs. Weekend Traffic (Bar Chart)
        # Purpose: Compares travel volume across week categories to inform staffing.
        # =========================================================================

        g2 = df.groupby("is_weekend")[traffic_col].mean()
        p2 = FIG_DIR / "weekday_vs_weekend_traffic.png"

        plt.figure(figsize=(5, 4))
        plt.bar(
            ["Weekday", "Weekend"],
            [g2.get(0, 0), g2.get(1, 0)],
            color=["coral", "seagreen"],
        )
        plt.ylabel("Avg Traffic Volume")
        plt.title("Weekday vs Weekend Traffic")
        plt.tight_layout()

        # Save plot to file and log confirmation
        plt.savefig(p2)
        plt.close()
        logger.info("Saved figure: %s", p2)

        # =========================================================================
        # Visualisation 3: Temperature vs. Traffic (Scatter Plot)
        # Purpose: Analyzes potential correlation between weather/temp and traffic.
        # =========================================================================

        p3 = FIG_DIR / "temperature_vs_traffic.png"

        plt.figure(figsize=(7, 4))
        plt.scatter(
            df[temp_col], df[traffic_col], alpha=0.4, color="slateblue"
        )
        plt.xlabel("Temperature")
        plt.ylabel("Traffic Volume")
        plt.title("Temperature vs. Traffic")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Save plot to file and log confirmation
        plt.savefig(p3)
        plt.close()
        logger.info("Saved figure: %s", p3)

        # Final success log entry
        logger.info("All figures created")

    except FileNotFoundError:
        logger.error(
            "Features file missing. Run feature_engineering.py first.",
            exc_info=True,
        )
        sys.exit(1)
    except Exception:
        logger.error("Visualisation step failed", exc_info=True)
        sys.exit(1)


# Standard Python entry point guard
if __name__ == "__main__":
    main()