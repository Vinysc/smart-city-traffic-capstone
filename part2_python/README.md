# Capstone Part 2 – Smart City Traffic Intelligence

A reproducible Python data pipeline for cleaning, engineering features, and analyzing the Metro Interstate Traffic Volume dataset.

## Layout

```
capstone_part2/
  data/Metro_Interstate_Traffic_Volume.csv  # Raw dataset
  data/cleaned_traffic.csv                  # Output from pipeline.py
  data/features_traffic.csv                 # Output from feature_engineering.py
  figures/                                  # Output from visualizations.py
  pipeline.py                               # Data cleaning and validation
  feature_engineering.py                    # Feature creation (time, weather, targets)
  visualizations.py                         # Matplotlib plotting script
  app.py                                    # CLI application for querying data
  pipeline.log                              # Generated log file
  README.md                                 # Project documentation
  requirements.txt                          # Project dependencies
  
```

## Setup

```
Install the required packages before running the pipeline:
pip install -r requirements.txt
```

## Run

```
# Execute the pipeline sequentially to clean data, engineer features, and generate visualizations:

# Clean and validate data
python pipeline.py

# Create features
python feature_engineering.py

# Create plots
python visualizations.py

# Print dataset overview and average traffic stats
python app.py summary

# Query traffic and weather for a specific date and time
python app.py by-datetime --datetime "2012-10-02 09:00:00"

# List periods exceeding a specified traffic volume threshold
python app.py high-traffic --threshold 5000

# Compare average traffic between weekdays and weekends
python app.py compare-days

# Recommend optimal travel times based on day type
python app.py recommend --day weekend
```

## Logging
- `logging.getLogger(__name__)` in each file
- console + `pipeline.log`
- format: time, level, module, message
- DEBUG = thresholds / maps; INFO = load/save/shape; WARNING = drops/imputes; ERROR = failures

## Congestion category
Quartiles of `traffic_volume`: Low / Medium / High / Severe