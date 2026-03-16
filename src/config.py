from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output folders
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
RECOMMENDATIONS_DIR = OUTPUT_DIR / "recommendations"

# MovieLens expected files
MOVIES_FILE = RAW_DATA_DIR / "movies.csv"
RATINGS_FILE = RAW_DATA_DIR / "ratings.csv"
TAGS_FILE = RAW_DATA_DIR / "tags.csv"
LINKS_FILE = RAW_DATA_DIR / "links.csv"