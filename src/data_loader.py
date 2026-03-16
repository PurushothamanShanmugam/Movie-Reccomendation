import pandas as pd
from pathlib import Path
from src.config import MOVIES_FILE, RATINGS_FILE, TAGS_FILE, LINKS_FILE


def load_movielens_data():
    """
    Load MovieLens CSV files from data/raw folder.
    Uses absolute project-aware paths from config.py
    so it works regardless of the current working directory.
    """

    required_files = [MOVIES_FILE, RATINGS_FILE, TAGS_FILE, LINKS_FILE]

    for file_path in required_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {file_path}\n"
                f"Please make sure the file exists inside data/raw/"
            )

    movies = pd.read_csv(MOVIES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)
    tags = pd.read_csv(TAGS_FILE)
    links = pd.read_csv(LINKS_FILE)

    return movies, ratings, tags, links