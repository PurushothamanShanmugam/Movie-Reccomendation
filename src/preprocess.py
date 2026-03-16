import pandas as pd
import os

def preprocess_data(movies, ratings):

   # os.makedirs("processed", exist_ok=True)

    merged = ratings.merge(movies, on="movieId")

    merged.to_csv("data/processed/merged_dataset.csv", index=False)

    print("Processed dataset saved to processed/merged_dataset.csv")

    return merged