import numpy as np
import pandas as pd


def build_q_table(ratings: pd.DataFrame):
    """
    Build Q-table initialized with zeros
    Rows = users
    Columns = movies
    """

    users = ratings["userId"].unique()
    movies = ratings["movieId"].unique()

    q_table = pd.DataFrame(
        0,
        index=users,
        columns=movies,
        dtype=float
    )

    return q_table


def train_q_learning(
    ratings: pd.DataFrame,
    episodes: int = 5,
    alpha: float = 0.1,
    gamma: float = 0.9
):
    """
    Train Q-learning recommender using ratings as reward
    """

    q_table = build_q_table(ratings)

    for episode in range(episodes):

        for _, row in ratings.iterrows():

            user = row["userId"]
            movie = row["movieId"]
            reward = row["rating"]

            current_q = q_table.loc[user, movie]

            max_future_q = q_table.loc[user].max()

            new_q = current_q + alpha * (
                reward + gamma * max_future_q - current_q
            )

            q_table.loc[user, movie] = new_q

    return q_table


def recommend_rl(
    user_id: int,
    q_table: pd.DataFrame,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    top_n: int = 10
):
    """
    Recommend movies using learned Q-table
    """

    if user_id not in q_table.index:
        raise ValueError("User not found in Q-table")

    rated_movies = ratings[ratings["userId"] == user_id]["movieId"].tolist()

    user_q_values = q_table.loc[user_id].copy()

    user_q_values = user_q_values.drop(rated_movies, errors="ignore")

    top_movies = user_q_values.sort_values(ascending=False).head(top_n)

    result = pd.DataFrame({
        "movieId": top_movies.index,
        "q_score": top_movies.values
    })

    result = result.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    return result[["movieId", "title", "genres", "q_score"]]


def evaluate_rl_rmse(
    ratings: pd.DataFrame,
    q_table: pd.DataFrame
):
    """
    Evaluate RL predictions using RMSE
    """

    predictions = []
    actuals = []

    for _, row in ratings.iterrows():

        user = row["userId"]
        movie = row["movieId"]

        if user in q_table.index and movie in q_table.columns:

            pred = q_table.loc[user, movie]

            predictions.append(pred)
            actuals.append(row["rating"])

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    return rmse