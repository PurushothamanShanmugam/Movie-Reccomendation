import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def extract_year_from_title(title: str) -> int:
    """
    Extract release year from movie title like 'Toy Story (1995)'.
    """
    match = re.search(r"\((\d{4})\)", str(title))
    if match:
        return int(match.group(1))
    return 2000


def prepare_movie_features(movies: pd.DataFrame, ratings: pd.DataFrame):
    """
    Create movie-level metadata features:
    - one-hot genres
    - release year
    - average movie rating
    """
    movies = movies.copy()

    movies["year"] = movies["title"].apply(extract_year_from_title)

    avg_movie_rating = ratings.groupby("movieId")["rating"].mean().reset_index()
    avg_movie_rating.columns = ["movieId", "avg_movie_rating"]

    movies = movies.merge(avg_movie_rating, on="movieId", how="left")
    movies["avg_movie_rating"] = movies["avg_movie_rating"].fillna(ratings["rating"].mean())

    genre_dummies = movies["genres"].str.get_dummies(sep="|")
    movie_features = pd.concat(
        [
            movies[["movieId", "title", "year", "avg_movie_rating"]],
            genre_dummies,
        ],
        axis=1,
    )

    return movie_features, genre_dummies.columns.tolist()


def prepare_user_features(ratings: pd.DataFrame, movies: pd.DataFrame, genre_columns: list[str]):
    """
    Create user features as average rating per genre.
    """
    movie_genres = movies[["movieId", "genres"]].copy()
    genre_dummies = movie_genres["genres"].str.get_dummies(sep="|")
    movie_genres = pd.concat([movie_genres[["movieId"]], genre_dummies], axis=1)

    merged = ratings.merge(movie_genres, on="movieId", how="left")

    user_feature_rows = []
    for user_id, group in merged.groupby("userId"):
        row = {"userId": user_id}

        for genre in genre_columns:
            genre_rated = group[group.get(genre, 0) == 1]
            if len(genre_rated) > 0:
                row[f"user_genre_{genre}"] = genre_rated["rating"].mean()
            else:
                row[f"user_genre_{genre}"] = ratings["rating"].mean()

        user_feature_rows.append(row)

    user_features = pd.DataFrame(user_feature_rows)
    return user_features


def build_training_data(ratings: pd.DataFrame, movie_features: pd.DataFrame, user_features: pd.DataFrame):
    """
    Build merged dataset for neural model training.
    """
    data = ratings.merge(movie_features, on="movieId", how="left")
    data = data.merge(user_features, on="userId", how="left")
    return data


def get_feature_columns(movie_features: pd.DataFrame, user_features: pd.DataFrame):
    """
    Return column lists for user and movie features.
    """
    movie_feature_cols = [col for col in movie_features.columns if col not in ["movieId", "title"]]
    user_feature_cols = [col for col in user_features.columns if col != "userId"]
    return user_feature_cols, movie_feature_cols


def build_neural_recommender(user_input_dim: int, movie_input_dim: int, learning_rate: float = 0.001):
    """
    Build two-branch neural recommender model.
    """
    user_input = Input(shape=(user_input_dim,), name="user_input")
    x_user = Dense(64, activation="relu")(user_input)
    user_embedding = Dense(32, activation="relu", name="user_embedding")(x_user)

    movie_input = Input(shape=(movie_input_dim,), name="movie_input")
    x_movie = Dense(64, activation="relu")(movie_input)
    movie_embedding = Dense(32, activation="relu", name="movie_embedding")(x_movie)

    combined = Concatenate()([user_embedding, movie_embedding])
    x = Dense(64, activation="relu")(combined)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="linear", name="rating_prediction")(x)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    return model


def train_neural_recommender(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 0.001
):
    """
    Prepare features, train neural recommender, and return model artifacts.
    """
    movie_features, genre_columns = prepare_movie_features(movies, ratings)
    user_features = prepare_user_features(ratings, movies, genre_columns)

    full_data = build_training_data(ratings, movie_features, user_features)
    user_feature_cols, movie_feature_cols = get_feature_columns(movie_features, user_features)

    X_user = full_data[user_feature_cols].values.astype("float32")
    X_movie = full_data[movie_feature_cols].values.astype("float32")
    y = full_data["rating"].values.astype("float32")

    X_user_train, X_user_test, X_movie_train, X_movie_test, y_train, y_test = train_test_split(
        X_user, X_movie, y, test_size=0.2, random_state=42
    )

    model = build_neural_recommender(
        user_input_dim=X_user.shape[1],
        movie_input_dim=X_movie.shape[1],
        learning_rate=learning_rate,
    )

    history = model.fit(
        [X_user_train, X_movie_train],
        y_train,
        validation_data=([X_user_test, X_movie_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    predictions = model.predict([X_user_test, X_movie_test], verbose=0).flatten()
    predictions = np.clip(predictions, 0.5, 5.0)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    artifacts = {
        "model": model,
        "history": history,
        "rmse": float(rmse),
        "movie_features": movie_features,
        "user_features": user_features,
        "genre_columns": genre_columns,
        "user_feature_cols": user_feature_cols,
        "movie_feature_cols": movie_feature_cols,
        "full_data": full_data,
    }

    return artifacts


def recommend_neural_model(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    artifacts: dict,
    top_n: int = 10
):
    """
    Generate top-N recommendations for a user using the trained neural model.
    """
    model = artifacts["model"]
    movie_features = artifacts["movie_features"]
    user_features = artifacts["user_features"]
    user_feature_cols = artifacts["user_feature_cols"]
    movie_feature_cols = artifacts["movie_feature_cols"]

    if user_id not in user_features["userId"].values:
        raise ValueError(f"User {user_id} not found in user features.")

    rated_movies = set(ratings.loc[ratings["userId"] == user_id, "movieId"].tolist())

    user_row = user_features[user_features["userId"] == user_id].iloc[0]
    user_vector = user_row[user_feature_cols].values.astype("float32").reshape(1, -1)

    candidate_movies = movie_features[~movie_features["movieId"].isin(rated_movies)].copy()

    movie_matrix = candidate_movies[movie_feature_cols].values.astype("float32")
    user_matrix = np.repeat(user_vector, repeats=len(candidate_movies), axis=0)

    predictions = model.predict([user_matrix, movie_matrix], verbose=0).flatten()
    predictions = np.clip(predictions, 0.5, 5.0)

    candidate_movies["predicted_rating"] = predictions
    candidate_movies = candidate_movies.sort_values(by="predicted_rating", ascending=False).head(top_n)

    result = candidate_movies.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")

    result = result.rename(columns={"title_x": "title", "title_y": "title_movie"})

    if "title" not in result.columns:
      if "title_movie" in result.columns:
        result["title"] = result["title_movie"]

    return result[["movieId", "title", "genres", "predicted_rating"]].reset_index(drop=True)

def precision_recall_at_k_neural(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    artifacts: dict,
    top_k: int = 10,
    relevant_threshold: float = 4.0
):
    """
    Compute Precision@K and Recall@K for neural recommender.
    """
    relevant_items = set(
        ratings[
            (ratings["userId"] == user_id) &
            (ratings["rating"] >= relevant_threshold)
        ]["movieId"].tolist()
    )

    if len(relevant_items) == 0:
        return 0.0, 0.0

    recs = recommend_neural_model(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        artifacts=artifacts,
        top_n=top_k,
    )

    recommended_items = set(recs["movieId"].tolist())
    true_positives = len(recommended_items.intersection(relevant_items))

    precision = true_positives / top_k if top_k > 0 else 0.0
    recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0.0

    return precision, recall