"""
recommender.py - Movie Recommendation Engine
=============================================
This module handles the core recommendation logic using
content-based filtering with cosine similarity.

How it works:
1. Load movie data from CSV + Kaggle Indian Movies dataset
2. Combine text features (genre, keywords, director, language) into one string
3. Convert text to numerical vectors using TF-IDF
4. Calculate cosine similarity between all movies
5. For given input movies, find the most similar ones
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


def load_kaggle_dataset() -> pd.DataFrame:
    """
    Load the Indian Movies IMDb dataset from Kaggle using kagglehub.
    Filters to movies with valid ratings and votes for quality recommendations.

    Returns:
        DataFrame with standardized columns matching our format.
    """
    try:
        import kagglehub

        # Download the dataset (cached after first download)
        path = kagglehub.dataset_download("nareshbhat/indian-moviesimdb")

        # Find the CSV file in the downloaded directory
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csv_files:
            return pd.DataFrame()

        kaggle_df = pd.read_csv(os.path.join(path, csv_files[0]))

        # ── Clean the Kaggle data ──

        # Remove rows with missing or invalid ratings
        kaggle_df = kaggle_df[kaggle_df["Rating(10)"] != "-"].copy()
        kaggle_df["rating"] = pd.to_numeric(kaggle_df["Rating(10)"], errors="coerce")
        kaggle_df = kaggle_df.dropna(subset=["rating"])

        # Remove rows with missing genres
        kaggle_df = kaggle_df[kaggle_df["Genre"] != "-"].copy()
        kaggle_df = kaggle_df.dropna(subset=["Genre"])

        # Clean votes: remove commas and convert to numeric
        kaggle_df["votes_clean"] = kaggle_df["Votes"].str.replace(",", "", regex=False)
        kaggle_df["votes_clean"] = pd.to_numeric(kaggle_df["votes_clean"], errors="coerce").fillna(0)

        # Filter to movies with decent ratings and enough votes for quality
        # This reduces ~50K movies to a manageable, high-quality subset
        kaggle_df = kaggle_df[
            (kaggle_df["rating"] >= 5.0) & (kaggle_df["votes_clean"] >= 500)
        ].copy()

        # Clean year column
        kaggle_df["year"] = pd.to_numeric(kaggle_df["Year"], errors="coerce").fillna(0).astype(int)

        # Standardize column names to match our format
        result = pd.DataFrame()
        result["movie_name"] = kaggle_df["Movie Name"].str.strip()
        result["genre"] = kaggle_df["Genre"].str.strip()
        result["director"] = ""  # Kaggle dataset doesn't have director info
        result["year"] = kaggle_df["year"].values
        result["rating"] = kaggle_df["rating"].values
        # Use genre words + language as keywords for similarity
        language = kaggle_df["Language"].fillna("").str.strip()
        result["keywords"] = language + " " + result["genre"].str.replace(",", " ", regex=False)
        result["source"] = "kaggle"

        # Remove duplicates by movie name
        result = result.drop_duplicates(subset=["movie_name"], keep="first")

        return result.reset_index(drop=True)

    except Exception as e:
        print(f"Warning: Could not load Kaggle dataset: {e}")
        return pd.DataFrame()


def load_movies(csv_path: str = None) -> pd.DataFrame:
    """
    Load movie data from both the local CSV and the Kaggle Indian Movies dataset.
    Merges them into a single DataFrame with standardized columns.

    Args:
        csv_path: Path to the local CSV file. Defaults to 'movies.csv' in the same directory.

    Returns:
        DataFrame containing combined movie data.
    """
    if csv_path is None:
        # Get the directory where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "movies.csv")

    # ── Load the local curated CSV ──
    local_df = pd.read_csv(csv_path)
    local_df = local_df.fillna("")
    local_df["source"] = "local"

    # ── Load the Kaggle Indian Movies dataset ──
    kaggle_df = load_kaggle_dataset()

    # ── Merge both datasets ──
    if not kaggle_df.empty:
        # Avoid duplicates: remove Kaggle movies that already exist in local CSV
        local_names = set(local_df["movie_name"].str.lower().str.strip())
        kaggle_df = kaggle_df[
            ~kaggle_df["movie_name"].str.lower().str.strip().isin(local_names)
        ]
        df = pd.concat([local_df, kaggle_df], ignore_index=True)
    else:
        df = local_df

    # Clean up: fill any missing values with empty strings
    df = df.fillna("")

    # Normalize movie names to lowercase for easier matching
    df["movie_name_lower"] = df["movie_name"].str.lower().str.strip()

    return df


def combine_features(row: pd.Series) -> str:
    """
    Combine multiple features of a movie into a single text string.
    This combined string is used to calculate similarity between movies.

    We combine: genre, keywords, director, language, and a text version of rating.

    Args:
        row: A single row from the movie DataFrame.

    Returns:
        A combined feature string.
    """
    # Include director info if available (repeat for moderate importance)
    director = str(row.get("director", "")).strip()
    director_text = (director + " ") * 2 if director else ""

    # Convert rating to a descriptive category for similarity matching
    try:
        rating = float(row["rating"]) if row["rating"] else 0
    except (ValueError, TypeError):
        rating = 0

    if rating >= 8.5:
        rating_text = "excellent highly_rated masterpiece "
    elif rating >= 7.5:
        rating_text = "good well_rated popular "
    elif rating >= 5.0:
        rating_text = "average decent watchable "
    else:
        rating_text = "low_rated "

    # Get genre and keywords
    genre = str(row.get("genre", ""))
    keywords = str(row.get("keywords", ""))

    # Combine all features into one string
    combined = f"{genre} {keywords} {director_text} {rating_text}"

    return combined.lower()


def build_similarity_matrix(df: pd.DataFrame):
    """
    Build a cosine similarity matrix from the movie features.

    Steps:
    1. Combine all features into a single text per movie
    2. Use TF-IDF to convert text to numerical vectors
    3. Calculate cosine similarity between all movie pairs

    Args:
        df: Movie DataFrame.

    Returns:
        A cosine similarity matrix (2D numpy array).
    """
    # Step 1: Create combined feature column
    df["combined_features"] = df.apply(combine_features, axis=1)

    # Step 2: Convert text features to TF-IDF vectors
    # TF-IDF = Term Frequency - Inverse Document Frequency
    # It measures how important a word is in a document relative to the whole collection
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])

    # Step 3: Calculate cosine similarity between all movies
    # Cosine similarity measures the angle between two vectors
    # Value ranges from 0 (completely different) to 1 (identical)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix


def find_movie_index(df: pd.DataFrame, movie_name: str) -> int:
    """
    Find the index of a movie in the DataFrame by its name.
    Uses case-insensitive partial matching.

    Args:
        df: Movie DataFrame.
        movie_name: Name of the movie to search for.

    Returns:
        Index of the movie, or -1 if not found.
    """
    movie_name = movie_name.lower().strip()

    # First try exact match
    exact_match = df[df["movie_name_lower"] == movie_name]
    if not exact_match.empty:
        return exact_match.index[0]

    # Then try partial match (if user types part of the name)
    partial_match = df[df["movie_name_lower"].str.contains(movie_name, na=False, regex=False)]
    if not partial_match.empty:
        return partial_match.index[0]

    return -1


def get_recommendations(
    df: pd.DataFrame,
    similarity_matrix,
    input_movies: list,
    top_n: int = 5,
) -> dict:
    """
    Get movie recommendations based on input movies.

    Algorithm:
    1. For each input movie, get its similarity scores with all other movies
    2. Average the similarity scores across all input movies
    3. Sort by highest similarity and return top N recommendations

    Args:
        df: Movie DataFrame.
        similarity_matrix: Pre-computed cosine similarity matrix.
        input_movies: List of movie names provided by the user.
        top_n: Number of recommendations to return.

    Returns:
        A dictionary with:
        - "recommendations": list of recommended movie dicts
        - "not_found": list of movie names that weren't found in the dataset
        - "found": list of movie names that were successfully matched
        - "total_movies": total number of movies in dataset
    """
    not_found = []
    found_indices = []
    found_names = []

    # Find indices for all input movies
    for movie_name in input_movies:
        if not movie_name.strip():
            continue
        idx = find_movie_index(df, movie_name)
        if idx == -1:
            not_found.append(movie_name)
        else:
            found_indices.append(idx)
            found_names.append(df.loc[idx, "movie_name"])

    # If no valid movies found, return empty results
    if not found_indices:
        return {
            "recommendations": [],
            "not_found": not_found,
            "found": found_names,
            "total_movies": len(df),
        }

    # Calculate average similarity scores across all input movies
    # This gives us a combined score that reflects similarity to ALL input movies
    avg_scores = sum(similarity_matrix[idx] for idx in found_indices) / len(found_indices)

    # Create a list of (index, score) pairs and sort by score (descending)
    scored_movies = list(enumerate(avg_scores))
    scored_movies.sort(key=lambda x: x[1], reverse=True)

    # Filter out the input movies themselves and get top N
    recommendations = []
    for idx, score in scored_movies:
        if idx not in found_indices and score > 0:
            movie = df.loc[idx]

            # Safely convert year and rating
            try:
                year = int(float(movie["year"])) if movie["year"] else 0
            except (ValueError, TypeError):
                year = 0
            try:
                rating = float(movie["rating"]) if movie["rating"] else 0.0
            except (ValueError, TypeError):
                rating = 0.0

            recommendations.append(
                {
                    "name": movie["movie_name"],
                    "genre": movie["genre"],
                    "director": movie.get("director", ""),
                    "year": year,
                    "rating": rating,
                    "score": round(float(score), 4),
                }
            )
        if len(recommendations) >= top_n:
            break

    return {
        "recommendations": recommendations,
        "not_found": not_found,
        "found": found_names,
        "total_movies": len(df),
    }
