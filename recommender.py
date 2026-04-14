"""
recommender.py - Movie Recommendation Engine
=============================================
This module handles the core recommendation logic using
content-based filtering with cosine similarity.

How it works:
1. Load movie data from CSV
2. Combine text features (genre, keywords, director) into one string
3. Convert text to numerical vectors using TF-IDF
4. Calculate cosine similarity between all movies
5. For given input movies, find the most similar ones
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


def load_movies(csv_path: str = None) -> pd.DataFrame:
    """
    Load the movie dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file. Defaults to 'movies.csv' in the same directory.

    Returns:
        DataFrame containing movie data.
    """
    if csv_path is None:
        # Get the directory where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "movies.csv")

    df = pd.read_csv(csv_path)

    # Clean up: fill any missing values with empty strings
    df = df.fillna("")

    # Normalize movie names to lowercase for easier matching
    df["movie_name_lower"] = df["movie_name"].str.lower().str.strip()

    return df


def combine_features(row: pd.Series) -> str:
    """
    Combine multiple features of a movie into a single text string.
    This combined string is used to calculate similarity between movies.

    We combine: genre, keywords, director, and a text version of rating.
    The rating is repeated to give it more weight in similarity calculation.

    Args:
        row: A single row from the movie DataFrame.

    Returns:
        A combined feature string.
    """
    # Repeat director name to give it moderate importance
    director_text = (str(row["director"]) + " ") * 2

    # Repeat rating category to give it some importance
    # Convert rating to a descriptive category
    rating = float(row["rating"]) if row["rating"] else 0
    if rating >= 8.5:
        rating_text = "excellent highly_rated masterpiece "
    elif rating >= 7.5:
        rating_text = "good well_rated popular "
    else:
        rating_text = "average decent watchable "

    # Combine all features into one string
    combined = f"{row['genre']} {row['keywords']} {director_text} {rating_text}"

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
        A tuple of (similarity_matrix, tfidf_matrix).
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
    partial_match = df[df["movie_name_lower"].str.contains(movie_name, na=False)]
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
            recommendations.append(
                {
                    "name": movie["movie_name"],
                    "genre": movie["genre"],
                    "director": movie["director"],
                    "year": int(movie["year"]),
                    "rating": float(movie["rating"]),
                    "score": round(float(score), 4),
                }
            )
        if len(recommendations) >= top_n:
            break

    return {
        "recommendations": recommendations,
        "not_found": not_found,
        "found": found_names,
    }
