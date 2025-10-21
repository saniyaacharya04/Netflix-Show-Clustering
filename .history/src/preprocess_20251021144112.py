import pandas as pd

def load_and_preprocess(csv_path="data/netflix_titles.csv"):
    """
    Load Netflix dataset and preprocess for clustering and visualization.

    Args:
        csv_path (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame
    """

    # ----------------------------
    # Load Data
    # ----------------------------
    df = pd.read_csv(csv_path)

    # ----------------------------
    # Fill missing values safely
    # ----------------------------
    df = df.copy()  # ensure we have a copy to avoid chained assignment issues

    # Fill ratings with mode
    if 'rating' in df.columns:
        mode_rating = df['rating'].mode()[0]
        df['rating'] = df['rating'].fillna(mode_rating)

    # Fill duration
    if 'duration' in df.columns:
        df['duration'] = df['duration'].fillna("0 min")

        # Convert duration to numeric minutes
        def duration_to_minutes(x):
            try:
                return int(x.split()[0])
            except:
                return 0
        df['duration_minutes'] = df['duration'].apply(duration_to_minutes)

    # Fill genres
    if 'listed_in' in df.columns:
        df['listed_in'] = df['listed_in'].fillna("Unknown")
        df['genres'] = df['listed_in'].apply(lambda x: [g.strip() for g in x.split(",")])

    # Fill descriptions
    if 'description' in df.columns:
        df['description'] = df['description'].fillna("")

    # Encode ratings numerically
    rating_mapping = {r: i for i, r in enumerate(sorted(df['rating'].unique()))}
    df['rating_encoded'] = df['rating'].map(rating_mapping)

    # ----------------------------
    # Convert release_year to int
    # ----------------------------
    if 'release_year' in df.columns:
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
        df['release_year'] = df['release_year'].fillna(df['release_year'].median()).astype(int)

    return df
