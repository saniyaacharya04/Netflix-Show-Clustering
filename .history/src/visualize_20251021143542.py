# src/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st  # needed for Streamlit messaging

def plot_genre_heatmap_actual(df, genre_cols=None):
    """
    Plots a heatmap of genre distribution across clusters.

    Parameters:
    - df: pd.DataFrame with at least 'Cluster' column and genre columns
    - genre_cols: list of genre column names (optional)
    
    Returns:
    - matplotlib figure object
    """
    # Automatically detect genre columns if not provided
    if genre_cols is None:
        excluded_cols = ['title', 'type', 'rating', 'duration', 'listed_in',
                         'release_year', 'Cluster', 'PCA1', 'PCA2',
                         'rating_encoded', 'duration_minutes', 'genres', 'description']
        genre_cols = [c for c in df.columns if c not in excluded_cols]

    # Keep only columns that exist in df
    genre_cols = [c for c in genre_cols if c in df.columns]

    if len(genre_cols) == 0:
        st.warning("No genre columns available for heatmap.")
        return plt.figure()  # return empty figure so Streamlit doesn't break

    try:
        # Group by cluster and sum genres
        genre_cluster = pd.concat([df['Cluster'], df[genre_cols]], axis=1).groupby('Cluster').sum()
    except KeyError as e:
        st.error(f"Error in genre heatmap: {e}")
        return plt.figure()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(genre_cluster, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Genre Distribution by Cluster", fontsize=16)
    plt.tight_layout()
    return fig
