# src/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd
import streamlit as st  # For warnings and error messages

def plot_pca(df, x='PCA1', y='PCA2'):
    """
    Interactive PCA scatter plot using Plotly.

    Parameters:
    - df: pd.DataFrame with 'Cluster', x, y, 'title', 'type', 'listed_in'
    - x, y: column names for PCA axes

    Returns:
    - Plotly figure
    """
    if not all(col in df.columns for col in [x, y, 'Cluster']):
        st.warning("PCA columns or 'Cluster' column missing.")
        return px.scatter()  # empty figure
    fig = px.scatter(
        df, x=x, y=y, color='Cluster',
        hover_data=['title','type','listed_in'] if all(col in df.columns for col in ['title','type','listed_in']) else None,
        title="PCA Scatter Plot by Cluster"
    )
    return fig

def plot_genre_heatmap_actual(df, genre_cols=None):
    """
    Plots a heatmap of genre distribution across clusters.

    Parameters:
    - df: pd.DataFrame with at least 'Cluster' column and genre columns
    - genre_cols: list of genre column names (optional)
    
    Returns:
    - matplotlib figure object
    """
    if 'Cluster' not in df.columns:
        st.warning("'Cluster' column missing. Cannot plot genre heatmap.")
        return plt.figure()

    # Auto-detect genre columns if not provided
    if genre_cols is None:
        excluded_cols = ['title', 'type', 'rating', 'duration', 'listed_in',
                         'release_year', 'Cluster', 'PCA1', 'PCA2',
                         'rating_encoded', 'duration_minutes', 'genres', 'description']
        genre_cols = [c for c in df.columns if c not in excluded_cols]

    # Filter only columns that exist
    genre_cols = [c for c in genre_cols if c in df.columns]

    if len(genre_cols) == 0:
        st.warning("No genre columns available for heatmap.")
        return plt.figure()

    try:
        genre_cluster = pd.concat([df['Cluster'], df[genre_cols]], axis=1).groupby('Cluster').sum()
    except KeyError as e:
        st.error(f"Error creating genre heatmap: {e}")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(genre_cluster, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Genre Distribution by Cluster", fontsize=16)
    plt.tight_layout()
    return fig

def plot_wordcloud(text, max_words=200, color_map='viridis'):
    """
    Generates a word cloud from text.

    Parameters:
    - text: str, concatenated text to generate wordcloud
    - max_words: int, max words to display
    - color_map: str, matplotlib colormap

    Returns:
    - matplotlib figure
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        st.warning("No text provided for wordcloud.")
        return plt.figure()

    wc = WordCloud(width=800, height=400, background_color='white',
                   max_words=max_words, colormap=color_map).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Word Cloud", fontsize=16)
    plt.tight_layout()
    return fig
