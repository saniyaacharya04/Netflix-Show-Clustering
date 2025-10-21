import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd
import streamlit as st

# ----------------------------
# PCA Scatter Plot
# ----------------------------
def plot_pca(df, x='PCA1', y='PCA2'):
    """
    Scatter plot of PCA components colored by cluster.
    """
    fig = px.scatter(
        df, x=x, y=y, color='Cluster', hover_data=['title', 'type', 'listed_in'],
        color_continuous_scale=px.colors.qualitative.Set1
    )
    return fig


# ----------------------------
# Genre Heatmap per Cluster
# ----------------------------
def plot_genre_heatmap_actual(df, genre_cols=None):
    """
    Heatmap showing count of each genre per cluster.
    """
    if genre_cols is None or len(genre_cols) == 0:
        st.warning("No genre columns available for heatmap.")
        return plt.figure()

    missing_cols = [col for col in genre_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Some genre columns are missing in DataFrame: {missing_cols}")
        genre_cols = [col for col in genre_cols if col in df.columns]

    if len(genre_cols) == 0:
        st.warning("No valid genre columns found after checking DataFrame.")
        return plt.figure()

    try:
        genre_cluster = pd.concat([df['Cluster'], df[genre_cols]], axis=1).groupby('Cluster').sum()
    except Exception as e:
        st.error(f"Error creating genre heatmap: {e}")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(genre_cluster, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_ylabel("Cluster")
    ax.set_xlabel("Genre")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# ----------------------------
# Wordcloud Generator
# ----------------------------
def plot_wordcloud(text, max_words=200, color_map='viridis'):
    """
    Generate a word cloud from text.
    """
    if not text.strip():
        st.warning("No text available to generate wordcloud.")
        return plt.figure()

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        colormap=color_map
    ).generate(text)

    fig = plt.figure(figsize=(10, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    return fig
