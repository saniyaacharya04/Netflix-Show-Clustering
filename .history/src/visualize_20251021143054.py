import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd

def plot_pca(df, x='PCA1', y='PCA2'):
    """
    PCA scatter plot colored by Cluster
    """
    fig = px.scatter(df, x=x, y=y, color='Cluster', hover_data=['title','type','listed_in'])
    return fig

def plot_genre_heatmap_actual(df):
    """
    Heatmap of actual genres per cluster.
    df['genres'] must be a list of genres.
    df['Cluster'] must exist.
    """
    # Explode the list of genres so each row has one genre
    exploded = df.explode('genres')
    
    # Pivot table: Cluster x Genre counts
    genre_cluster = exploded.pivot_table(index='Cluster', columns='genres', aggfunc='size', fill_value=0)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(genre_cluster, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_ylabel("Cluster")
    ax.set_xlabel("Genre")
    ax.set_title("Genre Distribution per Cluster")
    return fig

def plot_wordcloud(text, max_words=200, color_map='viridis'):
    """
    Generate a WordCloud plot from text
    """
    wc = WordCloud(width=800, height=400, background_color='white', max_words=max_words, colormap=color_map).generate(text)
    plt.figure(figsize=(10,4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return plt
