# src/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_genre_heatmap(df, genre_dummies):
    # Merge cluster info with genre dummies
    data = pd.concat([df['Cluster'], genre_dummies], axis=1)
    genre_cluster = data.groupby('Cluster').sum()
    
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(genre_cluster, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    return fig
