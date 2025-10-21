import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import pandas as pd

def plot_pca(X_scaled, df, title="Cluster Visualization (PCA Projection)"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:,0]
    df['PCA2'] = X_pca[:,1]
    fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', hover_data=['title','type','listed_in'], title=title)
    return fig

def genre_heatmap(filtered_df, genres_dummies):
    genre_cols = genres_dummies.columns.tolist()
    genre_cluster = pd.concat([filtered_df['Cluster'], genres_dummies], axis=1).groupby('Cluster')[genre_cols].sum()
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(genre_cluster, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    return fig

def rating_boxplot(filtered_df):
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Cluster', y='rating_encoded', data=filtered_df)
    plt.xticks(rotation=45)
    return plt

def cluster_cohesion(X_scaled, filtered_df):
    metrics = {}
    for c in filtered_df['Cluster'].unique():
        cluster_points = X_scaled[filtered_df['Cluster']==c]
        distances = pairwise_distances(cluster_points)
        avg_distance = distances.mean()
        metrics[c] = avg_distance
    return metrics

def compare_pca_plots(X_scaled, clustering_dict, title_prefix="Clustering Comparison"):
    """Return two PCA scatter plots side by side for K-Means and Hierarchical."""
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame({
        'PCA1': X_pca[:,0],
        'PCA2': X_pca[:,1]
    })
    
    plots = {}
    for method in clustering_dict.keys():
        df_pca['Cluster'] = clustering_dict[method]['labels']
        fig = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster',
                         title=f"{title_prefix}: {method}",
                         hover_data=df_pca.columns)
        plots[method] = fig
    return plots
