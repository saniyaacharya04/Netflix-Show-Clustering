from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd

def prepare_features(filtered_df):
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genres_dummies = pd.DataFrame(mlb.fit_transform(filtered_df['genres']),
                                  columns=mlb.classes_,
                                  index=filtered_df.index)
    X = pd.concat([filtered_df[['release_year','duration_minutes','rating_encoded']], genres_dummies], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, genres_dummies

def run_clustering(X_scaled, method="K-Means", n_clusters=5):
    if method == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    return labels, silhouette_avg

def compare_clustering(X_scaled, n_clusters=5):
    """Run K-Means and Hierarchical clustering for comparison."""
    
    # K-Means
    km_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km_model.fit_predict(X_scaled)
    km_silhouette = silhouette_score(X_scaled, km_labels)
    
    # Hierarchical
    hc_model = AgglomerativeClustering(n_clusters=n_clusters)
    hc_labels = hc_model.fit_predict(X_scaled)
    hc_silhouette = silhouette_score(X_scaled, hc_labels)
    
    return {
        'K-Means': {'labels': km_labels, 'silhouette': km_silhouette},
        'Hierarchical': {'labels': hc_labels, 'silhouette': hc_silhouette}
    }