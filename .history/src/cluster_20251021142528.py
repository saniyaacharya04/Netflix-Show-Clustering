from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd

def prepare_features(df, mlb):
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(df['genres']),
        columns=mlb.classes_,
        index=df.index
    )
    X = pd.concat([df[['release_year','duration_minutes','rating_encoded']], genre_dummies], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def cluster_data(X_scaled, method="K-Means", n_clusters=5):
    if method == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    return labels, silhouette

def compute_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_scaled)
