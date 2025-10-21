from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def perform_clustering(X, method="K-Means", n_clusters=5):
    if method == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return labels, score

def compare_clustering(X, n_clusters=5):
    # K-Means
    km_labels, km_score = perform_clustering(X, "K-Means", n_clusters)
    # Hierarchical
    hc_labels, hc_score = perform_clustering(X, "Hierarchical", n_clusters)
    
    return {
        "K-Means": {"labels": km_labels, "silhouette": km_score},
        "Hierarchical": {"labels": hc_labels, "silhouette": hc_score}
    }
