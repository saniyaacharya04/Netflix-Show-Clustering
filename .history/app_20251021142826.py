# ===============================================
# Netflix Intelligence Dashboard (Modular)
# ===============================================
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from src.preprocess import load_data, preprocess
from src.cluster import prepare_features, cluster_data, compute_pca
from src.visualize import plot_pca, plot_genre_heatmap, plot_wordcloud
from src.utils import filter_dataframe
from sklearn.metrics import pairwise_distances
import numpy as np

# ----------------------------
# Load & Preprocess Data
# ----------------------------
df = load_data()
df = preprocess(df)

# ----------------------------
# Sidebar Controls
# ----------------------------
st.title("📊 Netflix Intelligence Dashboard")
cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

# Filters
st.sidebar.subheader("Filter Shows")
all_genres = sorted({g for lst in df['genres'] for g in lst})
selected_genres = st.sidebar.multiselect("Genres", options=all_genres)
selected_ratings = st.sidebar.multiselect("Ratings", options=sorted(df['rating'].unique()))
selected_types = st.sidebar.multiselect("Type", options=sorted(df['type'].unique()))
year_range = st.sidebar.slider("Release Year Range", int(df['release_year'].min()), int(df['release_year'].max()), (2010, 2025))
if st.sidebar.button("Reset Filters"):
    selected_genres, selected_ratings, selected_types = [], [], []
    year_range = (int(df['release_year'].min()), int(df['release_year'].max()))

# Word Cloud settings
st.sidebar.subheader("Word Cloud Settings")
max_words = st.sidebar.slider("Max Words", 50, 500, 200)
color_map = st.sidebar.selectbox("Color Map", ["viridis","plasma","magma","cividis"])

# ----------------------------
# Apply Filters
# ----------------------------
filtered_df = filter_dataframe(df, genres=selected_genres, ratings=selected_ratings, types=selected_types, year_range=year_range)
if filtered_df.empty:
    st.warning("No shows found with selected filters.")
    st.stop()

# ----------------------------
# Prepare Features & Cluster
# ----------------------------
mlb = MultiLabelBinarizer()
X_scaled = prepare_features(filtered_df, mlb)
filtered_df['Cluster'], silhouette_avg = cluster_data(X_scaled, method=cluster_method, n_clusters=n_clusters)
st.sidebar.write(f"Silhouette Score: {silhouette_avg:.3f}")

# ----------------------------
# PCA for Visualization
# ----------------------------
pca_components = compute_pca(X_scaled)
filtered_df['PCA1'], filtered_df['PCA2'] = pca_components[:,0], pca_components[:,1]
st.subheader("Cluster Visualization (PCA Projection)")
st.plotly_chart(plot_pca(filtered_df))

# ----------------------------
# Genre Heatmap
# ----------------------------
st.subheader("Genre Distribution per Cluster")
genre_cols = mlb.classes_
st.pyplot(plot_genre_heatmap(filtered_df, genre_cols))

# ----------------------------
# Rating Distribution
# ----------------------------
st.subheader("Rating Distribution per Cluster")
plt.figure(figsize=(8,5))
import seaborn as sns
sns.boxplot(x='Cluster', y='rating_encoded', data=filtered_df, palette="Set3")
plt.xticks(rotation=45)
st.pyplot(plt)

# ----------------------------
# Shows Over Years
# ----------------------------
st.subheader("Shows Released Over Years per Cluster")
import plotly.express as px
year_cluster = filtered_df.groupby(['release_year','Cluster']).size().reset_index(name='count')
fig = px.line(year_cluster, x='release_year', y='count', color='Cluster', markers=True)
st.plotly_chart(fig)

# ----------------------------
# Word Clouds for All Clusters
# ----------------------------
st.subheader("Word Clouds of Descriptions per Cluster")
for c in sorted(filtered_df['Cluster'].unique()):
    st.markdown(f"**Cluster {c}**")
    text = " ".join(filtered_df[filtered_df['Cluster']==c]['description'].tolist())
    if text.strip():
        st.pyplot(plot_wordcloud(text, max_words=max_words, color_map=color_map))

# ----------------------------
# Cluster Cohesion Metrics
# ----------------------------
st.subheader("Cluster Cohesion Metrics")
for c in filtered_df['Cluster'].unique():
    cluster_points = X_scaled[filtered_df['Cluster']==c]
    avg_distance = np.mean(pairwise_distances(cluster_points))
    st.write(f"Cluster {c} average distance: {avg_distance:.2f}")

# ----------------------------
# Top Shows per Cluster
# ----------------------------
st.subheader("Top Shows per Cluster")
for i in range(n_clusters):
    st.markdown(f"Cluster {i}")
    st.write(filtered_df[filtered_df['Cluster']==i][['title','type','rating','duration','listed_in']].head(5))

# ----------------------------
# K-Means vs Hierarchical Comparison
# ----------------------------
st.subheader("K-Means vs Hierarchical Clusters")
from sklearn.cluster import KMeans, AgglomerativeClustering
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_scaled)
hc = AgglomerativeClustering(n_clusters=n_clusters).fit(X_scaled)
pca_km = compute_pca(X_scaled)
pca_hc = compute_pca(X_scaled)
fig_km = px.scatter(x=pca_km[:,0], y=pca_km[:,1], color=km.labels_, title="K-Means Clusters")
fig_hc = px.scatter(x=pca_hc[:,0], y=pca_hc[:,1], color=hc.labels_, title="Hierarchical Clusters")
st.plotly_chart(fig_km)
st.plotly_chart(fig_hc)

# ----------------------------
# Download CSV
# ----------------------------
st.subheader("Download Filtered Clustered Data")
st.download_button("Download CSV", filtered_df.to_csv(index=False), file_name="filtered_clustered_netflix.csv")
