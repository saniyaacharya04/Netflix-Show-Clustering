# app.py
import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances

from src.visualize import plot_pca, plot_genre_heatmap_actual, plot_wordcloud

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    path = 'data/netflix_titles.csv'
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    df['rating'].fillna(df['rating'].mode()[0], inplace=True)
    df['duration'].fillna("0 min", inplace=True)
    df['listed_in'].fillna("Unknown", inplace=True)
    df['description'].fillna("", inplace=True)
    df['genres'] = df['listed_in'].apply(lambda x: [g.strip() for g in x.split(',')])
    return df

df = load_data()

# ----------------------------
# Feature Engineering
# ----------------------------
def extract_duration(value):
    try:
        if 'min' in value:
            return int(value.split()[0])
        elif 'Season' in value:
            return int(value.split()[0]) * 60
        else:
            return 0
    except:
        return 0

df['duration_minutes'] = df['duration'].apply(extract_duration)
le = LabelEncoder()
df['rating_encoded'] = le.fit_transform(df['rating'])

# ----------------------------
# Sidebar Controls
# ----------------------------
st.title("Netflix Clustering Dashboard")

cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

# Filters
st.sidebar.subheader("Filter Shows")
all_genres = sorted({g for lst in df['genres'] for g in lst})
selected_genres = st.sidebar.multiselect("Genres", options=all_genres)
selected_ratings = st.sidebar.multiselect("Ratings", options=sorted(df['rating'].unique()))
selected_types = st.sidebar.multiselect("Type", options=sorted(df['type'].unique()))
year_range = st.sidebar.slider("Release Year Range", int(df['release_year'].min()), int(df['release_year'].max()), (2010, 2025))

# Word Cloud settings
st.sidebar.subheader("Word Cloud Settings")
max_words = st.sidebar.slider("Max Words", 50, 500, 200)
color_map = st.sidebar.selectbox("Color Map", ["viridis","plasma","magma","cividis"])

# ----------------------------
# Apply Filters
# ----------------------------
filtered_df = df.copy()
if selected_genres:
    filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: any(g in x for g in selected_genres))]
if selected_ratings:
    filtered_df = filtered_df[filtered_df['rating'].isin(selected_ratings)]
if selected_types:
    filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
filtered_df = filtered_df[(filtered_df['release_year'] >= year_range[0]) & (filtered_df['release_year'] <= year_range[1])]

if filtered_df.empty:
    st.warning("No shows found with selected filters.")
    st.stop()

# ----------------------------
# Prepare Features for Clustering
# ----------------------------
filtered_mlb = MultiLabelBinarizer()
filtered_genres_dummies = pd.DataFrame(
    filtered_mlb.fit_transform(filtered_df['genres']),
    columns=filtered_mlb.classes_,
    index=filtered_df.index
)

X_filtered = pd.concat([filtered_df[['release_year','duration_minutes','rating_encoded']], filtered_genres_dummies], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# ----------------------------
# Clustering
# ----------------------------
if cluster_method == "K-Means":
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
else:
    model = AgglomerativeClustering(n_clusters=n_clusters)

filtered_df['Cluster'] = model.fit_predict(X_scaled)
silhouette_avg = silhouette_score(X_scaled, filtered_df['Cluster'])
st.sidebar.write(f"Silhouette Score: {silhouette_avg:.3f}")

# ----------------------------
# PCA for Visualization
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
filtered_df['PCA1'] = X_pca[:,0]
filtered_df['PCA2'] = X_pca[:,1]

st.subheader("Cluster Visualization (PCA Projection)")
st.plotly_chart(plot_pca(filtered_df))

# ----------------------------
# Genre Distribution Heatmap
# ----------------------------
st.subheader("Genre Distribution per Cluster")
st.pyplot(plot_genre_heatmap_actual(filtered_df))

# ----------------------------
# Rating Distribution Boxplot
# ----------------------------
st.subheader("Rating Distribution per Cluster")
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,5))
sns.boxplot(x='Cluster', y='rating_encoded', data=filtered_df, palette="Set3")
plt.xticks(rotation=45)
st.pyplot(plt)

# ----------------------------
# Shows Released Over Years
# ----------------------------
st.subheader("Shows Released Over Years per Cluster")
import plotly.express as px
year_cluster = filtered_df.groupby(['release_year','Cluster']).size().reset_index(name='count')
fig = px.line(year_cluster, x='release_year', y='count', color='Cluster', markers=True)
st.plotly_chart(fig)

# ----------------------------
# Word Clouds per Cluster
# ----------------------------
st.subheader("Word Clouds of Descriptions per Cluster")
for c in sorted(filtered_df['Cluster'].unique()):
    st.markdown(f"**Cluster {c}**")
    text = " ".join(filtered_df[filtered_df['Cluster']==c]['description'].tolist())
    if text.strip():
        st.pyplot(plot_wordcloud(text, max_words=max_words, color_map=color_map))
    else:
        st.write("No description available for this cluster.")

# ----------------------------
# Cluster Cohesion Metrics
# ----------------------------
st.subheader("Cluster Cohesion Metrics")
for c in filtered_df['Cluster'].unique():
    cluster_points = X_scaled[filtered_df['Cluster']==c]
    distances = pairwise_distances(cluster_points)
    avg_distance = np.mean(distances)
    st.write(f"Cluster {c} average distance: {avg_distance:.2f}")

# ----------------------------
# Top Shows per Cluster
# ----------------------------
st.subheader("Top Shows per Cluster")
for i in range(n_clusters):
    st.markdown(f"Cluster {i}")
    st.write(filtered_df[filtered_df['Cluster']==i][['title','type','rating','duration','listed_in']].head(5))

# ----------------------------
# Download Filtered Clustered Data
# ----------------------------
st.subheader("Download Filtered Clustered Data")
st.download_button("Download CSV", filtered_df.to_csv(index=False), file_name="filtered_clustered_netflix.csv")
