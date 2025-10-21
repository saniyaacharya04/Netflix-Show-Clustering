import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer
from src.preprocess import load_and_preprocess
from src.visualize import plot_pca, plot_genre_heatmap_actual, plot_wordcloud

# ----------------------------
# Load Data
# ----------------------------
df = load_and_preprocess()

# ----------------------------
# Sidebar Controls
# ----------------------------
st.title("Netflix Clustering Dashboard")

cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

st.sidebar.subheader("Filter Shows")
all_genres = sorted({g for lst in df['genres'] for g in lst})
selected_genres = st.sidebar.multiselect("Genres", options=all_genres)
selected_ratings = st.sidebar.multiselect("Ratings", options=sorted(df['rating'].unique()))
selected_types = st.sidebar.multiselect("Type", options=sorted(df['type'].unique()))
year_range = st.sidebar.slider(
    "Release Year Range", int(df['release_year'].min()), int(df['release_year'].max()), (2010, 2025)
)

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
filtered_df = filtered_df[
    (filtered_df['release_year'] >= year_range[0]) & (filtered_df['release_year'] <= year_range[1])
]

if filtered_df.empty:
    st.warning("No shows found with selected filters.")
    st.stop()

# ----------------------------
# Prepare Features for Clustering
# ----------------------------
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(
    mlb.fit_transform(filtered_df['genres']),
    columns=mlb.classes_,
    index=filtered_df.index
)

X_filtered = pd.concat([filtered_df[['release_year','duration_minutes','rating_encoded']], genre_dummies], axis=1)
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
# PCA Visualization
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
filtered_df['PCA1'] = X_pca[:,0]
filtered_df['PCA2'] = X_pca[:,1]

st.subheader("Cluster Visualization (PCA Projection)")
st.plotly_chart(plot_pca(filtered_df))

# ----------------------------
# Genre Heatmap
# ----------------------------
st.subheader("Genre Distribution per Cluster")
heatmap_fig = plot_genre_heatmap_actual(filtered_df, genre_cols=mlb.classes_)
st.pyplot(heatmap_fig)

# ----------------------------
# Word Clouds per Cluster
# ----------------------------
st.subheader("Word Clouds of Descriptions per Cluster")
for c in sorted(filtered_df['Cluster'].unique()):
    st.markdown(f"**Cluster {c}**")
    text = " ".join(filtered_df[filtered_df['Cluster']==c]['description'].tolist())
    wc_fig = plot_wordcloud(text, max_words=max_words, color_map=color_map)
    st.pyplot(wc_fig)

# ----------------------------
# Top Shows per Cluster
# ----------------------------
st.subheader("Top Shows per Cluster")
for i in range(n_clusters):
    st.markdown(f"Cluster {i}")
    top_shows = filtered_df[filtered_df['Cluster']==i][['title','type','rating','duration','listed_in']].head(5)
    if not top_shows.empty:
        st.write(top_shows)
    else:
        st.write("No shows available for this cluster.")

# ----------------------------
# Download Filtered Clustered Data
# ----------------------------
st.subheader("Download Filtered Clustered Data")
st.download_button(
    "Download CSV",
    filtered_df.to_csv(index=False),
    file_name="filtered_clustered_netflix.csv"
)
