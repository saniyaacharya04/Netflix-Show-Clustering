import streamlit as st
from src.preprocess import load_data, get_filtered_df
from src.cluster import prepare_features, run_clustering
from src.visualize import plot_pca, genre_heatmap, rating_boxplot, cluster_cohesion
from src.utils import generate_wordcloud

# Load data
df = load_data()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.title("Netflix Clustering Dashboard")
cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

# Filters
all_genres = sorted({g for lst in df['genres'] for g in lst})
selected_genres = st.sidebar.multiselect("Genres", options=all_genres)
selected_ratings = st.sidebar.multiselect("Ratings", options=sorted(df['rating'].unique()))
selected_types = st.sidebar.multiselect("Type", options=sorted(df['type'].unique()))
year_range = st.sidebar.slider("Release Year Range", int(df['release_year'].min()), int(df['release_year'].max()), (2010,2025))

# Wordcloud settings
max_words = st.sidebar.slider("Max Words", 50, 500, 200)
color_map = st.sidebar.selectbox("Color Map", ["viridis","plasma","magma","cividis"])

# -----------------------------
# Filter Data
# -----------------------------
filtered_df = get_filtered_df(df, selected_genres, selected_ratings, selected_types, year_range)
if filtered_df.empty:
    st.warning("No shows found with selected filters.")
    st.stop()

# -----------------------------
# Prepare features and cluster
# -----------------------------
X_scaled, genres_dummies = prepare_features(filtered_df)
labels, silhouette_avg = run_clustering(X_scaled, cluster_method, n_clusters)
filtered_df['Cluster'] = labels
st.sidebar.write(f"Silhouette Score: {silhouette_avg:.3f}")

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("PCA Cluster Plot")
st.plotly_chart(plot_pca(X_scaled, filtered_df))

st.subheader("Genre Distribution Heatmap")
st.pyplot(genre_heatmap(filtered_df, genres_dummies))

st.subheader("Rating Distribution Boxplot")
st.pyplot(rating_boxplot(filtered_df))

st.subheader("Cluster Cohesion Metrics")
metrics = cluster_cohesion(X_scaled, filtered_df)
for c, dist in metrics.items():
    st.write(f"Cluster {c} average distance: {dist:.2f}")

st.subheader("Word Clouds per Cluster")
for c in sorted(filtered_df['Cluster'].unique()):
    st.markdown(f"**Cluster {c}**")
    text = " ".join(filtered_df[filtered_df['Cluster']==c]['description'].tolist())
    plt_wc = generate_wordcloud(text, max_words, color_map)
    if plt_wc:
        st.pyplot(plt_wc)

st.subheader("Top Shows per Cluster")
for i in range(n_clusters):
    st.markdown(f"Cluster {i}")
    st.write(filtered_df[filtered_df['Cluster']==i][['title','type','rating','duration','listed_in']].head(5))

# -----------------------------
# Clustering Comparison: K-Means vs Hierarchical
# -----------------------------
st.subheader("Clustering Comparison: K-Means vs Hierarchical")
comparison_dict = compare_clustering(X_scaled, n_clusters)
plots = compare_pca_plots(X_scaled, comparison_dict)

# Display PCA plots
st.plotly_chart(plots['K-Means'])
st.write(f"K-Means Silhouette Score: {comparison_dict['K-Means']['silhouette']:.3f}")
st.plotly_chart(plots['Hierarchical'])
st.write(f"Hierarchical Silhouette Score: {comparison_dict['Hierarchical']['silhouette']:.3f}")

# -----------------------------
# Download CSV
# -----------------------------
st.subheader("Download Filtered Clustered Data")
st.download_button("Download CSV", filtered_df.to_csv(index=False), file_name="filtered_clustered_netflix.csv")
