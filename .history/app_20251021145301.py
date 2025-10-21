import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from src.preprocess import load_data, feature_engineering
from src.cluster import perform_clustering, compare_clustering
from src.visualize import plot_pca_scatter, compare_pca_plots, generate_wordcloud

# Load & preprocess
df = feature_engineering(load_data())

# ----------------------
# Sidebar Controls
# ----------------------
st.title("Netflix Clustering Dashboard")

mode = st.sidebar.selectbox("Select Mode", ["Single Method", "Comparison Mode"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

# Filters
selected_genres = st.sidebar.multiselect("Genres", sorted({g for lst in df['genres'] for g in lst}))
selected_ratings = st.sidebar.multiselect("Ratings", sorted(df['rating'].unique()))
selected_types = st.sidebar.multiselect("Type", sorted(df['type'].unique()))
year_range = st.sidebar.slider("Release Year Range", int(df['release_year'].min()), int(df['release_year'].max()), (2010, 2025))

# WordCloud Settings
max_words = st.sidebar.slider("Max Words", 50, 500, 200)
color_map = st.sidebar.selectbox("Color Map", ["viridis","plasma","magma","cividis"])

# ----------------------
# Apply Filters
# ----------------------
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

# ----------------------
# Feature matrix
# ----------------------
mlb = MultiLabelBinarizer()
genres_dummies = pd.DataFrame(mlb.fit_transform(filtered_df['genres']), columns=mlb.classes_, index=filtered_df.index)
X = pd.concat([filtered_df[['release_year','duration_minutes','rating_encoded']], genres_dummies], axis=1)
X_scaled = StandardScaler().fit_transform(X)

# ----------------------
# Single vs Comparison Mode
# ----------------------
if mode == "Single Method":
    cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
    labels, score = perform_clustering(X_scaled, cluster_method, n_clusters)
    filtered_df['Cluster'] = labels
    st.sidebar.write(f"Silhouette Score: {score:.3f}")
    
    # PCA Plot
    X_pca = plot_pca_scatter(X_scaled, labels, f"{cluster_method} Clusters PCA")
    st.plotly_chart(X_pca)
    
elif mode == "Comparison Mode":
    comparison = compare_clustering(X_scaled, n_clusters)
    st.write("Silhouette Scores:")
    st.write(f"K-Means: {comparison['K-Means']['silhouette']:.3f}")
    st.write(f"Hierarchical: {comparison['Hierarchical']['silhouette']:.3f}")
    
    fig1, fig2 = compare_pca_plots(X_scaled, comparison['K-Means']['labels'], comparison['Hierarchical']['labels'])
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

# ----------------------
# Word Clouds
# ----------------------
st.subheader("Word Clouds per Cluster")
for c in sorted(filtered_df['Cluster'].unique()):
    st.markdown(f"**Cluster {c}**")
    text = " ".join(filtered_df[filtered_df['Cluster']==c]['description'].tolist())
    plt_wc = generate_wordcloud(text, max_words=max_words, colormap=color_map)
    if plt_wc:
        st.pyplot(plt_wc)
