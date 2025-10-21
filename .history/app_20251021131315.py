# ===============================================
# Netflix Clustering Dashboard (Enhanced)
# ===============================================
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import plotly.express as px

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/netflix_titles.csv')
    df = df.drop(['show_id', 'cast', 'director'], axis=1)
    df.drop_duplicates(inplace=True)
    df['rating'].fillna(df['rating'].mode()[0], inplace=True)
    df['duration'].fillna("0 min", inplace=True)
    df['listed_in'].fillna("Unknown", inplace=True)
    df['description'].fillna("", inplace=True)
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
df['genres'] = df['listed_in'].apply(lambda x: [i.strip() for i in x.split(',')])
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_, index=df.index)
features = pd.concat([df[['release_year', 'duration_minutes', 'rating_encoded']], genre_dummies], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(features)

# ----------------------------
# Sidebar Options
# ----------------------------
st.title("🎬 Netflix Show Clustering Dashboard (Enhanced)")
cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

# Dynamic filters
st.sidebar.subheader("Filter Shows")
selected_genres = st.sidebar.multiselect("Select Genres", options=mlb.classes_)
selected_ratings = st.sidebar.multiselect("Select Ratings", options=df['rating'].unique())
selected_types = st.sidebar.multiselect("Select Type", options=df['type'].unique())
year_range = st.sidebar.slider("Release Year Range", int(df['release_year'].min()), int(df['release_year'].max()), (2010, 2025))

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
# Clustering on Filtered Data
# ----------------------------
# Select features for clustering
filtered_genres = MultiLabelBinarizer()
filtered_genres_dummies = pd.DataFrame(filtered_genres.fit_transform(filtered_df['genres']),
                                       columns=filtered_genres.classes_, index=filtered_df.index)
filtered_features = pd.concat([filtered_df[['release_year', 'duration_minutes', 'rating_encoded']], filtered_genres_dummies], axis=1)
X_filtered = scaler.fit_transform(filtered_features)

if cluster_method == "K-Means":
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
else:
    model = AgglomerativeClustering(n_clusters=n_clusters)

filtered_df['Cluster'] = model.fit_predict(X_filtered)
silhouette_avg = silhouette_score(X_filtered, filtered_df['Cluster'])
st.sidebar.write(f"Silhouette Score: {silhouette_avg:.3f}")

# ----------------------------
# PCA for Visualization
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_filtered)
filtered_df['PCA1'] = X_pca[:, 0]
filtered_df['PCA2'] = X_pca[:, 1]

st.subheader("📊 Cluster Visualization (PCA Projection)")
fig = px.scatter(filtered_df, x='PCA1', y='PCA2', color='Cluster', hover_data=['title','type','listed_in'])
st.plotly_chart(fig)

# ----------------------------
# Cluster Summary
# ----------------------------
st.subheader("📝 Cluster Summary")
summary = filtered_df.groupby('Cluster')[['release_year','duration_minutes']].mean()
st.dataframe(summary)

# ----------------------------
# Word Clouds per Cluster
# ----------------------------
st.subheader("☁️ Word Clouds of Descriptions per Cluster")
selected_cluster = st.selectbox("Select Cluster for Word Cloud", sorted(filtered_df['Cluster'].unique()))
text = " ".join(filtered_df[filtered_df['Cluster']==selected_cluster]['description'].tolist())
if text.strip() != "":
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
else:
    st.write("No description available for this cluster.")

# ----------------------------
# Top Shows per Cluster
# ----------------------------
st.subheader("🎥 Top Shows per Cluster")
for i in range(n_clusters):
    st.markdown(f"**Cluster {i}**")
    st.write(filtered_df[filtered_df['Cluster']==i][['title','type','rating','duration','listed_in']].head(5))
