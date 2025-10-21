# ===============================================
# Ultimate Netflix Clustering Dashboard
# ===============================================
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
from wordcloud import WordCloud
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer

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

# ----------------------------
# Sidebar Controls
# ----------------------------
st.title("Ultimate Netflix Show Clustering Dashboard")
cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

# Filters
st.sidebar.subheader("Filter Shows")
selected_genres = st.sidebar.multiselect("Genres", options=mlb.classes_)
selected_ratings = st.sidebar.multiselect("Ratings", options=df['rating'].unique())
selected_types = st.sidebar.multiselect("Type", options=df['type'].unique())
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
filtered_genres_dummies = pd.DataFrame(MultiLabelBinarizer().fit_transform(filtered_df['genres']),
                                       columns=mlb.classes_, index=filtered_df.index)
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
fig = px.scatter(filtered_df, x='PCA1', y='PCA2', color='Cluster', hover_data=['title','type','listed_in'])
st.plotly_chart(fig)

# ----------------------------
# Genre Distribution Heatmap
# ----------------------------
st.subheader("Genre Distribution per Cluster")
genre_cluster = filtered_df.groupby('Cluster')[mlb.classes_].sum()
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(genre_cluster, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# ----------------------------
# Rating Distribution Boxplot
# ----------------------------
st.subheader("Rating Distribution per Cluster")
plt.figure(figsize=(8,5))
sns.boxplot(x='Cluster', y='rating_encoded', data=filtered_df, palette="Set3")
plt.xticks(rotation=45)
st.pyplot(plt)

# ----------------------------
# Shows Released Over Years
# ----------------------------
st.subheader("Shows Released Over Years per Cluster")
year_cluster = filtered_df.groupby(['release_year','Cluster']).size().reset_index(name='count')
fig = px.line(year_cluster, x='release_year', y='count', color='Cluster', markers=True)
st.plotly_chart(fig)

# ----------------------------
# Word Clouds per Cluster
# ----------------------------
st.subheader("Word Clouds of Descriptions per Cluster")
selected_cluster = st.selectbox("Select Cluster for Word Cloud", sorted(filtered_df['Cluster'].unique()))
text = " ".join(filtered_df[filtered_df['Cluster']==selected_cluster]['description'].tolist())
if text.strip() != "":
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=max_words, colormap=color_map).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
else:
    st.write("No description available for this cluster.")

# ----------------------------
# Cluster Cohesion Metrics
# ----------------------------
st.subheader("📏 Cluster Cohesion Metrics")
for c in filtered_df['Cluster'].unique():
    cluster_points = X_scaled[filtered_df['Cluster']==c]
    distances = pairwise_distances(cluster_points)
    avg_distance = np.mean(distances)
    st.write(f"Cluster {c} average distance: {avg_distance:.2f}")

# ----------------------------
# Top Shows per Cluster
# ----------------------------
st.subheader("🎥 Top Shows per Cluster")
for i in range(n_clusters):
    st.markdown(f"**Cluster {i}**")
    st.write(filtered_df[filtered_df['Cluster']==i][['title','type','rating','duration','listed_in']].head(5))

# ----------------------------
# Clustering Comparison
# ----------------------------
st.subheader("Clustering Comparison: K-Means vs Hierarchical")
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_scaled)
hc = AgglomerativeClustering(n_clusters=n_clusters).fit(X_scaled)
pca_km = PCA(n_components=2).fit_transform(X_scaled)
pca_hc = PCA(n_components=2).fit_transform(X_scaled)

fig = px.scatter(x=pca_km[:,0], y=pca_km[:,1], color=km.labels_, title="K-Means Clusters")
st.plotly_chart(fig)
fig = px.scatter(x=pca_hc[:,0], y=pca_hc[:,1], color=hc.labels_, title="Hierarchical Clusters")
st.plotly_chart(fig)

# ----------------------------
# Download Filtered Clustered Data
# ----------------------------
st.subheader("💾 Download Filtered Clustered Data")
st.download_button("Download CSV", filtered_df.to_csv(index=False), file_name="filtered_clustered_netflix.csv")
