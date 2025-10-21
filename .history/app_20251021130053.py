# ===============================================
# Netflix Clustering Dashboard
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
    df = pd.read_csv('netflix_titles.csv')
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
st.title("🎬 Netflix Show Clustering Dashboard")
cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means", "Hierarchical"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

# ----------------------------
# Clustering
# ----------------------------
if cluster_method == "K-Means":
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
else:
    model = AgglomerativeClustering(n_clusters=n_clusters)

df['Cluster'] = model.fit_predict(X)
silhouette_avg = silhouette_score(X, df['Cluster'])
st.sidebar.write(f"Silhouette Score: {silhouette_avg:.3f}")

# ----------------------------
# PCA for Visualization
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

st.subheader("📊 Cluster Visualization (PCA Projection)")
fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', hover_data=['title','type','listed_in'])
st.plotly_chart(fig)

# ----------------------------
# Cluster Summary
# ----------------------------
st.subheader("📝 Cluster Summary")
summary = df.groupby('Cluster')[['release_year','duration_minutes']].mean()
st.dataframe(summary)

# ----------------------------
# Word Clouds per Cluster
# ----------------------------
st.subheader("☁️ Word Clouds of Descriptions per Cluster")
selected_cluster = st.selectbox("Select Cluster for Word Cloud", sorted(df['Cluster'].unique()))

text = " ".join(df[df['Cluster']==selected_cluster]['description'].tolist())
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
    st.write(df[df['Cluster']==i][['title','type','rating','duration','listed_in']].head(5))
