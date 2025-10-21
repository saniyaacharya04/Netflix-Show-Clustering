import streamlit as st
from src.preprocess import load_data, preprocess
from src.cluster import prepare_features, cluster_data, compute_pca
from src.visualize import plot_pca, plot_genre_heatmap, plot_wordcloud
from src.utils import filter_dataframe
from sklearn.preprocessing import MultiLabelBinarizer

df = load_data()
df = preprocess(df)
mlb = MultiLabelBinarizer()
