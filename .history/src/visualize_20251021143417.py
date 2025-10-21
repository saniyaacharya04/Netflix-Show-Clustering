import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd

def plot_pca(df, x='PCA1', y='PCA2'):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color='Cluster',
        hover_data=['title', 'type', 'listed_in']
    )
    return fig

def plot_genre_heatmap_actual(df, genre_cols=None, st=None):
    if genre_cols is None:
        genre_cols = [c for c in df.columns if c not in ['title','type','rating','duration','listed_in','release_year','Cluster','PCA1','PCA2','rating_encoded','duration_minutes','genres','description']]
    
    genre_cols = [c for c in genre_cols if c in df.columns]

    if len(genre_cols) == 0:
        if st:
            st.warning("No genre columns available for heatmap.")
        return plt.figure()  # return empty figure

    genre_cluster = pd.concat([df['Cluster'], df[genre_cols]], axis=1).groupby('Cluster').sum()
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(genre_cluster, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    return fig


def plot_wordcloud(text, max_words=200, color_map='viridis'):
    wc = WordCloud(width=800, height=400, background_color='white',
                   max_words=max_words, colormap=color_map).generate(text)
    plt.figure(figsize=(10,4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return plt
