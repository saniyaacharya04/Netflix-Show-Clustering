import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from sklearn.decomposition import PCA

def pca_projection(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    return X_pca

def plot_pca_scatter(X_pca, labels, title="PCA Scatter"):
    fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=labels, title=title)
    return fig

def compare_pca_plots(X, km_labels, hc_labels):
    X_pca = pca_projection(X, km_labels)
    fig1 = plot_pca_scatter(X_pca, km_labels, "K-Means PCA")
    fig2 = plot_pca_scatter(X_pca, hc_labels, "Hierarchical PCA")
    return fig1, fig2

def generate_wordcloud(text, max_words=200, colormap="viridis"):
    if text.strip() == "":
        return None
    wc = WordCloud(width=800, height=400, background_color='white',
                   max_words=max_words, colormap=colormap).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return plt
