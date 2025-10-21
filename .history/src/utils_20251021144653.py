import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

def generate_wordcloud(text, max_words=200, color_map="viridis"):
    if not text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color='white', max_words=max_words, colormap=color_map).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return plt
