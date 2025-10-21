import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from .utils import extract_duration

def load_data(path='data/netflix_titles.csv'):
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
    df['duration'] = df['duration'].fillna("0 min")
    df['listed_in'] = df['listed_in'].fillna("Unknown")
    df['description'] = df['description'].fillna("")
    
    # Feature engineering
    df['duration_minutes'] = df['duration'].apply(extract_duration)
    le = LabelEncoder()
    df['rating_encoded'] = le.fit_transform(df['rating'])
    df['genres'] = df['listed_in'].apply(lambda x: [i.strip() for i in x.split(',')])
    return df

def get_filtered_df(df, genres=[], ratings=[], types=[], year_range=(2010,2025)):
    filtered_df = df.copy()
    if genres:
        filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: any(g in x for g in genres))]
    if ratings:
        filtered_df = filtered_df[filtered_df['rating'].isin(ratings)]
    if types:
        filtered_df = filtered_df[filtered_df['type'].isin(types)]
    filtered_df = filtered_df[(filtered_df['release_year'] >= year_range[0]) & (filtered_df['release_year'] <= year_range[1])]
    return filtered_df
