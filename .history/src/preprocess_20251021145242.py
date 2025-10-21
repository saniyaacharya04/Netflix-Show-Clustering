import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def load_data(path='data/netflix_titles.csv'):
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    df['rating'].fillna(df['rating'].mode()[0], inplace=True)
    df['duration'].fillna("0 min", inplace=True)
    df['listed_in'].fillna("Unknown", inplace=True)
    df['description'].fillna("", inplace=True)
    return df

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

def feature_engineering(df):
    df['duration_minutes'] = df['duration'].apply(extract_duration)
    le = LabelEncoder()
    df['rating_encoded'] = le.fit_transform(df['rating'])
    df['genres'] = df['listed_in'].apply(lambda x: [i.strip() for i in x.split(',')])
    return df
