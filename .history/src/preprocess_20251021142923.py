# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler

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

def prepare_features(df):
    # Duration
    df['duration_minutes'] = df['duration'].apply(extract_duration)
    # Rating encoding
    le = LabelEncoder()
    df['rating_encoded'] = le.fit_transform(df['rating'])
    # Genres
    df['genres'] = df['listed_in'].apply(lambda x: [i.strip() for i in x.split(',')])
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(df['genres']),
        columns=mlb.classes_,
        index=df.index
    )
    # Combine all features
    X = pd.concat([df[['release_year','duration_minutes','rating_encoded']], genre_dummies], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, genre_dummies
