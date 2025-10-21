import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def load_and_preprocess(path='data/netflix_titles.csv'):
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    
    # Avoid chained assignment warnings
    df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
    df['duration'] = df['duration'].fillna("0 min")
    df['listed_in'] = df['listed_in'].fillna("Unknown")
    df['description'] = df['description'].fillna("")

    # Extract numeric duration
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
    
    # Encode rating
    le = LabelEncoder()
    df['rating_encoded'] = le.fit_transform(df['rating'])
    
    # Split genres
    df['genres'] = df['listed_in'].apply(lambda x: [i.strip() for i in x.split(',')])
    
    return df
