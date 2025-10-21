def filter_dataframe(df, genres=None, ratings=None, types=None, year_range=None):
    filtered = df.copy()
    if genres:
        filtered = filtered[filtered['genres'].apply(lambda x: any(g in x for g in genres))]
    if ratings:
        filtered = filtered[filtered['rating'].isin(ratings)]
    if types:
        filtered = filtered[filtered['type'].isin(types)]
    if year_range:
        filtered = filtered[(filtered['release_year'] >= year_range[0]) & (filtered['release_year'] <= year_range[1])]
    return filtered
