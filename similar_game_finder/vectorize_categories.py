import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: open csv and store it as dataframe using pandas
os.chdir('C:\\Users\\JongbeenSong\\Desktop\\23-1\\빅데이터 처리\\BDP-main\\similar_game_finder')
CSV_ADDRESS = 'data\\combined_data_for_similar_game_finder.csv'
df = pd.read_csv(CSV_ADDRESS, encoding='cp949')

# # Step 2: We'll going to make string to list using split(','). and for NULL, just empty list
df['Categories'] = df['Categories'].fillna('').apply(lambda x: x.split(',') if x != '' else [])
df['Genres'] = df['Genres'].fillna('').apply(lambda x: x.split(',') if x != '' else [])
df['Tags'] = df['Tags'].fillna('').apply(lambda x: x.split(',') if x != '' else [])

# # Steps 3: then, we will vectorize each cell.
vectorizer = CountVectorizer()

# # Tokenize and build vocab for Categories/Genres/Tags
vectorizer.fit([' '.join(row) for row in df['Categories']])
df['Categories'] = df['Categories'].apply(lambda x: vectorizer.transform([' '.join(x)]).toarray())

vectorizer.fit([' '.join(row) for row in df['Genres']])
df['Genres'] = df['Genres'].apply(lambda x: vectorizer.transform([' '.join(x)]).toarray())

vectorizer.fit([' '.join(row) for row in df['Tags']])
df['Tags'] = df['Tags'].apply(lambda x: vectorizer.transform([' '.join(x)]).toarray())

# # Step 4: save it as csv
os.chdir('C:\\Users\\JongbeenSong\\Desktop')
df.to_csv('combined_data_for_similar_game_finder_categories_genres_tags.csv', index=False)

print('work done')
