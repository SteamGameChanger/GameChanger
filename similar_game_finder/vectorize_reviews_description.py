from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import save_npz
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import re
import os

os.chdir('C:\\Users\\JongbeenSong\\Desktop\\23-1\\빅데이터 처리\\BDP-main\\similar_game_finder')
CSV_ADDRESS = 'data\\combined_data_for_similar_game_finder_review_and_description.csv'
df = pd.read_csv(CSV_ADDRESS, encoding='cp949')

# Replace NaN values with empty string
df['Currently popular'] = df['Currently popular'].fillna('')
df['Game Description'] = df['Game Description'].fillna('')

stop_words = set(stopwords.words('english'))

# Define a function for preprocessing
def preprocess_text(text):
    result = []
    word_tokens = word_tokenize(text)
    for word in word_tokens:
        word = re.sub('[^a-zA-Z0-9]', '', word).strip()
        if word not in stop_words:
            result.append(word)
    return ' '.join(result)

# Apply preprocessing and store result in a new column
df['Currently popular'] = df['Currently popular'].apply(preprocess_text)
df['Game Description'] = df['Game Description'].apply(preprocess_text)

# Generate TF-IDF vectors for reveiws
vectorize_reviews = TfidfVectorizer()
tfidf_matrix = vectorize_reviews.fit_transform(df['Currently popular'])

# Save the vectorizer
with open('vectorizer_reviews.pickle', 'wb') as f:
    pickle.dump(vectorize_reviews, f)

# Save the TF-IDF vectors
save_npz('tfidf_matrix_reviews.npz', tfidf_matrix)

# # If you want to replace the 'Currently popular' column with its TF-IDF vectors
# df['Currently popular'] = list(tfidf_matrix)

# Generate TF-IDF vectors for descriptions
vectorize_descriptions = TfidfVectorizer()
tfidf_matrix = vectorize_descriptions.fit_transform(df['Game Description'])

# Save the vectorizer
with open('vectorizer_descriptions.pickle', 'wb') as f:
    pickle.dump(vectorize_descriptions, f)

# Save the TF-IDF vectors
save_npz('tfidf_matrix_descriptions.npz', tfidf_matrix)

# # If you want to replace the 'Game Description' column with its TF-IDF vectors
# df['Game Description'] = list(tfidf_matrix)

# Save the result as a new csv file
# df.to_csv('data\\combined_data_for_similar_game_finder_review_and_description_tfidf.csv', index=False)

print('work done!')
