import pandas as pd
import os
import re
import pickle
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

os.chdir('C:\\Users\\JongbeenSong\\Desktop\\23-1\\빅데이터 처리\\BDP-main\\similar_game_finder')

CSV_ADDRESS = 'data\\combined_data_for_similar_game_finder_merged.csv'

df = pd.read_csv(CSV_ADDRESS, encoding='cp949')

with open('vectorizer_reviews.pickle', 'rb') as f:
    vectorizer_reviews = pickle.load(f)

with open('vectorizer_descriptions.pickle', 'rb') as f:
    vectorizer_descriptions = pickle.load(f)

tfidf_matrix_reviews = load_npz('tfidf_matrix_reviews.npz')
tfidf_matrix_descriptions = load_npz('tfidf_matrix_descriptions.npz')

user_input = "THE LAW!!Looks to be a showdown atop a train. This will be your last fight. Good luck, Train Bandit.WHAT IS THIS GAME?Train Bandit is a simple score attack game. The Law will attack you from both sides.Your weapon is your keyboard. You'll use those keys to kick the living shit out of the law.React quickly by attacking the correct direction. React...or you're dead.THE FEATURES Unlock new bandits Earn Achievements Become Steam's Most Wanted? Battle elite officers Kick the law's ass"

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

user_input_vector = vectorizer_reviews.transform([preprocess_text(user_input)])

cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix_reviews)

top_indices = np.argsort(cosine_similarities[0])[-5:]
top_indices = top_indices[::-1]

print('The 5 most similar games to your input are:')
for idx in top_indices:
    print(idx)
    print(df.iloc[idx]['AppID'])
    print(df.iloc[idx]['Name'])
    print(df.iloc[idx]['Game Description'])
    print('\n')
