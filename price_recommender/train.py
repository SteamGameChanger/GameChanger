from train_model import TrainModel
from train_vectorizer import TrainVectorizer
import pandas as pd
import os

os.chdir('C:\\Users\\JongbeenSong\\Desktop\\23-1\\빅데이터 처리\\BDP-main\\price_recommender')

df = pd.read_csv('data\\combined_data_for_price_recommender_merged.csv', encoding='utf-8')

tv = TrainVectorizer(df)
X_train, X_test, y_train, y_test = tv.train_tfidf_vectorizer()
# X_train, X_test, y_train, y_test = tv.train_bert_vectorizer()

tm = TrainModel(df)
tm.train_xgboost(X_train, X_test, y_train, y_test)
tm.train_lgbm(X_train, X_test, y_train, y_test)
