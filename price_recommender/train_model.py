import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import os
import joblib

os.chdir('C:\\Users\\JongbeenSong\\Desktop\\23-1\\빅데이터 처리\\BDP-main\\price_recommender')

class TrainModel:
    def __init__(self, dataframe):
        self.df = dataframe

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        model = XGBRegressor()
        model.fit(X_train, y_train)

        joblib.dump(model, 'xgb_model.pkl')

    def train_lgbm(self, X_train, X_test, y_train, y_test):
        model = LGBMRegressor()
        model.fit(X_train, y_train)

        joblib.dump(model, 'lgb_model.pkl')
