import pandas as df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
import torch
import os
import joblib
import numpy as np

os.chdir('C:\\Users\\JongbeenSong\\Desktop\\23-1\\빅데이터 처리\\BDP-main\\price_recommender')

class TrainVectorizer:
    def __init__(self, dataframe):
        self.df = dataframe

    def bert_encode(self, text):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model_bert = BertModel.from_pretrained('bert-base-uncased')

        encoded = tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=512, # specifies maximum length of input sentences
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
            truncation=True
        )

        return model_bert(encoded['input_ids'], attention_mask=encoded['attention_mask'])[0].mean(1).detach().cpu().numpy()

    def train_tfidf_vectorizer(self):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.df['Game Description'])
        y = self.df['Price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

        return X_train, X_test, y_train, y_test

    def train_bert_vectorizer(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model_bert = BertModel.from_pretrained('bert-base-uncased')

        X = self.df['Game Description'].apply(self.bert_encode).values.tolist()
        y = self.df['Price'].values

        X = np.array(X)
        y = np.array(y)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        joblib.dump(tokenizer, 'bert_tokenizer.pkl')
        joblib.dump(model_bert, 'bert_model.pkl')

        return X_train, X_test, y_train, y_test
