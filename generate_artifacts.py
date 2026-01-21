import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

def load_data():
    print("Loading datasets...")
    # 1. Load spam.csv
    try:
        df1 = pd.read_csv('spam.csv', encoding='latin-1')
        df1 = df1.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, errors='ignore')
        df1 = df1.rename(columns={"v1":"label", "v2":"sms"})
    except Exception as e:
        print(f"Error loading spam.csv: {e}")
        return None

    # 2. Load combined_dataset.csv
    try:
        df2 = pd.read_csv("combined_dataset.csv")
        df2 = df2.rename(columns={"target":"label", "text":"sms"})
    except Exception as e:
        print(f"Error loading combined_dataset.csv: {e}")
        return None

    # 3. Concatenate
    df_sms = pd.concat([df1, df2], ignore_index=True)
    
    # Map label to int
    df_sms['label'] = df_sms['label'].map({'ham':0, 'spam':1, 0:0, 1:1})
    
    # Drop NaNs
    df_sms.dropna(subset=['label', 'sms'], inplace=True)
    df_sms['label'] = df_sms['label'].astype(int)
    
    print(f"Data loaded. Shape: {df_sms.shape}")
    print(df_sms['label'].value_counts())
    return df_sms

def train_and_save(df):
    print("Training model...")
    
    X = df['sms']
    y = df['label']
    
    # Vectorizer
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)
    
    # Model
    model = MultinomialNB()
    model.fit(X_vec, y)
    
    print("Saving artifacts...")
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Done. Artifacts saved: spam_model.pkl, vectorizer.pkl")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        train_and_save(df)
