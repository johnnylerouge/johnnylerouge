from email import message
import imp
from typing import Optional
from anyio import open_signal_receiver
from nbformat import read
import pandas
import numpy
from fastapi import Body, FastAPI
from pydantic import BaseModel
from pyrsistent import freeze
import joblib, os
from scipy.sparse import hstack
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
import sklearn
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from scipy.sparse import hstack
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
import re
from bs4 import BeautifulSoup
import string 

app=FastAPI()



tfidf_X1=joblib.load('tfidf_X1')
tfidf_X2=joblib.load('tfidf_X2')
reg=joblib.load('reg')
binarizer=joblib.load('binarizer')

token=ToktokTokenizer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = BeautifulSoup(text, 'lxml')
    text = text.get_text()
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)    
    text = text.strip(' ')
    punct = set(string.punctuation) 
    text = "".join([ch for ch in text if ch not in punct])
    stop_words = set(stopwords.words('english'))
    
    words=token.tokenize(text)
    
    text=[word for word in words if not word in stop_words]
    text=' '.join(map(str, text))
    text=token.tokenize(text)
    lemm_list=[]
    for word in text:
        x=lemmatizer.lemmatize(word, pos='v')
        lemm_list.append(x)
    text=' '.join(map(str, lemm_list))
    return text

def tags_normalization(text):
    text=text.replace('<','').replace('>', ' ')
    return text



class Item(BaseModel):
    content : str
    title : str



@app.post("/predict")
def tag_predict(question: Item, tfidf_X=tfidf_X1, tfidf_Y=tfidf_X2):
    unseen_data={'Title': preprocess(question.title), 'Body': preprocess(question.content)}
    unseen_data=pd.DataFrame(data=unseen_data, index=[0])
    tfidf_X=tfidf_X1.transform(unseen_data.Body)
    tfidf_Y=tfidf_X2.transform(unseen_data.Title)
    tfidf_unseen=hstack([tfidf_X, tfidf_Y])
    y_pred=reg.predict(tfidf_unseen)
    pred_list=binarizer.inverse_transform(y_pred)
    print (pred_list)
    return {"predicted tags": pred_list}

@app.get("/")
def read_root():
    return {"Hello": "World"}
