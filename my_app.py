# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:33:51 2022

@author: Satwik_Nvidia
"""

import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from joblib import Parallel, delayed
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=joblib.load('tfidf.pkl')
svc_from_joblib = joblib.load('svc.pkl')
app=FastAPI(title='sentiment model api',description="an simple test api",version="0.1")



#@app.get('/')
#def get_root():
#    return {'message':'sentiment_analysis_of_tweets_app'}

#@app.get('/sentiment_analysis')
#async def query_sentiment(text:str):
#    return analyze_sentiment(text)

@app.get("/predict-review")

def analyze_sentiment(text:str):
    
    tweet=[text]
    x_test=tfidf.transform(tweet)
    x_test=pd.DataFrame.sparse.from_spmatrix(x_test)
    prediction=svc_from_joblib.predict(x_test)
    return {'prediction': prediction.tolist()}

if __name__=='__main__':
    uvicorn.run(app)
    