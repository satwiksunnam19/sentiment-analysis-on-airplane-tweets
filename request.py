# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:21:41 2022

@author: Satwik_Nvidia
"""

import requests

query={'text':'this was a bad flight'}
response=requests.get('http://127.0.0.1:8000/sentiment_analyis/',params=query)
print(response.json())