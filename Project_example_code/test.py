#!/usr/bin/env python3
import numpy as np
import pandas as pd
from nltk.corpus import stopwords 

# df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],[7, 8, 9],[1, 8, 9]]), columns=['a', 'b', 'c'])
# print(df2)
# print(df2['a'].values[1:])
# print(df2['a'].values[:-1])
# nam = df2['a'].values[1:] != df2['a'].values[:-1]
# print(nam)
# nam = np.r_[True, nam]
# df2['uid'] = np.cumsum(nam)
# print(df2.head())
# 

# stopWords = set(stopwords.words('norwegian'))
# print(stopWords)
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
   'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
   'Rank': [2, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
   'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
print(df)
print()
sd = df[df.duplicated(subset=['Team','Rank'], keep=False)]
print(sd)