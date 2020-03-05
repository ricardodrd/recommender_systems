#!/usr/bin/env python2
import numpy as np
import pandas as pd
df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],[7, 8, 9],[1, 8, 9]]), columns=['a', 'b', 'c'])
print(df2)
print(df2['a'].values[1:])
print(df2['a'].values[:-1])
nam = df2['a'].values[1:] != df2['a'].values[:-1]
print(nam)
nam = np.r_[True, nam]
df2['uid'] = np.cumsum(nam)
print(df2.head())
