# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:40:41 2019

@author: Mohammed
"""

import numpy as np
import pandas as pd


s = pd.Series([1,3,5,np.nan, 6, 8])
print(s)

sRandom = pd.Series(np.random.rand(5), index=['a', 'b', 'c', 'd', 'e'])
print(sRandom)

print(sRandom.index)

sRandom = pd.Series(np.random.rand(5))
print(sRandom)

'''
When the data is a dict, and an index is not passed, the Series index will be ordered by the dict’s insertion order, if you’re using Python version >= 3.6 and Pandas version >= 0.23.
If you’re using Python < 3.6 or Pandas < 0.23, and an index is not passed, the Series index will be the lexically ordered list of dict keys.
'''
d = {'b': 1, 'a': 0, 'c': 2}

dictionary= pd.Series(d)

print(dictionary)

d1= pd.Series(dictionary, index=['a', 'b', 'c', 'd'])

print(d1)

'''
If data is a scalar value, an index must be provided. The value will be repeated to match the length of index.
'''
scalarValue = pd.Series(5., index=['A','B','C','D','E'])
print(scalarValue)
