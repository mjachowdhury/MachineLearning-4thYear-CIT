# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:40:41 2019

@author: Mohammed
"""

import numpy as np
import pandas as pd


'''
Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). The axis labels are collectively referred to as the index. The basic method to create a Series is to call:
'''
#s = pd.Series(data, index=[])

'''
Here, data can be many different things:

a Python dict
an ndarray
a scalar value (like 5)
The passed index is a list of axis labels. Thus, this separates into a few cases depending on what data is:
'''
s = pd.Series([1,3,5,np.nan, 6, 8])
print('Series Value \n: ',s)

'''
Series acts very similarly to a ndarray, and is a valid argument to most NumPy functions. However, operations such as slicing will also slice the index.
'''

print('printing index value :',s[0]) # printing index value
print('Printing index value before index 3 which ment to be from 0 -2 index \n:',s[:3])
 
'''
If data is an ndarray, index must be the same length as data. If no index is passed, one will be created having values [0, ..., len(data) - 1].
'''
sRandom = pd.Series(np.random.rand(5), index=['a', 'b', 'c', 'd', 'e'])
print(sRandom)

print(sRandom.index)

sRandom = pd.Series(np.random.rand(5))
print(sRandom)

 

'''
When the data is a dict, and an index is not passed, the Series index will be ordered by the dict’s insertion order, if you’re using Python version >= 3.6 and Pandas version >= 0.23.
If you’re using Python < 3.6 or Pandas < 0.23, and an index is not passed, the Series index will be the lexically ordered list of dict keys.
'''
#Series can be instantiated from dicts:
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

newSeries = pd.Series(np.random.rand(5), index=['a', 'b','c', 'd','e'])
print('Printing new series:\n',newSeries)
print('Random index value is:',newSeries['a'])
print('Random index value is:',newSeries['e'])





