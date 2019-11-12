# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:22:46 2019

@author: Mohammed
"""
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt 

 
# Load digits dataset
#digits = pd.read_csv("product_images.csv")

# Create feature matrix
#X = digits.data

# Create target vector
#y = digits.target

# View the first observation's feature values
#X[0]

import numpy
filename = 'product_images.csv'
raw_data = open(filename, 'rt',header=None)
data = numpy.loadtxt(raw_data, delimiter=",")
print(data.shape)