# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:02:23 2019

@author: Mohammed
"""



#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

data = genfromtxt("product_images.csv" , delimiter=',')

 
#sneakers = data[data.label == 0]

#print('Sneakers : ', data)
size = len(data)
print('Size : ',size)
image_size = 28 # width and length
no_of_different_labels = 2
image_pixels = image_size * image_size

idx_list = [idx + 1 for idx, val in
            enumerate(data) if val.any() == 0]

res = [data[i: j] for i, j in
        zip([0] + idx_list, idx_list + 
        ([size] if idx_list[-1] != size else []))]

print("The list after splitting by a value : " + str(res)) 
#sneakerImage = np.load(data)

data[:10]

#print(data)