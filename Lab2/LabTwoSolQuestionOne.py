# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:01:52 2019

@author: Mohammed
"""

'''
Question1. In this question you will go back to the bike sharing dataset from the previous lab available
 athttps://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset.This time loadthe CSV file “day.csv” using Pandas.
 
 A.Compare the average number of casual rentals and registered rentals depending on if it is a holiday 
 or not. What can you observe?
 
 B.The temperature values are already normalised for classification.What is the minimum and maximum 
 temperature in the dataset in Celsius?
 
 C.Usually there are more registered than casual renters. On which days in the data set is this not the case?
 
 D.Plot the temperatures against the number of casual and registered rentals. What can you observe?
 
 E.Create a 3d plot, which plots a point with coordinates “temp”, “hum”, and “windspeed”for each day. 
 Now dividethe dataset into two equal sized subsets, representing days with few casual rentals and 
 days with a lot of casual rentals(Hint: use np.median()). Colour the dots in your plot correspondingto 
 busy days red and the dots corresponding to non-busy daysgreen.
'''
import numpy as np
import pandas as pd
import matplotlib as plt

def bike():
    data = pd.read_csv("F:\MachineLearning\Lab1\day.csv")
    
    non_holiday = data[data['holiday'] == 0] [['casual', 'registered']]
    holiday = data[data['holiday'] == 1] [['casual', 'registered']]
    
    #A
    print('Average number of rental depend on holidays.\n')
    print('During holiday :',np.mean(holiday))
    print('During non-holiday: ',np.mean(non_holiday))
    print()
    #B
    print('Finding out min and max temp in celsius.\n')
    temp = data['temp'] * (39-(-8)) + (-8)
    
    print ('Maximum tempature:', np.max(temp) )
    print ('Minimum tempature:', np.min(temp) )
    print()
    #c
    print (data[data['registered'] < data['casual']][['dteday']])
    
    #d
    
    '''
    plt.figure()
    plt.scatter(temp, data['registered'], color='b')
    plt.scatter(temp, data['casual'], color = 'r')
    
    threshold = np.median(data['casual'])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection= '3d')
    ax.scatter(data[data['casual']<=threshold]['temp'],
               data[data['casual']<=threshold]['hum'],
               data[data['casual']<=threshold]['winspreed'], color='g')
    ax.scatter(data[data['casual']>threshold]['temp'],
               data[data['casual']>threshold]['hum'],
               data[data['casual']>threshold]['winspeed'], color ='r')
    
    '''
bike()  