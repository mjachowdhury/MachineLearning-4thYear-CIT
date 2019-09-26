# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:09:19 2019

@author: Mohammed
"""

import numpy as np
import pandas as pd


def bikes():
    
    df = pd.read_csv("F:\MachineLearning\Lab1\day.csv")
    #print(df)
    #print(df.describe())
    #print(df['holiday'])
    
    totalNumberOfHolidays = (df['holiday'].value_counts())
    print('Total number of holidays: ', totalNumberOfHolidays)
    
    #notHoliday = (df['holiday'].value_counts() == 0)
    
    #if((df['holiday'].value_counts()) == 0):
    #print('Average number of casula rentals:', np.mean(df['casual']) and (df['holiday'].value_counts() == 0))
    
    print('Average number of casula rentals:', np.mean(df['casual']))
    print('Average number of registered rentals:', np.mean(df['registered']))
    
    
    #B-maximum and minimum tempature
    
    print('Finding the minimum tempature:', np.min(df['temp']))
    print('Finding the minimum tempature:', np.max(df['temp']))
    
    
    
    '''
    for line in f:
        elements = line.split(",")
        if elements[0] != "instant":
            casual = int(elements[14])
            #registered = int(elements[15])
            holidayNumber = int(elements[6])
            if casual == holidays:
                casualNnumberOfRentals = casualNnumberOfRentals + holidayNumber
                numberOfDays = numberOfDays + 1
    f.close()
    print("Number of Days: ", numberOfDays)
    print("Average number of rentals: ", casualNnumberOfRentals / numberOfDays)
    '''
bikes()