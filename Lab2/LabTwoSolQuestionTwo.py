# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:34:58 2019

@author: Mohammed
"""
'''
Question 2. In this question you will analyse the Titanic passenger dataset, which you can download from 
Canvas. Loadthe CSV file “titanic.csv” using Pandas.

A.How many passengers were on the titanic, and what percentage survived?
B.Determine the survival rate for male and female passengers. What can you observe?
C.What is the average fare paid by survivors compared to non-survivors?
D.Create a file “titanic_short.csv”containing only the nameand age of all surviving passengers, 
who boarded the Titanic in Queenstownand whoseage has beenrecorded in the data(this wereonly 8 passengers).

'''

import numpy as np
import pandas as pd

def titanic():
    data = pd.read_csv("titanic.csv")
    
    
    print('Total number of pasenger')
    
    totalPasenger = data['Name'].value_counts()
    print(totalPasenger)
    

titanic()