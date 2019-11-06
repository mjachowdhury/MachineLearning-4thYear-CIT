# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:36:51 2019

@author: Mohammed
"""

import pandas as pd
import numpy as np
import re
from pandas import ExcelWriter
from pandas import ExcelFile
from collections import Counter
import seaborn as sns
import math


minimumWordLength=0
minimumWordOccurences=0


'''
    : Main Function
'''
def main():
    taskOne()
    taskTwo(trainDataList,minimumWordLength,minimumWordOccurences)
    taskThree(positiveReviewDataF,negativeReviewDataF, wordList)
    taskFour(positiveReviewOccurences, negativeReviewOccurences, trainPositiveReviewCount, trainNegativeReviewCount)
    taskFive(trainDataList, positiveLikeliHood,negativeLikeliHood, priorProbabilityPositive, priorProbabilityNegative)
    
'''
    : TaskOne Function Spliting and counting the review
'''
def taskOne():
    '''
        : Global Variables 
    '''
    global trainDataList
    global testDataList
    global trainLabelList
    global testLabelList
    global positiveReviewDataF
    global negativeReviewDataF
    global trainingRevs
    global trainPositiveReviewCount
    global trainNegativeReviewCount
    global trainingDataF
        
    '''
        :Reading Data From File  
    '''
    data = pd.read_excel('movie_reviews.xlsx')
     
    '''
         :Spliting Data Based on the Split (Train / Test)
    '''
    trainingDataF = data[['Review', 'Sentiment', 'Split']][data['Split'] == 'train']
    testDataF = data[['Review', 'Sentiment', 'Split']][data['Split'] == 'test']
     
     
     
    '''
        :Converting the DataFrame Review (train/test) to List
    '''
    trainDataList = trainingDataF['Review'].tolist()
    #trainDataList = trainingDataF['Review'].str.split()
    testDataList = testDataF['Review'].tolist()
    print(type(trainDataList))
    '''
        :Converting the DataFrame Sentiment (train/test) to List
    '''
    trainLabelList = trainingDataF['Sentiment'].tolist()
    testLabelList = testDataF['Sentiment'].tolist()
     
    '''
        :Separating Data From Training based on Sentiment - Positive/Negative
    '''
    positiveReviewDataF = trainingDataF[trainingDataF['Sentiment'] == 'positive']
    negativeReviewDataF = trainingDataF[trainingDataF['Sentiment'] == 'negative']
     
    '''
         :Separating Data From Test based on Sentiment - Positive/Negative
    '''   
    testpositiveReviewDataF = testDataF[testDataF['Sentiment'] == 'positive']
    testnegativeReviewDataF = testDataF[testDataF['Sentiment'] == 'negative']
     
    '''
         :Total Positive and Negative review from trining data
    ''' 
    trainPositiveReviewCount = positiveReviewDataF.shape[0]
    trainNegativeReviewCount = negativeReviewDataF.shape[0]
     
    '''
        :Total Positive and Negative review from test data
    '''
    testPositiveReviewCount = testpositiveReviewDataF.shape[0]
    testNegativeReviewCount = testnegativeReviewDataF.shape[0]
     
    '''
         :Printing From Training Data Set Positive and Negative Review Counts
    ''' 
    print("Totel Number of Positive Reviews in the Training Set: " , trainPositiveReviewCount)
    print("Total Number of Negative Reviews in the Training set: " , trainNegativeReviewCount)
    '''
        :Printing from Test Data Set Positive and Negative Review Counts
    '''
    print("Total Number of Positive Reviews in the Test Set: ", testPositiveReviewCount)
    print("Total Number of Negative Reviews in the Test Set: ", testNegativeReviewCount)
     
     #print('TRAINING DATA REVIEW AS LIST: ', positiveReviewDF)
     #sns.countplot(x='Sentiment', data=data) #will show the plot
    '''
         :Returning Four Lists
    ''' 
 
    return trainDataList, testDataList, trainLabelList, testLabelList

'''
    : Task Two Function. Removing non-alphanumeric char and checking conditions
'''

def taskTwo(trinDataList, minWordLen, minWordOcc):
    #Created global variable
    global wordList
    global trainAllWords
    
    minWordLen = int(input("ENTER MINUMUM LENGTH OF THE WORDS : "))
    minWordOcc = int(input("ENTER MINIMUM WORD OCCURANCE : "))
    
    
    #Created dic
    countWords = dict()
    #print('Printing one review ', trinDataList[1])
    #Removing all the non-alphanumeric char
    for word in trinDataList:
        word = word.lower()
        word.split()
        s = ""
        s = word
        s = re.sub("[^a-zA-Z0-9!]", ' ', word)
        words = s.split()

    #print('Words as a List', words)
    #print('printing single value: ', words[0])
    #Checking all the words based on the condition minimum word length
    for word in words:
        if len(word) > minWordLen:
            if word in countWords:

                countWords[word] += 1

            else:
                countWords[word] = 1
    print("Length of the word : " , countWords)
    #print('Total Words Count After Removing Non-Alphanumeric Characters : ',countWords)
    
    #Checking minimum word occurance condition
    #for key,val in list(countWords.items()):
     #   print("Name of the words: ", words, "number of times appear:--->", minWordOcc)
     #   if val < minWordOcc:
         #del countWords[key]
         
    #Converting dic to list  
    wordList = list(countWords.keys())
    #print('Word List: ', wordList )
    
    return wordList

'''
    :Task Three checking and comparing word list with review training data 
'''
def taskThree(posReview, negReview, wordL):
    
    #Created global variable
    global positiveReviewOccurences
    global negativeReviewOccurences
    
    #Created map
    positiveReviewOccurences = {}
    negativeReviewOccurences = {}
    
    #Converting dataframe to series
    posSeries = pd.Series(posReview['Review'], index=posReview.index)
    negSeries = pd.Series(negReview['Review'], index=negReview.index)
 
    #Checking and comparing wordList wtih positive review training data
    for word in wordL:
        count = 0
        for s in posSeries:
            if(' ' + word + ' ' ) in (' ' + s + ' '):
                count = count + 1
        positiveReviewOccurences[word] = count # Adding word to the dic
        
    #print("Positive Frequncy occurance: ",positiveReviewOccurences)
    
    print('\n')
    
    #Checking and comparing wordList wtih negative review training data
    for negWord in wordL:
        counts = 0
        for j in negSeries:
            if(' ' + negWord + ' ' ) in (' ' + j + ' '):
                counts = counts + 1
        negativeReviewOccurences[negWord] = counts # Adding word to the dic
        
    #print("Negative Fequency Occurance : ",negativeReviewOccurences)
    
    return positiveReviewOccurences, negativeReviewOccurences

'''
    : Task Four Finding the probability and prior
'''
def taskFour(posReviewOcc, negReviewOcc, posReviewCount, negReviewCount):
    #may be i do not need this
    global priorProbabilityPositive
    global priorProbabilityNegative
    global positiveLikeliHood
    global negativeLikeliHood
    #global priorProbability
    #global probabilityOfLikelihood
    
    priorProbabilityPositive = (posReviewCount) / (posReviewCount + negReviewCount)   
    #probPos = (posReviewCount + negReviewCount) / (posReviewCount)#my try
    
    priorProbabilityNegative = (negReviewCount) / (posReviewCount + negReviewCount)
    #probNeg = (posReviewCount + negReviewCount) / (negReviewCount)

    #may be i do not need htis
    #priorProbability = priorProbabilityPositive + priorProbabilityNegative
    
    positiveLikeliHood = dict()
    negativeLikeliHood = dict()

    laplace = 1

    for key, value in posReviewOcc.items():
        m = posReviewCount
        x =  (value + laplace) / (priorProbabilityPositive + (m * laplace))
        positiveLikeliHood[key] = x

    for key, value in negReviewOcc.items():
        m2 = negReviewCount
        y =  (value + laplace) / (priorProbabilityNegative + (m2 * laplace))
        negativeLikeliHood[key] = y

    print("\n ================PRIOR FOR POSITIVE ==================\n")
    print(priorProbabilityPositive)
    
    print("\n\t\t*********** PROBABILITY OF EACH WORDS FOR POSITIVE***************\n")
    #print(positiveLikeliHood)
    
       
    
    print("\n\t\t ==================PRIOR FOR NEGATIVE ===================\n")
    print(priorProbabilityNegative)

    print("\n\t\t*************** PROBABILITY OF EACH WORDS FOR NEGATIVE********\n")
    #print(negativeLikeliHood)
    
    #may be i do not need htis
    #probabilityOfLikelihood = {**positiveLikeliHood ,**negativeLikeliHood}
    
    return positiveLikeliHood, negativeLikeliHood, priorProbabilityPositive, priorProbabilityNegative

def taskFive(inputStr,positiveLikeliHood, negativeLikeliHood, priorProbabilityPositive,priorProbabilityNegative):
    
    log_likelihoodPositive = 0
    log_likelihoodNegative = 0
    
    
    for word in inputStr:
        #print(word)
        for feature in positiveLikeliHood:
            #print(feature)
            if word == feature:
                print("Positive likelihood: ", positiveLikeliHood[feature])
                 
    
    for word in inputStr:
        #print(word)
        for feature in negativeLikeliHood:
            #print(feature)
            if word == feature:
                print("Negative likelihood: ",negativeLikeliHood[feature])
                
    print(log_likelihoodPositive)
    print(log_likelihoodNegative)
 
'''
    : Calling main
'''    
    
           
main()             