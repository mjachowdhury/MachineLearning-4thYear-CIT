# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:02:23 2019

@author: Mohammed
"""

import pandas as pd
import numpy as np
import re
from pandas import ExcelWriter
from pandas import ExcelFile
from collections import Counter
import seaborn as sns

minimumWordLength = 2
minimumWordOccurences = 2
inputStrNewReview = 'I used to love my country. now I do not like my coountry'

'''
    : Main Function
'''
def main():
    taskOne()
    taskTwo(trainDataList,minimumWordLength,minimumWordOccurences)
    taskThree(positiveReviewDataF,negativeReviewDataF, wordList)
    taskFour(positiveReviewOccurences, negativeReviewOccurences, trainPositiveReviewCount, trainNegativeReviewCount)
    taskFive(inputStrNewReview, priorProbability,probabilityOfLikelihood, wordList )
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
    testDataList = testDataF['Review'].tolist()
    
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
    
    #Created dic
    countWords = dict()
    
    #Removing all the non-alphanumeric char
    for word in trinDataList:
        word = word.lower()
        word.split()
        s = ""
        s = word
        s = re.sub("[^a-zA-Z0-9!]", ' ', word)
        words = s.split()

    print('Words as a List', words)

    #Checking all the words based on the condition minimum word length
    for word in words:
        if len(word) > minWordLen:
            if word in countWords:

                countWords[word] += 1

            else:
                countWords[word] = 1

    print('Total Words Count After Removing Non-Alphanumeric Characters : ',countWords)
    
    #Checking minimum word occurance condition
    for key,val in list(countWords.items()):
        if val < minWordOcc:
         del countWords[key]
         
    #Converting dic to list  
    wordList = list(countWords.keys())
    print('Word List: ', wordList )
    
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
        
    print(positiveReviewOccurences)
    
    print('\n')
    
    #Checking and comparing wordList wtih negative review training data
    for negWord in wordL:
        counts = 0
        for j in negSeries:
            if(' ' + negWord + ' ' ) in (' ' + j + ' '):
                counts = counts + 1
        negativeReviewOccurences[negWord] = counts # Adding word to the dic
        
    print(negativeReviewOccurences)
    
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
    global priorProbability
    global probabilityOfLikelihood
    
    priorProbabilityPositive = (posReviewCount) / (posReviewCount + negReviewCount)   
    #probPos = (posReviewCount + negReviewCount) / (posReviewCount)#my try
    
    priorProbabilityNegative = (negReviewCount) / (posReviewCount + negReviewCount)
    #probNeg = (posReviewCount + negReviewCount) / (negReviewCount)

    #may be i do not need htis
    priorProbability = priorProbabilityPositive + priorProbabilityNegative
    
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

    print("\n ======================PRIOR FOR POSITIVE =============================\n")
    print(priorProbabilityPositive)
    
    print("\n\t\t********************** PROBABILITY OF EACH WORDS FOR POSITIVE*********************\n")
    print(positiveLikeliHood)
    
       
    
    print("\n\t\t ==================PRIOR FOR NEGATIVE =============================\n")
    print(priorProbabilityNegative)

    print("\n\t\t********************** PROBABILITY OF EACH WORDS FOR NEGATIVE***********************\n")
    print(negativeLikeliHood)
    
    #may be i do not need htis
    probabilityOfLikelihood = {**positiveLikeliHood ,**negativeLikeliHood}
    
    return positiveLikeliHood, negativeLikeliHood, priorProbabilityPositive, priorProbabilityNegative

#training = trainDataList[:10000]
#trainLabelList = [0, 1]

# =============================================================================
# def taskFive(training, trainLabelList):
#     D_c = [[]] * len(trainLabelList)
#     
#     n_c = [None] * len(trainLabelList)
#     
#     logprior = [None] * len(trainLabelList)
#     
#     logLikeliHood = [None] * len(trainLabelList)
#     
#     for obs in training:
#         if obs[1] >= 90:
#             D_c[1] = D_c[1] + [obs]
#         elif obs[1] < 90:
#             D_c[0] = D_c[1] + [obs]
#         
#     V = []
#     for obs in training:
#         for word in obs[0]:
#             if word in V:
#                 continue
#             else:
#                 V.append(word)
#     
#     V_size = len(V)
#     
#     #n_docs: total number of documents in training set
#     n_docs = len(training)
# 
#     for ci in range(len(trainLabelList)):
#         #Store n_c value for each class
#         n_c[ci] = len(D_c[ci])
#         
#         #Compute P(c)
#         logprior[ci] = np.log((n_c[ci] + 1)/ n_docs)
# 
# 
#         #Counts total number of words in class c
#         count_w_in_V = 0
#         for d in D_c[ci]:
#             count_w_in_V = count_w_in_V + len(d[0])
#         denom = count_w_in_V + V_size
# 
#         dic = {}
#         #Compute P(w|c)
#         for wi in V:
#             #Count number of times wi appears in D_c[ci]
#             count_wi_in_D_c = 0
#             for d in D_c[ci]:
#                 for word in d[0]:
#                     if word == wi:
#                         count_wi_in_D_c = count_wi_in_D_c + 1
#             numer = count_wi_in_D_c + 1
#             dic[wi] = np.log((numer) / (denom))
#         logLikeliHood[ci] = dic
#         
#     return (V, logprior, logLikeliHood)
# =============================================================================

def taskFive(inputStr, logprior, loglikelihood, wordL):
    #Initialize logpost[ci]: stores the posterior probability for class ci
    logpost = [None] * len(trainLabelList)
    
    for ci in trainLabelList:
        sumloglikelihoods = 0
        for word in inputStr:
            if word in wordL:
                #This is sum represents log(P(w|c)) = log(P(w1|c)) + log(P(wn|c))
                sumloglikelihoods += loglikelihood[ci][word]
        
    #Computes P(c|d)
        logpost[ci] = logprior[ci] + sumloglikelihoods

    #Return the class that generated max ĉ
    return logpost.index(max(logpost)) 
               
'''
    : Calling main
'''               
main()