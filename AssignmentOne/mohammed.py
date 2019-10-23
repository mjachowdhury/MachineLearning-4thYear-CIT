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
    '''
        : Main Function
    '''
def main():
    taskOne()
    taskTwo(trainDataList,minimumWordLength,minimumWordOccurences)
    taskThree(positiveReviewDataF,negativeReviewDataF, wordList)
    taskFour(positiveReviewOccurences, negativeReviewOccurences, trainPositiveReviewCount, trainNegativeReviewCount)

    '''
        : TaskOne Function
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
        : Reading Data From File
    '''
    data = pd.read_excel('movie_reviews.xlsx')
    
    '''
        : Spliting Data Based on the Split (Train / Test)
    '''
    trainingDataF = data[['Review', 'Sentiment', 'Split']][data['Split'] == 'train']
    testDataF = data[['Review', 'Sentiment', 'Split']][data['Split'] == 'test']
    
    '''
        : Converting the DataFrame Review (train/test) to List
    '''
    trainDataList = trainingDataF['Review'].tolist()
    testDataList = testDataF['Review'].tolist()
    
    '''
        : Converting the DataFrame Sentiment (train/test) to List
    '''
    trainLabelList = trainingDataF['Sentiment'].tolist()
    testLabelList = testDataF['Sentiment'].tolist()
    
    '''
        : Separating Data From Training based on Sentiment - Positive/Negative
    '''
    positiveReviewDataF = trainingDataF[trainingDataF['Sentiment'] == 'positive']
    negativeReviewDataF = trainingDataF[trainingDataF['Sentiment'] == 'negative']
    
    '''
        : Separating Data From Test based on Sentiment - Positive/Negative
    '''    
    testpositiveReviewDataF = testDataF[testDataF['Sentiment'] == 'positive']
    testnegativeReviewDataF = testDataF[testDataF['Sentiment'] == 'negative']
    
    '''
        : Total Positive and Negative review from trining data
    '''
    trainPositiveReviewCount = positiveReviewDataF.shape[0]
    trainNegativeReviewCount = negativeReviewDataF.shape[0]
    
    '''
        : Total Positive and Negative review from test data
    '''
    testPositiveReviewCount = testpositiveReviewDataF.shape[0]
    testNegativeReviewCount = testnegativeReviewDataF.shape[0]
    
    '''
        : Printing From Training Data Set Positive and Negative Review Counts
    '''
    print("Totel Number of Positive Reviews in the Training Set: " , trainPositiveReviewCount)
    print("Total Number of Negative Reviews in the Training set: " , trainNegativeReviewCount)
    
    '''
        : Printing from Test Data Set Positive and Negative Review Counts
    '''
    
    print("Total Number of Positive Reviews in the Test Set: ", testPositiveReviewCount)
    print("Total Number of Negative Reviews in the Test Set: ", testNegativeReviewCount)
    
    #print('TRAINING DATA REVIEW AS LIST: ', positiveReviewDF)
    #sns.countplot(x='Sentiment', data=data) #will show the plot
    '''
        : Returning Four Lists
    '''
    return trainDataList, testDataList, trainLabelList, testLabelList


def taskTwo(tdf, minW, minO):
    counts = dict()
    #replaced = [re.sub(r"[^a-zA-Z0-9]", "," , word) for word in tdf]

    #print(replaced)

    for word in tdf:
        word = word.lower()
        word.split()
        s = ""
        s = word
        s = re.sub("[^a-zA-Z0-9!]", ' ', word)
        words = s.split()

    print('Words as a List', words)

    for word in words:
        if len(word) > minW:
            if word in counts:

                counts[word] += 1

            else:
                counts[word] = 1

    print('Words count: ',counts)

    for key,val in list(counts.items()):
        if val < minO:
         del counts[key]

    global wordList
    
    wordList = list(counts.keys())
    print('Word List: ', wordList )
    return wordList


def taskThree(posReview, negReview, wordL):
    
    global positiveReviewOccurences
    global negativeReviewOccurences
    
    posSeries = pd.Series(posReview['Review'], index=posReview.index)
    negSeries = pd.Series(negReview['Review'], index=negReview.index)
 
    positiveReviewOccurences = {}
    
    for word in wordL:
        count = 0
        for s in posSeries:
            if(' ' + word + ' ' ) in (' ' + s + ' '):
                count = count + 1
        positiveReviewOccurences[word] = count
    print(positiveReviewOccurences)
    
    print('\n')
    
    negativeReviewOccurences = {}
    
    for negWord in wordL:
        counts = 0
        for j in negSeries:
            if(' ' + negWord + ' ' ) in (' ' + j + ' '):
                counts = counts + 1
        negativeReviewOccurences[negWord] = counts
    print(negativeReviewOccurences)
    
    return positiveReviewOccurences, negativeReviewOccurences

def task4(posReviewOcc, negReviewOcc, posReviewCount, negReviewCount):
    #def task4(pro , nro , tpc , tnc ):
    probPos = (posReviewCount) / (posReviewCount + negReviewCount)  #original first one
    #probPos = (posReviewCount + negReviewCount) / (posReviewCount)#my try
    
    probNeg = (negReviewCount) / (posReviewCount + negReviewCount)
    #probNeg = (posReviewCount + negReviewCount) / (negReviewCount)


    posLikelihoodD = dict()
    negLikelihoodD = dict()

    laplace = 1

    for key, value in posReviewOcc.items():
        m = posReviewCount
        x =  (value + laplace) / (probPos + (m * laplace))
        posLikelihoodD[key] = x

    for key, value in negReviewOcc.items():
        m2 = negReviewCount
        y =  (value + laplace) / (probNeg + (m2 * laplace))
        negLikelihoodD[key] = y

    print("*****************************************************")
    print(posLikelihoodD)
    print("*****************************************************")
    print(probPos)
    
    print("*****************************************************")

    print(negLikelihoodD)
    
    print("*****************************************************")

    print(probNeg)


    return posLikelihoodD, negLikelihoodD, probPos, probNeg
     
 #=============================================================================
     #posReviewC = posReview.apply(lambda i: list(set(i))) # making sure a word occurs only once per row
     #all_words = [j for i in posReviewC.values.tolist() for j in i]
     #postive = {}
     
     #for j in wordL:
      #   postive[j] = all_words.count(j)
     #print(postive)
 #=============================================================================
     #df = pd.DataFrame(columns=['Review'], posReview)
     #word_list = ['the', 'up', 'me']

    # pos=posReview.stack().groupby(level=0)\
     #.apply(lambda x: x.drop_duplicates().value_counts())\
     #.sum(level=1)[wordL]
     #print(pos)
 #=============================================================================
# =============================================================================
#       countPosReview = []
#       for i in posReview:
#           if i not in countPosReview:
#               print('Number', i, 'is presented', wordL.count(i), 'times in wordL')
#               countPosReview.append(i)
#          # print(posReview.count(x))
# =============================================================================
 #=============================================================================
    
# =============================================================================
#     wordfreq = [posReview.count(p) for p in posReview] & [wordL.count(w) for w in wordL]
#     countPosReview = dict(zip(wordfreq))
#     print(countPosReview)
# =============================================================================
    
# =============================================================================
#     documents_words = set(wordL)
#     countPosReview = {}
#     #features = {}
#     for word in posReview:
#         countPosReview['contains({})'.format(word)] = (word in documents_words)
#         print('Pos review word appear: ', countPosReview)
#         return countPosReview
# =============================================================================
    
# =============================================================================
#     countPosReview = dict()
#     countNegReview = dict()
#     
#     res = pd.Series(' '.join(df['posReview']).lower().split()).value_counts()[:100]
#     countPosReview[res]+=1
#     for k in posReview:
#         for v in wordL:
#             if k in v:
#                 countPosReview[k]+=1
# =============================================================================
        #if word in wordL:
            #res=set(word) & set(word)
            #res=pd.Series(wordL).value_counts()
            #word=res
            #countPosReview[res] +=1
            #else:
                #continue
# =============================================================================
#     print('Pos review word appear: ', countPosReview)
#     for word in negReview:
#         #for row in negReview:
#         if word in wordL:
#             countNegReview[word] +=1
#         else:
#             continue
#     print('Neg review word appear: ', countNegReview)
# =============================================================================
# =============================================================================
#     for word in posReview:
#         if word in wordL:
#             if word in countPosReview:
#                 countPosReview[word] +=1
#             else:
#                 countPosReview[word] =1
#     print('Pos review word appear: ', countPosReview)
#     
#     for word in negReview:
#         if word in wordL:  
#             if word in countNegReview:
#                 countNegReview[word] +=1     
#             else:
#                 countNegReview[word] =1
#     print('Neg review word appear: ', countNegReview)            
# =============================================================================
                
                
main()