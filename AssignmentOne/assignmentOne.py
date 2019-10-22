# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:18:20 2019

@author: Mohammed
"""

import pandas as pd
import numpy as np

minWordLength = 2
minWordOccurences = 2

#movies = pd.read_excel("movie_reviews.xlsx")
#df = pd.DataFrame(movies, columns = ['Review','Sentiment','Split'])
#df = pd.DataFrame(movies, columns = ['Review','Sentiment','Split'])

#trainingDF = df[df['Split'] == "train"]
#evaluationDF = df[df['Split'] == "test"]

def task1():
    movies = pd.read_excel("movie_reviews.xlsx")
    df = pd.DataFrame(movies, columns = ['Review', 'Sentiment', 'Split'])

    trainingDF = df[df['Split'] == "train"]
    trainData = trainingDF['Review']
    trainDataList = list(trainData)
   
    trainLabels = df[df['Split'] == "train"]
    trainLabels = trainLabels['Sentiment']
    trainLabelsList = list(trainLabels)
    
    
    testDF = df[df['Split'] == "test"]
    testData = testDF['Review']
    testDataList = list(testData)
    
    testLabels = df[df['Split'] == "train"]
    testLabels = testLabels['Sentiment']
    testLabelsList = list(testLabels)
    
    positiveDFTrainSet = df[df['Split'] == "train"]
    positiveReviewTrainSet = positiveDFTrainSet['Sentiment'] == "positive"
    valueCountsPositiveTrainSet = positiveReviewTrainSet.value_counts()
    
    negativeDFTrainSet = df[df['Split'] == "train"]
    negativeReviewTrainSet = negativeDFTrainSet['Sentiment'] == "negative"
    valueCountsNegativeTrainSet = negativeReviewTrainSet.value_counts()
    
    
    positiveDFTestSet = df[df['Split'] == "test"]
    positiveReviewTestSet = positiveDFTestSet['Sentiment'] == "positive"
    valueCountsPositiveTestSet = positiveReviewTestSet.value_counts()
    
    
    negativeDFTestSet = df[df['Split'] == "test"]
    negativeReviewTestSet = negativeDFTestSet['Sentiment'] == "negative"
    valueCountsNegativeTestSet = negativeReviewTestSet.value_counts()
    
    
    #trainPosReviewCount = trainLabes[trainLabels['Sentiment']=="Positive"]
#     trainNegReviewCount = (trainLabels.value_counts() ) - (trainPosReviewCount)
# 
#     testPosReviewCount = testLabels[testLabels['Sentiment']=='Positive'].value_counts()
#     testNegReviewCount = (testLabels.value_counts() ) - (testPosReviewCount)
# =============================================================================

# =============================================================================
    print("Number of positive reviews in the training set: " , valueCountsPositiveTrainSet)
    print("Number of negative reviews in the training set: " , valueCountsNegativeTrainSet)
    
    print("Number of positive reviews in the training set: " , valueCountsPositiveTestSet)
    print("Number of negative reviews in the training set: " , valueCountsNegativeTestSet)
    
    #print("Number of positive reviews in the training set: " , positiveReviewTrainSet)
    #print("Number of negative reviews in the training set: " , negativeReviewTrainSet)
    #print("Number of positive reviews in the test set: ", positiveReviewTestSet)
    #print("Number of negative reviews in the test set: ", negativeReviewTestSet)
# =============================================================================

    #print(testDataList)
    #print(trainDataList)
    #print(trainLabelsList)
    #print(list(testLabels.value_counts()))
    #print(movies.head()) # will print first five rows
    #print(movies.describe())#will print details of the file

    return trainDataList, testDataList, trainLabelsList, testLabelsList

def task2():
    taskOne = task1()
    test =taskOne[0].str.replace('[^a-zA-Z0-9 ]', ' ').str.lower().str.split(expand=True).stack().value_counts()
    print(test)
# =============================================================================
#     counts = dict()
# 
#     for word in taskOne.replace('&', '').replace('%', '').replace('@', '').replace('!', '').replace('.', '').replace('..', '').replace('...', '').replace(',', '').replace('-', '').split():
#         word = word.lower()
# 
#         if len(word) > minWordLength:
#             if word in counts:
# 
#                 counts[word] += 1
# 
#             else:
#                 counts[word] = 1
#         else:
#             continue
# 
#     for x in counts:
#         if counts[x].values() < minWordOccurences:
#             del counts[x]
#         else:
#             continue
# 
# 
#     wordList = list(counts.keys())
# 
#     return wordList
# =============================================================================

def task3(df, listofwords ):
    positiveReviewOccurences = dict()

    for word in listofwords:
        for row in df:
            if word in row:
                positiveReviewOccurences[word]+=1
            else:
                continue

def main():   
    task1()
    #task2()
    #task3()
main()