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

minWordLength = 2
minWordOccurences = 2

def main():
    task1()
    task2(trainData,minWordLength,minWordOccurences)
    task3(positiveReviewDF,negativeReviewDF, wordList)
    #task4(positiveReviewOccurences, negativeReviewOccurences, trainPosReviewCount, trainNegReviewCount)

def task1():
    global trainData
    global testData
    global trainLabel
    global testLabel
    global positiveReviewDF
    global negativeReviewDF
    global trainingRevs

    df = pd.read_excel('movie_reviews.xlsx')

    trainingDF = df[['Review', 'Sentiment', 'Split']][df['Split'] == 'train']
    evaluationDF = df[['Review', 'Sentiment', 'Split']][df['Split'] == 'test']

    trainData = trainingDF['Review'].tolist()
    testData = evaluationDF['Review'].tolist()

    trainLabel = trainingDF['Sentiment'].tolist()
    testLabel = evaluationDF['Sentiment'].tolist()

    positiveReviewDF = trainingDF[trainingDF['Sentiment'] == 'positive']
    negativeReviewDF = trainingDF[trainingDF['Sentiment'] == 'negative']

    testpositiveReviewDF = evaluationDF[evaluationDF['Sentiment'] == 'positive']
    testnegativeReviewDF = evaluationDF[evaluationDF['Sentiment'] == 'negative']

    trainPosReviewCount = positiveReviewDF.shape[0]
    trainNegReviewCount = negativeReviewDF.shape[0]

    testPosReviewCount = testpositiveReviewDF.shape[0]
    testNegReviewCount = testnegativeReviewDF.shape[0]

    print("Number of positive reviews in the training set: " , trainPosReviewCount)
    print("Number of negative reviews in the training set: " , trainNegReviewCount)
    print("Number of positive reviews in the test set: ", testPosReviewCount)
    print("Number of negative reviews in the test set: ", testNegReviewCount)

    return trainData, testData, trainLabel, testLabel


def task2(tdf, minW, minO):
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


def task3(posReview, negReview, wordL):
    
    countPosReview = dict()
    countNegReview = dict()
    
    for word in posReview:
        if word in wordL:
            if word in countPosReview:
                countPosReview[word] +=1
            else:
                countPosReview[word] =1
    print('Pos review word appear: ', countPosReview)
    
    for word in negReview:
        if word in wordL:  
            if word in countNegReview:
                countNegReview[word] +=1     
            else:
                countNegReview[word] =1
    print('Neg review word appear: ', countNegReview)            
                
                
main()