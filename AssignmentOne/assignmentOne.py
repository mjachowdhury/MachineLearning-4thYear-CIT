# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:18:20 2019

@author: Mohammed
"""

import pandas as pd
import numpy as np

minWordLength = 2
minWordOccurences = 2

df = pd.read_excel("movie_reviews.xlsx",skipinitialspace=True)
trainingDF = df['Review','Sentiment','Split'][df['Split'] == 'train']
evaluationDF = df['Review','Sentiment','Split'][df['Split'] == 'test']



def task1(trainingDF, evaluationDF):
    trainData = trainingDF['Review']
    testData = evaluationDF['Review']

    trainLabels = trainingDF['Sentiments']
    testLabels = evaluationDF['Sentiments']

    trainPosReviewCount = trainLabels[trainLabels['Sentiments']=='Positive'].value_counts()
    trainNegReviewCount = (trainLabels.value_counts() ) - (trainPosReviewCount)

    testPosReviewCount = testLabels[testLabels['Sentiments']=='Positive'].value_counts()
    testNegReviewCount = (testLabels.value_counts() ) - (testPosReviewCount)

    print("Number of positive reviews in the training set: " , trainPosReviewCount)
    print("Number of negative reviews in the training set: " , trainNegReviewCount)
    print("Number of positive reviews in the test set: ", testPosReviewCount)
    print("Number of negative reviews in the test set: ", testNegReviewCount)

    print(testData.value_counts())

    return trainData.to_list(), testData.to_list(), trainLabels.to_list(), testLabels.to_list()

def task2(trainData, minWordLength, minWordOccurences):
    counts = dict()

    for word in trainData.replace('&', '').replace('%', '').replace('@', '').replace('!', '').replace('.', '').replace('..', '').replace('...', '').replace(',', '').replace('-', '').split():
        word = word.lower()

        if len(word) > minWordLength:
            if word in counts:

                counts[word] += 1

            else:
                counts[word] = 1
        else:
            continue

    for x in counts:
        if counts[x].values() < minWordOccurences:
            del counts[x]
        else:
            continue


    wordList = list(counts.keys())

    return wordList

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
    task2()

main()