# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:02:23 2019

@author: Mohammed Alom
Student Number: R00144214
SDH4A
"""

import pandas as pd
import numpy as np
import re
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import svm


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
    taskFive(trainDataList[1], positiveLikeliHood,negativeLikeliHood, priorProbabilityPositive, priorProbabilityNegative)
    #taskSix()
'''
    : TaskOne Function Spliting and counting the review
'''
#data = pd.read_excel('movie_reviews.xlsx')

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
    #print(type(trainDataList))
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

    print('\nWords as a List', words)
    #print('printing single value: ', words[0])
    #Checking all the words based on the condition minimum word length
    for word in words:
        if len(word) > minWordLen:
            if word in countWords:

                countWords[word] += 1

            else:
                countWords[word] = 1
    print("\nLength of the word : " , countWords)
    #print('Total Words Count After Removing Non-Alphanumeric Characters : ',countWords)
    
    #Checking minimum word occurance condition
    #for key,val in list(countWords.items()):
     #   print("Name of the words: ", words, "number of times appear:--->", minWordOcc)
     #   if val < minWordOcc:
         #del countWords[key]
         
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
        
    print("\nPositive Frequncy occurance: ",positiveReviewOccurences)
    
    print('\n')
    
    #Checking and comparing wordList wtih negative review training data
    for negWord in wordL:
        counts = 0
        for j in negSeries:
            if(' ' + negWord + ' ' ) in (' ' + j + ' '):
                counts = counts + 1
        negativeReviewOccurences[negWord] = counts # Adding word to the dic
        
    print("\nNegative Fequency Occurance : ",negativeReviewOccurences)
    
    return positiveReviewOccurences, negativeReviewOccurences

'''
    : Task Four Finding the probability and prior
'''
def taskFour(posReviewOcc, negReviewOcc, posReviewCount, negReviewCount):
    global priorProbabilityPositive
    global priorProbabilityNegative
    global positiveLikeliHood
    global negativeLikeliHood
    
    
    priorProbabilityPositive = (posReviewCount) / (posReviewCount + negReviewCount)   
     
    priorProbabilityNegative = (negReviewCount) / (posReviewCount + negReviewCount)
     
    
    positiveLikeliHood = dict()
    negativeLikeliHood = dict()

    laplace = 1

    for key, value in posReviewOcc.items():
        mPos = posReviewCount
        x =  (value + laplace) / (priorProbabilityPositive + (mPos * laplace))
        positiveLikeliHood[key] = x

    for key, value in negReviewOcc.items():
        mNeg = negReviewCount
        y =  (value + laplace) / (priorProbabilityNegative + (mNeg * laplace))
        negativeLikeliHood[key] = y

    print("\n =======PRIOR FOR POSITIVE =======\n")
    print(priorProbabilityPositive)
    
    print("\n\t\t***** PROBABILITY OF EACH WORDS FOR POSITIVE *****\n")
    print(positiveLikeliHood)
    
       
    
    print("\n\t\t===== PRIOR FOR NEGATIVE =======\n")
    print(priorProbabilityNegative)

    print("\n\t\t***** PROBABILITY OF EACH WORDS FOR NEGATIVE *****\n")
    print(negativeLikeliHood)
    

    return positiveLikeliHood, negativeLikeliHood, priorProbabilityPositive, priorProbabilityNegative

def taskFive(inputStr,positiveLikeliHood, negativeLikeliHood, priorProbabilityPositive,priorProbabilityNegative):
    
    log_likelihoodPositive = 0
    log_likelihoodNegative = 0
    
    newList = str(inputStr).split()
    
    #print("Printing one review : ", newList)
    
    for word in newList:
        #print(word)
        for feature in positiveLikeliHood:
            #print(feature)
            if word == feature:
                print("\nPositive likelihood: ", positiveLikeliHood[feature])
                log_likelihoodPositive = log_likelihoodPositive+math.log(positiveLikeliHood[feature])
                
   
        for feature in negativeLikeliHood:
            #print(feature)
            if word == feature:
                print("\nNegative likelihood : ",negativeLikeliHood[feature])
                log_likelihoodNegative = log_likelihoodNegative+math.log(negativeLikeliHood[feature])
    
    print("Positive :" ,log_likelihoodPositive)
    print("Negative :", log_likelihoodNegative)
 
    if log_likelihoodPositive - log_likelihoodNegative > math.log(priorProbabilityPositive) - math.log(priorProbabilityNegative ):
        print("\nPositive")
    else:
        print("\nNegative")
               

def taskSix():
     
    movie = pd.read_excel('movie_reviews.xlsx')
    trainingDataF = movie[["Review","Sentiment"]][movie["Split"]=='train'] 
    trainingDataF["Sentiment"]= trainingDataF["Sentiment"].map({"positive": 1 , "negative": 0})
    target = movie["Sentiment"]
     
    allResult = []
    trainingDataF = trainingDataF.values  
      #kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
    kf = model_selection.KFold(n_splits=10, shuffle = True)
    for train_index, test_index in kf.split(trainingDataF):
        clf = svm.SVC()
        clf.fit(trainingDataF[train_index], target[train_index])
        result = clf.predict(trainingDataF[test_index])
        print(result[result !=target[test_index]])
          
        allResult.append(metrics.accuracy_score(result,target[test_index]))
    print("Accuracy is : ", np.mean(allResult))   
            
    ROC_X = []
    ROC_Y = []
       
    for k in range(1, 10):
        true_positive = []
        true_negative = []
        false_positive = []
        false_negative = []
           
        for train_index, test_index in kf.split(trainingDataF, target):
            
            clf = neighbors.KNeighborsClassifier(n_neighbors=k)
            clf.fit(trainingDataF.iloc[train_index], target[train_index])
            predicted_labels = clf.predict(trainingDataF.iloc[test_index])
               
            C = metrics.confusion_matrix(target[test_index], predicted_labels)
               
            true_positive.append(C[1,1])
            true_negative.append(C[0,0])
            false_positive.append(C[0,1])
            false_negative.append(C[1,0])
               
            print("k =",k)
            print("True positive:", np.sum(true_positive))
            print("True negative:", np.sum(true_negative))
            print("False positive:", np.sum(false_positive))
            print("False negative:", np.sum(false_negative))
            print()
           
           
            ROC_X.append(np.sum(true_positive))
            ROC_Y.append(np.sum(false_positive))
       
    print(ROC_X)
    print(ROC_Y)
       
    plt.close('all')
    plt.figure()
    plt.scatter(ROC_X,ROC_Y)
    plt.axis([0,np.max(ROC_X),0,np.max(ROC_Y)])
 

                 
'''
    : Calling main
'''    
    
           
main()