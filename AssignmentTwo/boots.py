# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:19:05 2019

@author: Mohammed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
import time
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
 


#loading the file and read the file
data = pd.read_csv("product_images.csv")


def main():
    print('\n=====TASK ONE=====')
    task_one(data)
    print('\n=====TASK TWO=====')
    task_two(data)
    print('\n=====TASK THREE=====')
    task_three(data)
    

def task_one(data):
    #seperating the lavels
    sneakers = data[data.label == 0] #0 for sneakers level
    #Display Sneaker Image
    plt.imshow(sneakers.values[0][1:].reshape(28,28))
    plt.show()
    #print(sneakers)
    totalSneakers = len(sneakers)
    print('Total Sneakers : ',totalSneakers)
    print('Sneakers size: ',sneakers.size)
    print('Sneakers shape: ',sneakers.shape)
    
    #seperating the lavels
    ankleBoots = data[data.label == 1] #1 for ankle boots level
    #Display Ankle Boots Image
    plt.imshow(ankleBoots.values[1][1:].reshape(28,28))
    plt.show()
    #print(ankleBoots)
    totalAnkleBoots = len(ankleBoots)
    print('Total Ankle_Boots : ',totalAnkleBoots)
    print('Ankle_Boots size: ',ankleBoots.size)
    print('Ankle_Boots shape: ',ankleBoots.shape)

def task_two(data):
    #parameterize the data
    data = data.head(1000)
    #Display the values and lables
    #print('Image Values: ', data.values)
    #print('Image label : ', data.label)
    
    #List
    trainingTime = []
    predictionTime = []
    predictionAccuracy = []

    #K-Fold cross validation
    kf = model_selection.KFold(n_splits=2, shuffle=True)
    for train_index,test_index in kf.split(data.values):
        
        #liner perception
        clf1 = linear_model.Perceptron()
        #clf2 = svm.SVC(kernel="rbf", gamma=1e-3)    
        #clf3 = svm.SVC(kernel="sigmoid", gamma=1e-4)
        
        #Train Start time
        trainStartTime = time.time()
        print('\nTrain Start Time was %g on seconds :'%trainStartTime )
        
        clf1.fit(data.values[train_index], data.label[train_index ])
        
        #eTrain end time
        trainEndTime = time.time()
        print('Train End Time was %g on seconds'%trainEndTime )
        print('Total Proess Elapsed time was %g on seconds '% (trainEndTime - trainStartTime ))
        #Adding train time to the list
        trainingTime.append(trainEndTime - trainStartTime )
        print('Total Elapsed Train Processing Time: ', trainingTime)
        
        #Prediction start time
        predictionStartTime = time.time()
        print('\nPrediction Start Time was %g on seconds : '%predictionStartTime )
        
        prediction1 = clf1.predict(data.values[test_index])
        #Prediction end time
        predictionEndTime = time.time()
        print('Prediction End Time was %g on seconds  : '%predictionEndTime )
        print('Total Prediction Proess Time was %g seconds  : '%(predictionEndTime - predictionStartTime ))
        #Adding prediction time to the list
        predictionTime.append(predictionEndTime - predictionStartTime )
        print('Elapsed Prediction Processing Time : ', predictionTime)
        
        #Print out the prediction label
        print('\nPrediction level: ', prediction1)
        '''
        clf2.fit(data.values[train_index], data.label[train_index])
        prediction2 = clf2.predict(data.values[test_index])
        
        clf3.fit(data.values[train_index], data.label[train_index])
        prediction3 = clf3.predict(data.values[test_index])
        '''
        #Finding the accuracy
        score1 = metrics.accuracy_score(data.label[test_index], prediction1)
        #score2 = metrics.accuracy_score(data.label[test_index], prediction2)
        #score3 = metrics.accuracy_score(data.label[test_index], prediction3)
        
        #Adding score to the list and print the score   
        predictionAccuracy.append(score1)
        print("\nPerceptron accuracy score: ", score1)
        
        #print("SVM with RBF kernel accuracy score: ", score2)
        #print("SVM with Sigmoid kernel accuracy score: ", score3)
        
        #Calculating the confusion matrix
        confusion = metrics.confusion_matrix(data.label[test_index], prediction1)
        print('\nConfusion Matrics: \n', confusion)    
        print()
    
    #Calculating mx, min and average
    print('\nMaximum Prediction Accuracy :', np.max(predictionAccuracy))
    print('Minimum Prediction Accuracy :', np.min(predictionAccuracy))
    print('Average Prediction Accuracy :', np.mean(predictionAccuracy))
    print()
    print('Maximum Time For Train was %g seconds:'%np.max(trainingTime))
    print('Minimum Time For Train was %g seconds:'%np.min(trainingTime))
    print('Average Time For Train was %g seconds:'%np.mean(trainingTime))
    print()    
    print('Maximum Time For Test was %g seconds:'%np.max(predictionTime))
    print('Minimum Time For Test was %g seconds:'%np.min(predictionTime))
    print('Average Time For Test was %g seconds:'%np.mean(predictionTime))
        

'''
    #Parameterize the data
    data = data.head(1000)
    #Display the values and lables
    #print('Image Values: ', data.values)
    #print('Image label : ', data.label)
    
    trainingTime = []
    testTime = []
    predictionAccuracy = []
    
    #K-Fold cross validation
    kf = model_selection.KFold(n_splits=2, shuffle=True)
    for train_index,test_index in kf.split(data.values):
        
        clf1 = linear_model.Perceptron()
        clf2 = svm.SVC(kernel="rbf", gamma=1e-3)    
        clf3 = svm.SVC(kernel="sigmoid", gamma=1e-4)
        
        trainStartTime = time.time()
        print('Train Start Time: ',trainStartTime )
        
        clf1.fit(data.values[train_index], data.label[train_index ])
        prediction1 = clf1.predict(data.values[test_index])
        #print('Prediction level: ', prediction1)
        
        clf2.fit(data.values[train_index], data.label[train_index])
        prediction2 = clf2.predict(data.values[test_index])
        
        clf3.fit(data.values[train_index], data.label[train_index])
        prediction3 = clf3.predict(data.values[test_index])
        
        trainEndTime = time.time()
        print('End Time : ',trainEndTime )
        print('Total Proess Time : ', (trainEndTime - trainStartTime ))
        trainingTime.append(trainEndTime - trainStartTime )
        print('Training Time: ', trainingTime)
        
        testStartTime = time.time()
        print('Test Start Time: ',testStartTime )
        
        score1 = metrics.accuracy_score(data.label[test_index], prediction1)
        score2 = metrics.accuracy_score(data.label[test_index], prediction2)
        score3 = metrics.accuracy_score(data.label[test_index], prediction3)
        
        testEndTime = time.time()
        print('End Time : ',testEndTime )
        print('Total Proess Time : ', (testEndTime - testStartTime ))
        testTime.append(testEndTime - testStartTime )
        print('Prediction Time: ', testTime)
        
        predictionAccuracy.append(score1)
        print("Perceptron accuracy score: ", score1)
        print("SVM with RBF kernel accuracy score: ", score2)
        print("SVM with Sigmoid kernel accuracy score: ", score3)
        
        confusion = metrics.confusion_matrix(data.label[test_index], prediction1)
        print('Confusion Matrics: \n', confusion)    
        print()
    
    print('Maximum Time For Prediction Accuracy :', np.max(predictionAccuracy))
    print('Minimum Time For Prediction Accuracy :', np.min(predictionAccuracy))
    print('Average Time For Prediction Accuracy :', np.mean(predictionAccuracy))
    print()
    print('Maximum Time For Train :', np.max(trainingTime))
    print('Minimum Time For Train :', np.min(trainingTime))
    print('Average Time For Train :', np.mean(trainingTime))
    print()    
    print('Maximum Time For Test :', np.max(testTime))
    print('Minimum Time For Test :', np.min(testTime))
    print('Average Time For Test :', np.mean(testTime))
'''        

def task_three(data):
    #Parameterize the data
    data = data.head(1000)
    #Display the values and lables
    #print('Image Values: ', data.values)
    #print('Image label : ', data.label)
    
    #List to store the training and prediction process time
    trainingTime = []
    predictionTime = []
    predictionAccuracy = []
    
    #K-Fold cross validation spliting the data
    kf = model_selection.KFold(n_splits=2, shuffle=True)
    for train_index,test_index in kf.split(data.values):
        
        #clf1 = linear_model.Perceptron()
        #support vector machine 
        clf2 = svm.SVC(kernel="rbf", gamma=1e-3)#gama different choices
        #clf3 = svm.SVC(kernel="rbf", gamma=1e-4)
        
        clf4 = svm.SVC(kernel="linear", gamma=1e-4)#linear kernel
        
        #Train Start time
        trainStartTime = time.time()
        print('Train Start Time: ',trainStartTime )
        
        #clf1.fit(data.values[train_index], data.label[train_index ])
        #prediction1 = clf1.predict(data.values[test_index])
        #print('Prediction level: ', prediction1)
        
        clf2.fit(data.values[train_index], data.label[train_index])#support vector machine
        #Training end time
        trainEndTime = time.time()
        print('End Time : ',trainEndTime )
        print('Total Proess Time : ', (trainEndTime - trainStartTime ))
        #Adding train time to the list
        trainingTime.append(trainEndTime - trainStartTime )
        print('Training Time: ', trainingTime)
        
        #Prediction start time 
        predictionStartTime = time.time()
        print('Prediction Start Time: ',predictionStartTime )
        
        prediction2 = clf2.predict(data.values[test_index])
        print('Predict Label: ', prediction2)
        #Prediction time end
        predictionEndTime = time.time()
        print('Prediction End Time : ',predictionEndTime )
        print('Total Proess Time : ', (predictionEndTime - predictionStartTime ))
        #Adding prediction time to the list
        predictionTime.append(predictionEndTime - predictionStartTime )
        print('Prediction Time: ', predictionTime)
        
        #clf3.fit(data.values[train_index], data.label[train_index])#support vector machine
        #prediction3 = clf3.predict(data.values[test_index])
        
        #Classifier and prediction for liner
        clf4.fit(data.values[train_index], data.label[train_index])#linear kernel
        prediction4 = clf4.predict(data.values[test_index])
        
       
        
        
        
       # score1 = metrics.accuracy_score(data.label[test_index], prediction1)
       #accuracy score for rbf
        score2 = metrics.accuracy_score(data.label[test_index], prediction2)
        
        #score3 = metrics.accuracy_score(data.label[test_index], prediction3)
        #Accuracy score for liner
        score4 = metrics.accuracy_score(data.label[test_index], prediction4)
        
        
        #Adding accuracy score to the list
        predictionAccuracy.append(score2)
        #print("Perceptron accuracy score: ", score2)
        
        print("SVM with RBF linear accuracy score: ", score2)
        
        print("SVM with Sigmoid liner accuracy score: ", score4)
        #Getting confusion matrix
        confusion = metrics.confusion_matrix(data.label[test_index], prediction2)
        print('\nConfusion Matrics: \n', confusion)    
        print()
    #Calculating mx, min and average
    print('\nMaximum Prediction Accuracy :', np.max(predictionAccuracy))
    print('Minimum Prediction Accuracy :', np.min(predictionAccuracy))
    print('Average Prediction Accuracy :', np.mean(predictionAccuracy))
    print()
    print('Maximum Time For Train was %g seconds:'%np.max(trainingTime))
    print('Minimum Time For Train was %g seconds:'%np.min(trainingTime))
    print('Average Time For Train was %g seconds:'%np.mean(trainingTime))
    print()    
    print('Maximum Time For Test was %g seconds:'%np.max(predictionTime))
    print('Minimum Time For Test was %g seconds:'%np.min(predictionTime))
    print('Average Time For Test was %g seconds:'%np.mean(predictionTime))
        
    
main()
