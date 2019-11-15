# -*- coding: utf-8 -*-
 
"""
Created on Mon Nov 11 16:19:05 2019

@author: Mohammed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from decimal import Decimal
import time


#loading the file and read the file
data = pd.read_csv("product_images.csv")


def main():
    numberOfSamples = int(input('ENTER NUMBER OF SAMPLE : '))
    numberOfSplits = int(input('ENTER NUMBER OF SPLITS: '))
    
    print('\n========= TASK ONE =========')
    task_one(data)
    
    print('\n========= TASK TWO =========')
    task_two(data, numberOfSamples, numberOfSplits)
    
    print('\n========= TASK THREE =========')
    task_three(data, numberOfSamples, numberOfSplits)
    

def task_one(data):
    sneakers = data[data.label == 0] #0 for sneakers level
    plt.imshow(sneakers.values[0][1:].reshape(28,28))
    plt.show()
    #print(sneakers)
    totalSneakers = len(sneakers)
    print('\nTOTAL SNEAKERS : ',totalSneakers)
    print('SNEAKER SIZE: ',sneakers.size)
    print('SNEAKER SHAPE: ',sneakers.shape)
    
    
    ankleBoots = data[data.label == 1] #1 for ankle boots level     
    plt.imshow(ankleBoots.values[1][1:].reshape(28,28))
    plt.show()
    #print(ankleBoots)
    totalAnkleBoots = len(ankleBoots)
    print('\nTOTAL ANKLE BOOTS: ',totalAnkleBoots)
    print('ANKLE BOOT SIZE: ',ankleBoots.size)
    print('ANKLE BOOT SHAPE: ',ankleBoots.shape)

def task_two(data, numberOfSamples, numberOfSplits):
    
    data = data.head(numberOfSamples)
    #Display the values and lables
    #print('Image Values: ', data.values)
    #print('Image label : ', data.label)
    
    trainingTime = []
    predictionTime = []
    predictionAccuracy = []

    #K-Fold cross validation
    kf = model_selection.KFold(n_splits=numberOfSplits, shuffle=True)
    for train_index,test_index in kf.split(data.values):
        
        #liner perception
        clf1 = linear_model.Perceptron()
                
        trainStartTime = time.time()
        print('\nTrain Start Time was on %g  seconds :' %trainStartTime )   
        clf1.fit(data.values[train_index], data.label[train_index ])
        trainEndTime = time.time()
        print('Train End Time was on %g seconds'%trainEndTime )
        print('Total Proess Elapsed time was on %g seconds '% (trainEndTime - trainStartTime ))
        trainingTime.append(trainEndTime - trainStartTime )
        
        predictionStartTime = time.time()
        print('\nPrediction Start Time was on %g seconds : '%predictionStartTime )        
        prediction1 = clf1.predict(data.values[test_index])
        predictionEndTime = time.time()
        print('Prediction End Time was on %g seconds  : '%predictionEndTime )
        print('Total Prediction Proess Time was on %g seconds  : '%(predictionEndTime - predictionStartTime ))
        predictionTime.append(predictionEndTime - predictionStartTime )
        print('\nPREDICTION LABEL: \n', prediction1)
         
        score1 = metrics.accuracy_score(data.label[test_index], prediction1)
        predictionAccuracy.append(score1)
        print("\nPerceptron accuracy score: ", score1)
        confusion = metrics.confusion_matrix(data.label[test_index], prediction1)
        print('\nConfusion Matrics: \n', confusion)  
        
        print()
    
    #Calculating mx, min and average
    print('\n======== LINEAR PERCEPTION PREDICTION ========')
    print('\nMaximum Prediction Accuracy :', np.max(predictionAccuracy))
    print('Minimum Prediction Accuracy :', np.min(predictionAccuracy))
    print('Average Prediction Accuracy :', np.mean(predictionAccuracy))
    
    print('\n======== LINEAR PERCEPTION TRAIN PROCESS TIME ========')
    print('Maximum Time For Train was %g seconds:'%np.max(trainingTime))
    print('Minimum Time For Train was %g seconds:'%np.min(trainingTime))
    print('Average Time For Train was %g seconds:'%np.mean(trainingTime))
    
    print('\n======== LINEAR PERCEPTION PREDICTION PROCESS TIME ========')
    print('Maximum Time For Prediction was %g seconds:'%np.max(predictionTime))
    print('Minimum Time For Prediction was %g seconds:'%np.min(predictionTime))
    print('Average Time For Prediction was %g seconds:'%np.mean(predictionTime))
 
def task_three(data,numberOfSamples, numberOfSplits):
     
    data = data.head(numberOfSamples)
    #Display the values and lables
    #print('Image Values: ', data.values)
    #print('Image label : ', data.label)
    
    #List to store the training and prediction process time and prediction accuracy
    rbfTrainingTime = []
    rbfPredictionTime = []
    
    linearTrainingTime = []
    linearPredictionTime = []
    
    linearPredictionAccuracy ={ '1e-3':[], '1e-4':[],'1e-5':[],'1e-6':[],'1e-7': []  }
    rbfPredictionAccuracy = {'1e-3':[], '1e-4':[],'1e-5':[],'1e-6':[],'1e-7':[]}
      
    gamma_Value_List = [1e-3, 1e-4,1e-5,1e-6,1e-7]
    
    #K-Fold cross validation spliting the data
    kf = model_selection.KFold(n_splits=numberOfSplits, shuffle=True)
    for train_index,test_index in kf.split(data.values):
        
        for i in gamma_Value_List:           
            clf2 = svm.SVC(kernel="rbf", gamma=i) 
            clf4 = svm.SVC(kernel="linear", gamma=i) 
            
            #Classifier and prediction for rbf
            rbfTrainStartTime = time.time()
            print('\nTrain Start Time was on %g seconds: '%rbfTrainStartTime )
            clf2.fit(data.values[train_index], data.label[train_index])          
            rbfTrainEndTime = time.time()
            print('End Time was on %g seconds: '%rbfTrainEndTime )          
            rbfTrainingTime.append(rbfTrainEndTime - rbfTrainStartTime )
             
            
            rbfPredictionStartTime = time.time()
            print('\nPrediction Start Time was on %g seconds: '%rbfPredictionStartTime )
            prediction2 = clf2.predict(data.values[test_index])             
            rbfPredictionEndTime = time.time()
            print('Prediction End Time was on %g seconds: '%rbfPredictionEndTime )           
            rbfPredictionTime.append(rbfPredictionEndTime - rbfPredictionStartTime )
            print('\nRBF PREDICTION LABEL: \n', prediction2)
           
            score2 = metrics.accuracy_score(data.label[test_index], prediction2)   
            rbfPredictionAccuracy['{:.0e}'.format(Decimal(i))].append(score2)
            print("\nSVM with RBF linear accuracy score: ", score2)            
            confusionRBF = metrics.confusion_matrix(data.label[test_index], prediction2)
            print('\nRBF Confusion Matrics: \n', confusionRBF)    
            
            #Classifier and prediction for liner
            linerClassifierStartTime = time.time()           
            clf4.fit(data.values[train_index], data.label[train_index])   
            linerClassifierEndTime = time.time()
            linearTrainingTime.append(linerClassifierEndTime - linerClassifierStartTime )
            
            linearPredictionStartTime = time.time()
            prediction4 = clf4.predict(data.values[test_index])
            linearPredictionEndTime = time.time()
            linearPredictionTime.append(linearPredictionEndTime - linearPredictionStartTime)          
            print('\nLINEAR PREDICTION LABEL :', prediction4)
             
            score4 = metrics.accuracy_score(data.label[test_index], prediction4)
            linearPredictionAccuracy['{:.0e}'.format(Decimal(i))].append(score4)
            print("\nSVM with Sigmoid liner accuracy score: ", score4)           
            confusionLinear = metrics.confusion_matrix(data.label[test_index], prediction4)
            print('\nLinear Confusion Matrics: \n', confusionLinear)    
            
            print()
        
    #Calculating max, min and average for RBF
    print('\n==== RBF TRAIN PROCESS TIME ====') 
    print('\nMaximum Time For RBF Train was %g seconds:'%np.max(rbfTrainingTime))
    print('Minimum Time For RBF Train was %g seconds:'%np.min(rbfTrainingTime))
    print('Average Time For RBF Train was %g seconds:'%np.mean(rbfTrainingTime))
    
    print('\n==== RBF PREDICTION PROCESS TIME ====')    
    print('\nMaximum Time For RBF Prediction was %g seconds:'%np.max(rbfPredictionTime))
    print('Minimum Time For RBF Prediction was %g seconds:'%np.min(rbfPredictionTime))
    print('Average Time For RBF Prediction was %g seconds:'%np.mean(rbfPredictionTime))
    
    #Calculating mx, min and average for Linear
    print('\n==== LINEAR TRAIN PROCESS TIME ====')  
    print('\nMaximum Time For Linear Train was %g seconds:'%np.max(linearTrainingTime))
    print('Minimum Time For Linear Train was %g seconds:'%np.min(linearTrainingTime))
    print('Average Time For Linear Train was %g seconds:'%np.mean(linearTrainingTime))
    
    print('\n==== LINEAR TRAIN PROCESS TIME ====')    
    print('\nMaximum Time For Linear Prediction was %g seconds:'%np.max(linearPredictionTime))
    print('Minimum Time For Linear Prediction was %g seconds:'%np.min(linearPredictionTime))
    print('Average Time For Linear Prediction was %g seconds:'%np.mean(linearPredictionTime))
        
    print('\n=========== RBF PREDICTION ACCURACY ============')
    for i in gamma_Value_List:       
        print('\nRBF ==={:.0e}'.format(Decimal(i)), '=== RBF PREDICTION ACCURACY\n')
        print('Maximum RBF Prediction Accuracy:', np.max(rbfPredictionAccuracy['{:.0e}'.format(Decimal(i))]))
        print('Maximum RBF Prediction Accuracy:', np.min(rbfPredictionAccuracy['{:.0e}'.format(Decimal(i))]))
        print('Maximum RBF Prediction Accuracy:', np.mean(rbfPredictionAccuracy['{:.0e}'.format(Decimal(i))]))
    
    print('\n=========== LINEAR PREDICTION ACCURACY ============')
    for i in gamma_Value_List:       
        print('\nLINEAR === {:.0e}'.format(Decimal(i)), '=== LINEAR PREDICTION ACCURACY\n')
        print('Maximum Linear Prediction Accuracy:', np.max(linearPredictionAccuracy['{:.0e}'.format(Decimal(i))]))
        print('Maximum Linear Prediction Accuracy:', np.min(linearPredictionAccuracy['{:.0e}'.format(Decimal(i))]))
        print('Maximum Linear Prediction Accuracy:', np.mean(linearPredictionAccuracy['{:.0e}'.format(Decimal(i))]))
    
        
main()
