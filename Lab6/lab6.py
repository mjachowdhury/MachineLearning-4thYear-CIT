# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:11:41 2019

@author: Mohammed
"""

from sklearn import datasets
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn import model_selection
import matplotlib.pyplot as plt

def main():
    digits = datasets.load_digits()
    print(digits.DESCR)
    print()
    
    plt.imshow(digits.data[0,:].reshape(8,8))

    kf = model_selection.KFold(n_splits=2, shuffle=True)

    for train_index,test_index in kf.split(digits.data):          
        clf1 = linear_model.Perceptron()
        clf2 = svm.SVC(kernel="rbf", gamma=1e-3)    
        clf3 = svm.SVC(kernel="sigmoid", gamma=1e-4)    
    
        clf1.fit(digits.data[train_index], digits.target[train_index ])
        prediction1 = clf1.predict(digits.data[test_index])
    
        clf2.fit(digits.data[train_index], digits.target[train_index])
        prediction2 = clf2.predict(digits.data[test_index])
    
        clf3.fit(digits.data[train_index], digits.target[train_index])
        prediction3 = clf3.predict(digits.data[test_index])
        
        score1 = metrics.accuracy_score(digits.target[test_index], prediction1)
        score2 = metrics.accuracy_score(digits.target[test_index], prediction2)
        score3 = metrics.accuracy_score(digits.target[test_index], prediction3)
        
        print("Perceptron accuracy score: ", score1)
        print("SVM with RBF kernel accuracy score: ", score2)
        print("SVM with Sigmoid kernel accuracy score: ", score3)
        
        print()

main()
