# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:53:51 2019

@author: Mohammed
"""

from sklearn import datasets
from sklearn import model_selection
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

def main():
    iris = datasets.load_iris()
    maxK = 50
    kf = model_selection.KFold(n_splits=len(iris.target), shuffle=True)
    correct = [0]*(maxK-1)
    for k in range(1, maxK):
        print('/r', 100*(k+1)/maxK, "%", end="")
        for train_index, test_index in kf.split(iris.data):
            clf = neighbors.KNeighborsClassifier(n_neighbors=k)
            clf.fit(iris.data[train_index], iris.target[train_index])
            prediction = clf.predict(iris.data[test_index])
            if iris.target[test_index] == prediction:
                correct[k-1] = correct[k-1] + 1
                
    plt.close("all")
    plt.figure()
    plt.plot(range(1, maxK), np.array(correct)/len(iris.target))
    
main()