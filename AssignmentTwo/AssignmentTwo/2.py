#Parameterize the data
data = data.head(1000)
#Display the values and lables
print('Image Values: ', data.values)
print('Image label : ', data.label)

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
    print('Confusion Matrics: ', confusion)    
    print()

print('Maximum Time For Prediction Accuracy :', np.max(predictionAccuracy))
print('Minimum Time For Prediction Accuracy :', np.min(predictionAccuracy))
print('Average Time For Prediction Accuracy :', np.mean(predictionAccuracy))

print('Maximum Time For Train :', np.max(trainingTime))
print('Minimum Time For Train :', np.min(trainingTime))
print('Average Time For Train :', np.mean(trainingTime))
    
print('Maximum Time For Test :', np.max(testTime))
print('Minimum Time For Test :', np.min(testTime))
print('Average Time For Test :', np.mean(testTime))
    