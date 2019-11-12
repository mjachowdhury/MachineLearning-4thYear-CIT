# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:44:52 2019

@author: Mohammed
"""
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras.utils import to_categorical
#from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
from PIL import Image
import matplotlib.cm as cm
from sklearn import datasets
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn import model_selection
from numpy import array
from sklearn.model_selection import KFold
import time

#loading the file and read the file
data = pd.read_csv("product_images.csv")


#data.head()
#print the data
#print(data)

#seperating the lavels
sneakers = data[data.label == 0] #0 for sneakers level
#sneakersI = np.asarray(sneakers)
plt.imshow(sneakers.values[0][1:].reshape(28,28))
#print(sneakers)

totalSneakers = len(sneakers)
print('Total Sneakers : ',totalSneakers)

print('Sneakers size: ',sneakers.size)
#print('Sneakers data type: ',sneakers.dtype)
print('Sneakers shape: ',sneakers.shape)
'''
#Display Sneakers Image
i = 5
sneakerImage = sneakers.values[i][1:]
sneakerImageLength = int(math.sqrt(sneakerImage.shape[0]))
sneaker2DImage = np.int8(np.reshape(sneakerImage, (sneakerImageLength, sneakerImageLength)))
ImageSneaker = Image.fromarray(sneaker2DImage, 'L')
#Showing the image
#plt.imshow(ImageSneaker)
#plt.show()
'''
#plt.imshow(sneakers.loc[2:2], cmap=plt.cm.gray, interpolation='bilinear')
#plt.imshow(sneakers.loc[2:2], cmap=plt.cm.gray, interpolation='nearest')

#Seperating the levels
ankleBoots = data[data.label == 1] # 1 for ankleBoots
totalAnkleBoots = len(ankleBoots)
print('Total AnkleBoots : ', totalAnkleBoots)
plt.imshow(ankleBoots.values[1][1:].reshape(28,28))

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
print()
print('Maximum Time For Train :', np.max(trainingTime))
print('Minimum Time For Train :', np.min(trainingTime))
print('Average Time For Train :', np.mean(trainingTime))
print()    
print('Maximum Time For Test :', np.max(testTime))
print('Minimum Time For Test :', np.min(testTime))
print('Average Time For Test :', np.mean(testTime))
    


'''
#Displaying AnkleBoots
i = 5
ankleImage = ankleBoots.values[i][1:]
ankleImageLength = int(math.sqrt(ankleImage.shape[0]))
ankle2DImage = np.int8(np.reshape(ankleImage, (ankleImageLength, ankleImageLength)))
ImageAnkle = Image.fromarray(ankle2DImage, 'L')

#plt.imshow(ImageAnkle)
#plt.show()
'''



#train_df, test_df = train_test_split(data, shuffle=False)
#print('Printing K-Fold: ', train_df.head())

'''
#added some parameters
kf = KFold(n_splits = 2, shuffle = True, random_state = 2)
result = next(kf.split(data), None)
print (result)
#(array([0, 2, 3, 5, 6, 7, 8, 9]), array([1, 4]))

#train = data.iloc[result[0]]
#test =  data.iloc[result[1]]

#print('Train', train)
#print('Test', test)
'''

'''
data = data.values#converting pandas to numpy
#print('After converting to numpy', data)

#kfold = KFold(2, True, 1)
kf = model_selection.KFold(n_splits=2, shuffle=True)
for train, test in kf.split(data):
    print('train: %s, \n test: %s' % (train, test))
    clf1 = linear_model.Perceptron()
    clf2 = svm.SVC(kernel="rbf", gamma=1e-3)    
    clf3 = svm.SVC(kernel="sigmoid", gamma=1e-4)
    
    clf1.fit(data[train], data[train])
    prediction1 = clf1.predict(data[test])
    
    clf2.fit(data[train], data[train])
    prediction2 = clf2.predict(data[test])
    
    clf3.fit(data[train], data.target[train])
    prediction3 = clf3.predict(data[test])
    
    score1 = metrics.accuracy_score(data[test], prediction1)
    score2 = metrics.accuracy_score(data[test], prediction2)
    score3 = metrics.accuracy_score(data[test], prediction3)
    
    print("Perceptron accuracy score: ", score1)
    print("SVM with RBF kernel accuracy score: ", score2)
    print("SVM with Sigmoid kernel accuracy score: ", score3)
        
    print()
'''




'''
#changing to numpy array
sneakersI = np.asarray(sneakers)
print('Numpy array sneaker image: ', sneakersI)
lum_img = sneakersI[:]
plt.imshow(lum_img)
'''
'''
plt.imshow(sneakersI[0,1], cmap="gray")
plt.show()
'''
'''
imgplot = plt.imshow(sneakersI)
print(imgplot)
'''
'''
X = np.random.random((100, 100))
plt.imshow(X, cmap="gray")
plt.show()
'''
'''
from scipy import misc
f = misc.face()
misc.imsave('face.png', f)
#import matplotlip.pyplot as plt
plt.imshow(f)
plt.show()
'''
'''
import cv2
imageName = sneakers
img = cv2.imread(imageName, 0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])

height, width = img.shape[:2]
print('Image - ',img)
print('Image -', [10, 10])
for y in range(height):
    for x in range(width):
        print('Image - :', img[y , x], end = "\t")
    print("\t")
'''    
#SI= sneakers.iloc[0,1:2]
#print('Image', SI)
'''
sneakerImage = []
for i in tqdm(range(sneakers.shape[0])):
    img = image.laod_img('sneakers/' + sneakers['d'][i].astype['str']+ '.png', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    sneakerImage.append(img)
X = np.array(sneakerImage)
print('Image', X)    
'''
'''
counter = dict()
for row in sneakers:
    pixels = row[:-1]
    pixels = np.array(pixels, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    image = Image.fromarray(pixels)    
    label = row[-1]
    if label not in counter:
        counter[label] = 0
    counter[label] += 1
    
    filename = '{}{}.jpg'.format(label, counter[label])
    image.save(filename)

    print('saved:', filename)    
'''
''' 
array = np.array(sneakers, dtype=np.uint8)
new_image = Image.fromarray(array)
new_image.save('nn.png') 
'''
'''
sneakerImage = sneakers.iloc[0, :].values
plt.imshow(np.squeeze(sneakerImage))
plt.show()
'''
#x = sneakers.iloc[0, :].values
#plt.imshow(x.reshape(28,28))
#plt.show()
#print(x)





#printing the lavels

#print(ankleBoots)

#plt.imshow(sneakers[0,:].reshape(28,28))
#plt.show()

#plt.imsave('sneaker.png', np.array(sneakers).reshape(1280,960),cmap=cm.gray)
#plt.imshow(np.array(sneakers).reshape(1280, 960))
'''
mnist = input_data.read_data_sets('mnist')

testImage = (np.array(mnist.test.sneakers[0], dtype='float')).reshape(28,28)

img = pil.fromarray(np.uint8(testImage * 255) , 'L')
img.show()
'''
'''
mnist = input_data.read_data_sets('mnist')
sneakerImage = mnist.test.images[0]
sneakerImage = np.array(sneakerImage, dtype = 'float')
pixels = sneakerImage.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
'''


'''
import numpy as np
import matplotlib.pyplot as plt
 
# Generate random array
width = int(input('Enter width: '))
height = int(input('Enter height: '))
iMat = np.random.rand(width*height).reshape((width,height))
# Show it!
plt.imshow(iMat,'gray')
plt.show() 
'''