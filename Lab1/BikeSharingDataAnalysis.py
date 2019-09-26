# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:33:22 2019

@author: Mohammed
"""

"""
Question 3. In this question you will perform a basic analysis of a bike sharing datasetavailable 
athttps://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset.Find the CSV file “day.csv”and create a 
program that goes through each lineand reads the weather condition indicator and number of bikerentals 
recorded.Note, that the first line does not contain a valid data record.It should then output the number 
of days in the dataset with clear weather conditions and theaverage number of bike rentalson cleardays.
(The result should be 463dayswith an average of 4877 bike rentals per day)

"""
import csv

#opening the csv file 
"""
with open("F:\MachineLearning\Bike-Sharing-Dataset\day.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        print (', '.join(row))
"""
        
def bikes():
    weatherit = 1
    numberOfRentals = 0
    numberOfDays = 0
    
    f = open("F:\MachineLearning\Lab1\day.csv", 'r')
    for line in f:
        elements = line.split(",")
        if elements[0] != "instant":
            weather = int(elements[8])
            rentals = int(elements[15])
            if weather == weatherit:
                numberOfRentals = numberOfRentals + rentals
                numberOfDays = numberOfDays + 1
    f.close()
    print("Number of Days: ", numberOfDays)
    print("Average number of rentals: ", numberOfRentals / numberOfDays)

bikes()
l