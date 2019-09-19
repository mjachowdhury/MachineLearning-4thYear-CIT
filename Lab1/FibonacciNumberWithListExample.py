# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:32:39 2019

@author: Mohammed
"""
"""
Question 1. The Fibonacci numbers are the numbers in the following integer 
sequence:0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...By definition, the first 
two numbers in the Fibonacci sequence are 0 and 1, and each subsequent number is the 
sum of the previous two, i.e. 푓푖=푓푖−1+푓푖−2.Create a program that creates a list and 
will populate it with the first 40 Fibonacci numbers. The program should then ask the user 
to enter an integer value between 1 and 40 to indicate which number in the Fibonacci series 
they would like to see and the application should display that number. For example, if the 
user enters 13, the 13th number is 144.
"""
def fibonacci():
    num = 40
    f = [0,1] #creating list
    for i in range(2,num):
        f.append(f[i-1] + f[i-2])
    i = input("Which Fibonacci number do you want: ")
    print("The " + i + "th Fibonacci number is "+ str(f[int(i)-1]))

fibonacci()