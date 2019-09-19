# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:09:04 2019

@author: Mohammed
"""

def Fibonacci(n):
    if n<0:
        print("Incorrect input")
    # First Fibonacci number is 0 
    elif n==1:
        return 0
    # Second Fibonacci number is 1 
    elif n==2:
        return 1
    else:
        return Fibonacci(n-1)+Fibonacci(n-2)
  
# Driver Program 
num=int(input("enter number: "))
"""Fibonacci(num)"""

print(Fibonacci(num))
