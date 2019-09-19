# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:47:39 2019

@author: Mohammed
this programm is not complete

"""
import re
import string
frequncy = {}

def main():
    
    fileOpen = open("BramStokerBook.txt", "r")
    text_string = fileOpen.read()
    text_string = fileOpen.read.lower()
    
    """if fileOpen == "r":
        contents = fileOpen.read()
        print(contents)"""
        
    fileRead = fileOpen.readlines()
    match_pattern = re.search(r'\b[a-z]{2,15}\b', text_string )
    
    for word in match_pattern:
        count = frequency.get(word,0)
        frequency[word] = count + 1
        
    frequency_list = frequency.keys()
    
    for x in fileRead:
        print(x)
        
if __name__=="__main__":
    main()
    
