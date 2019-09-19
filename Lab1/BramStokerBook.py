# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:47:39 2019

@author: Mohammed
"""
def main():
    
    fileOpen = open("BramStokerBook.txt", "r")
    
    """if fileOpen == "r":
        contents = fileOpen.read()
        print(contents)"""
        
    f1 = fileOpen.readlines()
    for x in f1:
        print(x)
        
if __name__=="__main__":
    main()
    
