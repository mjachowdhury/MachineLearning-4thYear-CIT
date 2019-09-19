# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:11:31 2019

@author: Mohammed
"""
"""
Question 2. Write a program that reads the contents of the Bram Stoker’s Dracula. The eBook can be 
downloaded at http://www.gutenberg.org/cache/epub/345/pg345.txt.The objective of this program is to read 
all the data from this file and output words that occur at a specific frequency within the text.  
Your program should read all words from the file. It should record all words that have a character 
length ofat least minWordLengthalong with their frequency of occurrencein the novel. Your program should 
then print each word along with the frequency of the word, which occurs more often in the novel than 
minWordOccurence.The result of minWordLength=3 and minWordOccurence=300 
should be:-“which”->636-“could”->458-“would”->408-“there”-> 508-“shall”->410
"""
def dracula():
    minWordLength = 5
    minWordOccurance = 100
    
    f = open("BramStokerBook.txt", "r")
    document = f.read()
    f.close()
    allWords = document.replace(".", "").replace(",", "").split()#going to create list
    wordOccurances = {}
    for word in allWords:
        if(len(word) >= minWordLength):
            if(word in wordOccurances):
                wordOccurances[word] = wordOccurances[word] + 1
            else:
                wordOccurances[word] = 1
    for word in wordOccurances:
        if wordOccurances[word] >= minWordOccurance:
            print(word + " : " + str(wordOccurances[word]))
            
dracula()
        
    