# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:06:16 2019

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

#this programm not complete but its working

import re
import string

frequency = {}
document_text = open('BramStokerBook.txt', 'r')
text_string = document_text.read().lower()
match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)
 
for word in match_pattern:
    count = frequency.get(word,0)
    frequency[word] = count + 1
     
frequency_list = frequency.keys()
 
for words in frequency_list:
    print (words, frequency[words])