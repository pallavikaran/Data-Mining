
import re
from itertools import ifilterfalse
import nltk
import numpy as np

def clean_review(line):
     #remove non-alphanumeric characters and tokenize the reviews
        line = line.lower()
        line = re.sub(r'[^a-zA-Z0-9, ,\']', " ", line)
        tokens = nltk.word_tokenize(line)
        return tokens

def filterListofWords(wordList):
#filter words of length less than 2        
        wordList[:] = ifilterfalse(lambda i: (len(i)<3 ) , wordList)
        return wordList

def removeStopwords(wordList):
        #remove stop words
        stopwords = nltk.corpus.stopwords.words('english')
        wordList[:] = ifilterfalse(lambda i: (i in stopwords) , wordList)
        return wordList
        
def word_Lemmantization(wordList,wordNetLemma):
    #lemmatize words
	for i,word in enumerate(wordList):
		wordList[i] =wordNetLemma.lemmatize(word)
	
	return wordList
 
 # extract a specific column from the matrix
def column(matrix, i):
    return [row[i] for row in matrix]


# fill the null values in a column with the mean value
def fill_avg(colmn):
    mask = np.isnan(colmn)
    masked_arr = np.ma.masked_array(colmn,mask)
    mean_val = np.mean(masked_arr,axis=0)
    return masked_arr.filled(mean_val)

        
