
from nltk.stem import WordNetLemmatizer

from ComputeFeatures import *
from TextPreprocessing import *
from ModelData import *

def mineReviews(fileName):
    
    wordNetLemma = WordNetLemmatizer()
    IgnoreColumn = 'total_reviews_written_by_user'
    reviewFile = open(fileName)
    header = reviewFile.next().rstrip().split(',')
 
    y_data = []
    X_data = []
    
    reviewData = []
    
    for line in reviewFile:
        splitted = line.rstrip().split(',')

        if fileName == 'consolidated_yelp_testing_data.csv':
            iIndex = header.index('review')
        else:
            iIndex = 1   
             
        wordList =  clean_review(splitted[iIndex])  
        wordList = filterListofWords(wordList)  
        wordList = removeStopwords(wordList)   
        
        splitted[iIndex] = len(wordList)
     
        wordList = word_Lemmantization(wordList,wordNetLemma)
        
        reviewData.append(wordList)
       
        label = int(splitted[0])
        features = []
        
        if fileName == 'consolidated_yelp_testing_data.csv':
            y_data.append(splitted[29])
 
        begin = iIndex
        end = iIndex+29
        for item in splitted[begin:end]:
            if(item == 'NULL'):
                features.append(np.nan)
            else:
                features.append(float(item))
    
        if fileName != 'consolidated_yelp_testing_data.csv':
            y_data.append(label)
        X_data.append(features)
        
    reviewFile.close()
    
    TopicModel_data = TransformFeatureDoc(reviewData)
    reviewData = [] 
    reviewFeature = []
    reviewFeature = getDocumentFeatures(TopicModel_data)
    
    if fileName != 'consolidated_yelp_testing_data.csv':
        y_data = np.array(y_data)
    X_data = np.array(X_data)
    
    if(len(X_data) != len(reviewFeature)):
        raise "Length of X_data and reviewFeature is not matched"
    else:
        X_data = np.concatenate((X_data, reviewFeature), axis=1)
    
    
    try:
        iIndex = header.index(IgnoreColumn)-1
        X_data = np.delete(X_data,iIndex,1)
    except:
        print("\t No column matched for removing")
    
    #Handle NULL(nan) values
    X_data[:,2] = fill_avg(column(X_data,2))
     
    X_data = standardize_features(X_data) 
    
    return y_data, X_data
    
