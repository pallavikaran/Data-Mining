
import os.path
from gensim import corpora, models
import gensim


class MyCorpus(object):
    def __iter__(self):
        for line in open('reviews.txt'):
            yield dictionary.doc2bow(line.lower().split())


def CreateDictionary(texts):
    #creating a dictionay
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once]
             for text in texts]

    dictionary = corpora.Dictionary(texts)
    dictionary.save('deerwester.dict')
    return dictionary

    
def CreateDocVector(texts):
    
    if not (os.path.exists('deerwester.dict')):
        dictionary = CreateDictionary(texts)
    else:        
        dictionary = corpora.Dictionary.load('deerwester.dict')
        
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus


def TransformFeatureDoc(texts,num_topics=100):

    if not (os.path.exists('deerwester.dict')):
        dictionary = CreateDictionary(texts)
    else:
        #Load the Dictionary
        dictionary = corpora.Dictionary.load('deerwester.dict')
        
    corpus = CreateDocVector(texts)
        
    tfidf = models.TfidfModel(corpus)
    
    corpus_tfidf = tfidf[corpus]
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    corpus_lda = lda[corpus_tfidf]
    return corpus_lda
    

def getDocumentFeatures(TopicModel_data, num_topics=100):
        
    DocData = []
    for doc in TopicModel_data:
        DocData.append(doc)

    numpy_matrix = gensim.matutils.corpus2dense(DocData,num_topics)
    return numpy_matrix.T
        







