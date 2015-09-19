import pandas as pd
from pandas import DataFrame
import nltk
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from nltk.corpus import wordnet as wn
import pickle
from scipy.sparse import hstack
from scipy import sparse

def loadTrain(filename):
    traindf = DataFrame.from_csv(filename, sep='\t',index_col=False)
    traindf1 =traindf[['essay_id','essay_set','essay','rater1_domain1','rater2_domain1','domain1_score']]
    filtercase =('b','c','d','e','f','g','h','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','.')
    X=traindf1['essay']
    Y=traindf1['rater1_domain1']
    return X,Y
    
def filter(X,value,strategy):
    if strategy=='order':
        return X[:value]
      
def generateFeatures(X):
    """ Features related to basic counters of words and sentences: """
    essay_word= [nltk.word_tokenize(raw) for raw in X ]
    ##Word Count
    essay_word_count= [len(nltk.word_tokenize(raw)) for raw in X]
    avg_essay_word_count=np.mean(essay_word_count) 
    ##Sentence Count
    sentence_per_essay = [  nltk.sent_tokenize(raw) for raw in X ]
    sentence_count_per_essay = [ len(sentences) for sentences in sentence_per_essay ]
    avg_sentence_count= np.mean(sentence_count_per_essay)
    ##Word tokens that have more than 6 characters divided by the number of word tokens
    count_word_tokens_greater_6= [ [len(wordgreaterthan6)/len(words) for wordgreaterthan6 in words if len(wordgreaterthan6)>6 ] for words in essay_word]
    ##Word tokens that have less than 4 characters divided by the number of word tokens
    count_word_tokens_less_4= [ [len(wordlessthan4)/len(words) for wordlessthan4 in words if len(wordlessthan4)<4 ] for words in essay_word]
    avg_word_per_sentence_per_essay= [] 
    from sets import Set
    count_lemma_set_for_essay=[]
    for words in essay_word:
        lemma_set=Set()
        for word in words:
            for synset in wn.synsets(word):
                for lemma in synset.lemmas():
                    lemma_set.add(lemma)
        count_lemma_set_for_essay.append(len(lemma_set))   
    """ Number of word tokens divided by number of sentences """
    word_token_by_count_sentences = [  float(count_word)/count_sentences for  count_word,count_sentences in zip(essay_word_count,sentence_count_per_essay)  ]
    
    """ Number of non-initial CAPS words divided by number of sentences """
    
    """ Number of characters in the essay divided by number of sentences"""
    character_count_by_count_sentences = [  float(len(x))/count_sentences for  x,count_sentences in zip(X,sentence_count_per_essay)  ]
    #Features related to nonlinear combinations of different attributes:
    fourth_root_essay_word_count = [  pow(c,0.25) for c in essay_word_count]
    countPOSTag = [ Counter(tag for word,tag in  nltk.pos_tag(nltk.Text(nltk.word_tokenize(raw)))) for raw in X]
    pendict = {1: 'CC', 2: 'CD', 3: 'DT', 4: 'EX', 5: 'FW', 6: 'IN', 7: 'JJ', 8: 'JJR', 9: 'JJS', 10: 'LS', 11: 'MD', 12: 'NN', 13: 'NNS', 14: 'NNP', 15: 'NNPS', 16: 'PDT', 17: 'POS', 18: 'PRP', 19: 'PRP$', 20: 'RB', 21: 'RBR', 22: 'RBS', 23: 'RP', 24: 'SYM', 25: 'TO', 26: 'UH', 27: 'VB', 28: 'VBD', 29: 'VBG', 30: 'VBN', 31: 'VBP', 32: 'VBZ', 33: 'WDT', 34: 'WP', 35: 'WP$', 36: 'WRB', 'NN': 12, 'FW': 5, 'PRP': 18, 'RB': 20, 'NNS': 13, 'NNP': 14, 'PRP$': 19, 'WRB': 36, 'CC': 1, 'PDT': 16, 'VBN': 30, 'WP$': 35, 'JJS': 9, 'JJR': 8, 'SYM': 24, 'VBP': 31, 'WDT': 33, 'JJ': 7, 'VBG': 29, 'WP': 34, 'VBZ': 32, 'DT': 3, 'POS': 17, 'TO': 25, 'LS': 10, 'VB': 27, 'RBS': 22, 'RBR': 21, 'EX': 4, 'IN': 6, 'RP': 23, 'CD': 2, 'VBD': 28, 'MD': 11, 'NNPS': 15, 'UH': 26,  '.':37 , 37:'.' , ':':38, 38:':','-NONE-':39,39:'-NONE-' , ',':40, 40:','}
    essay_POS_features=[]
    for i in xrange(0,len(countPOSTag)):
        vector=[0]*40
        for key in countPOSTag[i]:
            if key in pendict.values():
                vector[pendict[key]-1]=int(countPOSTag[i][key])
        essay_POS_features.append(vector)
    #Transforming into TF-IDF feature vectors
    vectorizer = CountVectorizer(min_df=1,decode_error='ignore')
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(X)
    X = transformer.fit_transform(X)
    #Combining features into a single feature matrix 
    X=hstack((X,essay_POS_features))
    X=hstack((X, sparse.csr_matrix(np.array(fourth_root_essay_word_count)).transpose() ))
    X=hstack((X, sparse.csr_matrix(np.array(character_count_by_count_sentences)).transpose() ))
    X=hstack((X, sparse.csr_matrix(np.array(word_token_by_count_sentences)).transpose() ))
    X=hstack((X, sparse.csr_matrix(np.array(count_lemma_set_for_essay)).transpose() ))
    #X=hstack((X, sparse.csr_matrix(np.array(count_word_tokens_less_4)).transpose() ))
    #X=hstack((X, sparse.csr_matrix(np.array(count_word_tokens_greater_6)).transpose() ))
    X=hstack((X, sparse.csr_matrix(np.array(sentence_count_per_essay)).transpose() ))
    X=hstack((X, sparse.csr_matrix(np.array(essay_word_count)).transpose() ))
    return X
     
def saveFeatures(X,values):
    with open('features'+values+'.pickle', 'wb') as handle:
        pickle.dump(X, handle)

def classifyandscore(name,X,Y):
    X_train, X_test, y_train, y_test =cross_validation.train_test_split(X.toarray(),Y, test_size=0.4, random_state=0)
    if(name=='SVM'):
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        print clf.score(X_test, y_test)
    elif (name=='RandomForest'):
        clf = RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
        print clf.score(X_test, y_test)

X,Y= loadTrain('data/training_set_rel3.tsv')
X=filter(X,10,'order')
Y=filter(Y,10,'order')
X=generateFeatures(X)
classifyandscore('SVM',X,Y)
