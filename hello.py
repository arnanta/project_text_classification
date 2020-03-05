
import pandas as pd

import numpy as np

import json

import nltk

import re

import csv

import matplotlib.pyplot as plt 

import seaborn as sns

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

# function for text cleaning 

def clean_text(text):

    # remove backslash-apostrophe 

    text = re.sub("\'", "", text) 

    # remove everything except alphabets 

    text = re.sub("[^a-zA-Z]"," ",text) 

    # remove whitespaces 

    text = ' '.join(text.split()) 

    # convert text to lowercase 

    text = text.lower()     

    return text



# function for word frequency
    
def freq_words(x, terms = 30): 

  all_words = ' '.join([text for text in x]) 

  all_words = all_words.split() 

  fdist = nltk.FreqDist(all_words) 

  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})   

  # selecting top 20 most frequent words 

  d = words_df.nlargest(columns="count", n = terms)   

  # visualize words and frequencies

  plt.figure(figsize=(12,15)) 

  ax = sns.barplot(data=d, x= "count", y = "word") 

  ax.set(ylabel = 'Word') 

  plt.show()  
  
# function to remove stopwords

def remove_stopwords(text):

    no_stopword_text = [w for w in text.split() if not w in stop_words]

    return ' '.join(no_stopword_text)


#inference function
def infer_tags(q):

    q = clean_text(q)

    q = remove_stopwords(q)

    q_vec = tfidf_vectorizer.transform([q])

    q_pred = clf.predict(q_vec)

    return multilabel_binarizer.inverse_transform(q_pred)


data = pd.read_csv('book.txt', sep="\t", header=None)
data.columns = ["Wikipedia_ID","Freebase ID","Book_title","Author","Publication date","Book_genres","Plot"]
#data.fillna(0, inplace=True)    #used to fill NaN with some other value
#data.to_csv('book.tsv', sep = '\t')

data.drop(columns=["Freebase ID","Author","Publication date"], inplace=True) #Dropping unneccesary attributes
data.dropna(subset = ["Book_genres","Wikipedia_ID"], inplace=True)    #dropping rows with NaN values
#data.to_csv('book.tsv', sep = '\t')

genres = []

# extract genres
for i in data['Book_genres']: 
  genres.append(list(json.loads(i).values())) 



# add to 'data' dataframe  
data['genre_new'] = genres

all_genres = sum(genres,[])  #total number of genres
#len(set(all_genres))


"""
all_genres = nltk.FreqDist(all_genres)    #for plotting genres

# create dataframe

all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 

                              'Count': list(all_genres.values())})

g = all_genres_df.nlargest(columns="Count", n = 50) 

plt.figure(figsize=(12,15)) 

ax = sns.barplot(data=g, x= "Count", y = "Genre") 

ax.set(ylabel = 'Count') 
plt.show() """


data['clean_plot']=data['Plot'].apply(lambda x: clean_text(x))  #text cleanup using function
data.drop(columns=["Plot","Book_genres"], inplace=True)  #things like space, ", and all other unwanted stuffs

#nltk.download('stopwords')    #to download the list of stopwords

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

data['clean_plot'] = data['clean_plot'].apply(lambda x: remove_stopwords(x))

# print frequent words 

#freq_words(data['clean_plot'], 100)

#using multilabelbinarizer to convert multiple labels into binaries

from sklearn.preprocessing import MultiLabelBinarizer

#227 unique genres

multilabel_binarizer = MultiLabelBinarizer()

multilabel_binarizer.fit(data['genre_new'])



# transform target variable

y = multilabel_binarizer.transform(data['genre_new'])

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

xtrain, xval, ytrain, yval = train_test_split(data['clean_plot'], y, test_size=0.2, random_state=9)

xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score

lr = LogisticRegression()
clf = OneVsRestClassifier(lr)

# fit model on train data
clf.fit(xtrain_tfidf, ytrain)

# make predictions for validation set
y_pred = clf.predict(xval_tfidf)

y_pred[3]

# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)

t = 0.2 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)

for i in range(5): 
  k = xval.sample(1).index[0] 
  print("Book: ", data['Book_title'][k], "\nPredicted genre: ", infer_tags(xval[k])), print("Actual genre: ",data['genre_new'][k], "\n")
















