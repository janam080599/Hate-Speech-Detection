#!/usr/bin/env python
# coding: utf-8

#Basic Cleaning
import warnings
warnings.filterwarnings("ignore")                     #Ignoring unnecessory warnings

import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import nltk                                         #Natural language processing tool-kit
import string
#import spacy
from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer

from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
from gensim.models import Word2Vec                                   #For Word2Vec

data_path = r"C:\Users\akxjo\Desktop\SRI Work\SRI_mycode\train.csv"
data = pd.read_csv(data_path)
data_sel = data.head(10000)                                #Considering only top 10000 rows

# Shape of our data
data_sel.columns

##not necessary
def partition(x):
    if x == 1:
        return 'disastrous'
    return 'non'

##Labelling of Data
X = data['text'] # Features
y = data.target # Target variable

##LowerCasing
data["text_lower"] = data["text"].str.lower()
data.head(10000)


##Removal of Punctuations
# To drop the new column created in last cell
##data.drop(["text_lower"], axis=1, inplace=True)

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

data["text_wo_punct"] = data["text"].apply(lambda text: remove_punctuation(text))
data.head()

##Removal of Stop Words
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

data["text_wo_stop"] = data["text_wo_punct"].apply(lambda text: remove_stopwords(text))
data.head()

##Removal of Frequent Words(can also use TFIDF, this is another method)
#First get the most common words
from collections import Counter
cnt = Counter()
for text in data["text_wo_stop"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)
#Now Removing them
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

data["text_wo_stopfreq"] = data["text_wo_stop"].apply(lambda text: remove_freqwords(text))
data.head()

##Removal of Rare Words
n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

data["text_wo_stopfreqrare"] = data["text_wo_stopfreq"].apply(lambda text: remove_rarewords(text))
data.head()

##Stemming using PorterStemmer
from nltk.stem.porter import PorterStemmer

# Drop the two columns 
##df.drop(["text_wo_stopfreq", "text_wo_stopfreqrare"], axis=1, inplace=True) 

stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

data["text_stemmed"] = data["text"].apply(lambda text: stem_words(text))
data.head()

##Lemmatizing

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

data["text_lemmatized"] = data["text"].apply(lambda text: lemmatize_words(text))
data.head()



##Stemming, using Snowball
import re
temp =[]
snow = nltk.stem.SnowballStemmer('english')
for sentence in X:
    sentence = sentence.lower()                 # Converting to lowercase
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags
    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        #Removing Punctuations
    
    words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]   # Stemming and removing stopwords
    temp.append(words)
    
X = temp 
print(X[3])

##
sent = []
for row in X:
    sequ = ''
    for word in row:
        sequ = sequ + ' ' + word
    sent.append(sequ)

X = sent
print(X[3])

##BagOfWords
count_vect = CountVectorizer(max_features=500)
bow_data = count_vect.fit_transform(X)
print(bow_data[30])

##TFIDF
final_tf = X
tf_idf = TfidfVectorizer(max_features=500)
tf_data = tf_idf.fit_transform(final_tf)
print(tf_data[1])

##Word2Vec
w2v_data = X
splitted = []
for row in w2v_data: 
    splitted.append([word for word in row.split()])     #splitting words

train_w2v = Word2Vec(splitted,min_count=5,size=50, workers=4)

##Avg.Word2Vec
avg_data = []
for row in splitted:
    vec = np.zeros(50)
    count = 0
    for word in row:
        try:
            vec += train_w2v[word]
            count += 1
        except:
            pass
    avg_data.append(vec/count)

print(avg_data[1])

##TFIDF Word2Vec
tf_w_data = X
tf_idf = TfidfVectorizer(max_features=5000)
tf_idf_data = tf_idf.fit_transform(tf_w_data)
tf_w_data = []
tf_idf_data = tf_idf_data.toarray()
i = 0
for row in splitted:
    vec = [0 for i in range(50)]
    
    temp_tfidf = []
    for val in tf_idf_data[i]:
        if val != 0:
            temp_tfidf.append(val)
    
    count = 0
    tf_idf_sum = 0.0
    for word in row:
        try:
            count += 1
            tf_idf_sum = tf_idf_sum + temp_tfidf[count-1]
            vec += (temp_tfidf[count-1] * train_w2v[word])
        except:
            pass
    vec = (float)(1/tf_idf_sum) * vec
    tf_w_data.append(vec)
    i = i + 1

print(tf_w_data[1])


