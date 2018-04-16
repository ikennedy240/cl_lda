#i need all these things
import logging
import os
from gensim import corpora, models, similarities, matutils
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction import stop_words
from operator import itemgetter
import numpy as np
import regex as re
import difflib

#logging because gensim isn't very verbose
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# these stopwords include neighborhood names and some other domain specific terms
with open('resources/hoods.txt') as f:
    hoods = f.read()
stopwords = hoods + " ".join(list(stop_words.ENGLISH_STOP_WORDS))
with open('resources/stopwords.txt') as f:
    f.write(stopwords)
stopwords = stop_words.ENGLISH_STOP_WORDS


#takes a list of documents and returns a corpus and dictionary
def df_to_corpus(documents):
    #turns each tweet into a list of words
    texts = [[word for word in document.lower().split() if word not in stopwords] for document in documents]
    #makes a dictionary based on those texts (this is the full df) and saves it
    dictionary = corpora.Dictionary(texts)
    #applies a bag of words vectorization to make the texts into a sparse matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    return(corpus, dictionary)

#this is a bad filename for what is a large set of craigslist data
df = pd.read_csv("data/seattle_cleaned.csv",  index_col = 0, dtype = {'GEOID10':object,'blockid':object}).dropna()

#clean up the text for readablility
#df.body_text = df.body_text.str.replace('\n|\r',' ').str.replace(r'\s+',' ').str.replace(r'^\W+','').dropna()
#drop texts with no word characters after cleaning, then resets the index, preserving the
#imported index as id (this is so I can mess around with what gets dropped)

#df = df[df.body_text.str.contains(r'\w')]

#there were a lot of almost identical texts, this is a dirty way to filter them out
df['body_100']=df.body_text.str.slice(stop=100)
df = df.drop_duplicates('body_100').drop('body_100', axis=1).reset_index().rename({'index':'id'},axis=1)
df.body_text.str.contains('dollasigns').sum()
test = df.body_text[df.body_text.str.contains('\$+')]
test.shape
df.body_text = df.body_text.str.replace(r'!!!!+',' shii ').str.replace(r'!!!', ' mitsu ').str.replace(r'!!', ' nii ').str.replace(r'!', 'ichi')
df.body_text = df.body_text.str.replace('\$\$+','dollasigns').str.replace('\$', 'dollasign')
df['body_text_clean'] = df.body_text.str.replace(r'\W',' ')
#calls above funtion to make a corpus and dictionary, only passes word characters
df.body_text_clean = df.body_text_clean.str.replace('shii', '!!!!').str.replace('mitsu', '!!!').str.replace('nii', '!!').str.replace('ichi', '!').str.replace('dollasigns', '$$').str.replace('dollasign', '$')
df.to_csv('data/seattle_cleaned.csv')

corpus, dictionary = df_to_corpus(df.body_text_clean.values)
dictionary.save('models/cl_dictionary4_15.dict')
#Then fit an LDA model (or load model 4_12 below)
n_topics = 50
n_passes = 20
#Run this if you have access to more than one core set workers=n_cores-1
model = models.ldamulticore.LdaMulticore(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0, workers=3)
#otherwise run this
model = models.LdaModel(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0)

#save the model for future use
model.save('models/4_15model')
#reload an old model
#model = models.LdaModel.load('models/4_12model')

#Make LDA corpus of our Data
cl_lda = model[corpus]
#make dense numpy array of the topic proportions for each document
doc_topic_matrix = matutils.corpus2dense(cl_lda, num_terms=n_topics).transpose()
#join topic proportions to the documents
df.columns

df = df.reset_index(drop=True).join(pd.DataFrame(doc_topic_matrix))

df.shape
#save the mearged df so we don't have to run the model next time
df.to_csv('data/cl_lda4_15.csv')
#use these lines to load saved data:
#df= pd.read_csv('cl_lda4_12.csv', index_col=0)
#df.rename(dict(zip(df.columns[3:],range(n_topics))),axis=1, inplace=True)

df.high_white = np.where(df['percent_white']>=85, 1, 0)
(df['percent_white']>=85).sum()
pd.cut(df.percent_white,10).value_counts()
df.columns
#make a list of topics and clean it for readablility
topics = ["Topic "+str(x[0])+": "+re.sub(r'\s+', ', ',re.sub(r'\d|\W',' ',x[1]))[2:-2] for x in model.print_topics(n_topics,20)]

#Count the occurence of each topic by high_white
no_text = df[list(range(n_topics))].join(df.high_white)
no_text[no_text<.01] = 0
no_text = np.sign(no_text)
#I've run this with means too, but I'll eventually want to do a chi-squared test
#so I think counts are better
all = no_text.mean()
high = no_text[no_text.high_white==1].mean()
low = no_text[no_text.high_white==0].mean()

#make a df of that data
mean_diff = pd.DataFrame(data=[all, high, low]).transpose().rename(columns = {0:'all_r', 1:'high_white', 2:'low_white'})
topic_comparison = pd.DataFrame(data=[all.drop('high_white').sort_values(ascending=False).index, high.drop('high_white').sort_values(ascending=False).index, low.drop('high_white').sort_values(ascending=False).index]).transpose().rename(columns = {0:'all_r', 1:'high_white', 2:'low_white'})
#calculate the absolute value of the difference between the two categories)
mean_diff['difference']  = abs(mean_diff.high_white - mean_diff.low_white)
mean_diff['prop'] = abs(mean_diff.high_white/mean_diff.low_white)
mean_diff.sort_values(by='prop', ascending=False)
topic_comparison

#do a quick Random Forest Classification to see which topics are most useful for distinguishing
#beetween high and low white neighborhoods
#SKIP DOWN AND USE top_ten_mean TO AVOID SKLEARN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[list(range(n_topics))], df['high_white'], random_state=0)
"""Random Forest Classifier"""
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, n_jobs = 3).fit(X_train,y_train)
predictions = rf.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, predictions)
sorted_import_index = rf.feature_importances_.argsort()
rf.score(X_test, y_test)
#make a summary dataframe of the relative counts for the 10 most important topics
#this one is orders them based off the RFC (ie these topics are good for sorting
#based on their proportion in a document)
sorted_rfc = mean_diff.loc[list(sorted_import_index)].iloc[::-1]
sorted_rfc
#this one based off the difference in occurence (ie these topics occur at
#more than 1% in different numbers in the high_white and low_white samples)
sorted_mean = mean_diff.sort_values('prop', ascending=False)[1:]
#pick which one to use for output
sorted_topics = sorted_mean

#looping to print top matches for the most imported topics for qual analysis

text_output = 'output/4_15.txt'
sample_topics = 10
sample_texts = 5
with open(text_output, 'w', encoding='utf-8') as f:
    for j in sorted_topics.index:
        print("Topic #", j,' occurred in \n', round(sorted_topics.loc[j],2), '\n', topics[int(j)], file=f)
    print("\n ------- Sample Documents ------- \n\n", file=f)
    for j in sorted_topics.index[1:sample_topics]:
        print("Topic #", j,' occurred in \n', round(sorted_topics.loc[j]), '\n', topics[int(j)], file=f)
        print("\n Top 5 answers fitting topic", j, "are: \n \n", file=f)
        for i in range(sample_texts):
            tmp = df.sort_values(by=j, ascending=False).iloc[i]
            print("Topic", j, "Rank", i+1, file=f)
            print(": \n Was ",round(tmp.loc[j]*100,2),"percent topic", j, ':\n', tmp.body_text, '\n', file=f)
