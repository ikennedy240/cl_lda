from census import Census
from us import states
import pandas as pd
import os
import numpy as np
import censusgeocode as cg
from time import sleep
import urllib.request
from urllib.parse import urlencode
import grequests
import requests
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import json
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
import regex as re

df = pd.read_csv('data/lda_processed5_17.csv',index_col=0,dtype = {'GEOID10':object,'blockid':object})



'''prepare a train/test set and a validation set'''
# Split data into training and test sets
with open('resources/hoods.txt') as f:
    neighborhoods = f.read().splitlines()
from sklearn.feature_extraction import stop_words
stop_words = neighborhoods + list(stop_words.ENGLISH_STOP_WORDS)
from sklearn.model_selection import train_test_split
pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*', flags=re.IGNORECASE)

x ='|'.join(stop_words)
df_small.clean_text = df_small.clean_text.str.lower().replace(,'')
df_small.clean_text.head(10).values
X = [[word for word in pattern.sub('', document.lower()).split() if len(word)>3] for document in df.clean_text.values]


regex_pat = re.compile(r'FUZ', flags=re.IGNORECASE)
pd.Series(['foo', 'bellvue', np.nan]).str.replace(r'\b(' + r'|'.join(stop_words) + r')\b\s*', 'bar')

X = [' '.join(x) for x in X]
X_train, X_test, y_train, y_test = train_test_split(X, df['high_white'], random_state=0)



"""CountVectorizer"""
from sklearn.feature_extraction.text import CountVectorizer
# Fit the CountVectorizer to the training data
vect = CountVectorizer(stop_words=stop_words,ngram_range=(1,3)).fit(X_train)
len(vect.get_feature_names())
X_train_vectorized =  vect.transform(X_train)
"""LogisticRegression"""
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty = 'l1',C=.1).fit(X_train_vectorized, y_train)

# Predict the transformed test documents
predictions = model.predict_proba(vect.transform(X_test))[:,1]
roc_auc_score(y_test, predictions)
confusion_matrix(y_test, model.predict(vect.transform(X_test)))
accuracy_score(y_test, model.predict(vect.transform(X_test)))
# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())
57946 - 64290
# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1]
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:50]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[-20:]]))


"""SVM"""
from sklearn.svm import SVC
svc = SVC(C=1000, probability = True).fit(X_train_vectorized,y_train)
predictions = svc.predict_proba(vect.transform(X_test))[:,1]
sub_predictions = svc.predict_proba(vect.transform(S_X))[:,1]
roc_auc_score(y_test, predictions)
roc_auc_score(S_y, sub_predictions)
accuracy_score(S_y, sub_predictions)

"""Random Forest Classifier"""
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train_vectorized, y_train)
predictions = rf.predict_proba(vect.transform(X_test))[:,1]
roc_auc_score(y_test, predictions)


sorted_import_index = rf.feature_importances_.argsort()


print('Smallest Import:\n{}\n'.format(feature_names[sorted_import_index[:20]]))
print('Largest Import:\n{}\n'.format(feature_names[sorted_import_index[-30:]]))

feature_names.shape
features = feature_names[sorted_import_index[-1000:]]
'''LogisticRegression With Limited Features'''
lf_vect = CountVectorizer(stop_words=stop_words,ngram_range=(1,4), vocabulary = features).fit(X_train)
len(lf_vect.get_feature_names())
X_train_vectorized = lf_vect.transform(X_train)
X_train_vectorized
lf_model = LogisticRegression(C=.1).fit(X_train_vectorized, y_train)

predictions = lf_model.predict_proba(lf_vect.transform(X_test))[:,1]
sub_predictions = lf_model.predict_proba(lf_vect.transform(S_X))[:,1]

print('AUC: ', roc_auc_score(y_test, predictions))
print('AUC: ', roc_auc_score(S_y, sub_predictions))

feature_names = np.array(lf_vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = lf_model.coef_[0].argsort()
opp_coef_index = (-lf_model.coef_[0]).argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1]
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:99]]))
print('Largest Coefs: \n{}'.format(feature_names[opp_coef_index[:1000]]))

Largest Coefs:

print(df[df.body_text.str.contains("brewer")].body_text.values)

"""TfidfVectorizer"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words=stop_words, ngram_range =(1,2), vocabulary=features).fit(X_train)
tfidf
len(tfidf.get_feature_names())
X_train_tfidf = tfidf.transform(X_train)
model = LogisticRegression().fit(X_train_tfidf, y_train)
predictions = model.predict(tfidf.transform(X_test))
sub_predictions = model.predict(tfidf.transform(S_X))

print('AUC: ', roc_auc_score(y_test, predictions))
print('AUC: ', roc_auc_score(S_y, sub_predictions))


feature_names = np.array(tfidf.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1]
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:20]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[-50:]]))

svc = SVC(C=100).fit(X_train_tfidf,y_train)
predictions = svc.predict(tfidf.transform(X_test))
roc_auc_score(y_test, predictions)

"""With Cross Validation"""
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
clf = make_pipeline(CountVectorizer(stop_words=stop_words,ngram_range=(1,4)), LogisticRegression(C=1))
score = cross_val_score(clf, X_train, y_train, cv=5, scoring = 'roc_auc')
score
fullmodel = LogisticRegression(C=.1).fit(vect.transform(df.listingText), df.high_white)
s_vectorized = vect.transform(S_X)
s_predictions = fullmodel.predict(s_vectorized)
roc_auc_score(S_y, s_predictions)
confusion_matrix(y_test, predictions)
confusion_matrix(S_y, s_predictions)

df = pd.read_csv('cl_lda.csv', index_col = 0)
