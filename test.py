
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

with open('resources/hoods.txt') as f:
    hoods = f.read()
stopwords = hoods + " ".join(list(stop_words.ENGLISH_STOP_WORDS))
with open('resources/stopwords.txt') as f:
    f.write(stopwords)
stopwords = stop_words.ENGLISH_STOP_WORDS
df = pd.read_csv("data/cl_dropped.csv", index_col = 0).reset_index(drop=True).dropna()
name = hoods[0]
name
len(hoods)
x = [len(x) for x in hoods]
hoods[]
[x.index(y) for y if]
sub_hoods = hoods[:10]
for name in sub_hoods:
    df['body_text'] = df.body_text.str.replace(name, "#PLACENAME")
    if hoods.index(name) % 100 == 0:
        print(hoods.index(name))


hoods = pd.read_csv('resources/hoods.txt')
hoods = [str.strip(x) for x in hoods.columns]
str.split(hoods)[0]
name = 'amazon'
hoods.index(name)
hoods.index(name) % 100

hoods
