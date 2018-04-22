
import os
import csv
from pprint import pprint
from six import iteritems
from gensim import corpora, models, similarities
import pandas as pd
import numpy as np
from gensim import corpora

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

df =  pd.read_csv('immi_rated.csv')
df.text = df.text.str.replace('\n','')
df['text'].to_csv('immi_text.txt', sep=' ', index=False, header=False)


df[df.sent==1].text.sample(10).values

class MyCorpus(object):
    def __init__(self, text_file, dictionary=None):
        """
        Checks if a dictionary has been given as a parameter.
        If no dictionary has been given, it creates one and saves it in the disk.
        """
        self.file_name = text_file
        if dictionary is None:
            self.prepare_dictionary()
        else:
            self.dictionary = dictionary

    def __iter__(self):
        for line in open(self.file_name, encoding = 'latin-1'):
            # assume there's one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(line.lower().split())

    def prepare_dictionary(self):
        stop_list = set('for a of the and to in'.split())  # List of stop words which can also be loaded from a file.

        # Creating a dictionary using stored the text file and the Dictionary class defined by Gensim.
        self.dictionary = corpora.Dictionary(line.lower().split() for line in open(self.file_name, encoding = 'latin-1'))

        # Collecting the id's of the tokens which exist in the stop-list
        stop_ids = [self.dictionary.token2id[stop_word] for stop_word in stop_list if
                    stop_word in self.dictionary.token2id]

        # Collecting the id's of the token which appear only once
        once_ids = [token_id for token_id, doc_freq in iteritems(self.dictionary.dfs) if doc_freq == 1]

        # Removing the unwanted tokens using collected id's
        self.dictionary.filter_tokens(stop_ids + once_ids)

        # Saving dictionary in the disk for later use:
        self.dictionary.save('dictionary.dict')

sum(1 for line in open('immi_text.txt', encoding='utf=8'))
df.shape
corpus_memory_friendly = MyCorpus(text_file='immi_text.txt')
print(corpus_memory_friendly)

len(dictionary)

dictionary =  corpora.Dictionary.load('dictionary.dict')


lda = models.LdaModel(corpus_memory_friendly, id2word=dictionary, num_topics=20, minimum_probability=0.0)
for x in range(20):
    print(x, ': ', lda.print_topic(x), '\n')

corpus_lda = lda[corpus_memory_friendly]
count = 0
for line in corpus_lda:
    count+=1
    print(line)
    if count >20:
        break
x = pd.DataFrame(columns=columns)
count = 0

for line in corpus_lda:
    y = [x[1] for x in line]
    x = x.append(pd.DataFrame(data=y).transpose())
    count+=1
    if count % 1000 == 0:
        print(count)
    if count == 134825:
        break

df.shape
65434+135825
columns = list(range(20))
columns

x = x.append(pd.DataFrame(data=y).transpose())

x.shape
from gensim.matutils import corpus2dense
numpy_matrix = corpus2dense(corpus_lda, num_terms=20)
numpy_matrix.shape

visdata = plygensim.prepare(model, corpus, dictionary)
x = pyLDAvis.prepared_data_to_html(visdata)
pyLDAvis.display(visdata)
