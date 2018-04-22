"""
This Workflow takes in cleaned text labeled with one or more stratifiers
that the researcher is interested in using to break up text Analysis
"""
import preprocess
import lda_output
import pandas as pd
import logging
from gensim import corpora, models, similarities, matutils

"""Start Logging"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""Import text as DataFrame"""
data_path = 'data/narrow_immi_rated.csv'
df = pd.read_csv(data_path)
corpus, dictionary = preprocess.df_to_corpus(df.text)

"""Run Lda Model"""
n_topics = 10
n_passes = 1
#Run this if you have access to more than one core set workers=n_cores-1
model = models.ldamulticore.LdaMulticore(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0, workers=3)
#otherwise run this
#model = models.LdaModel(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0)
#save the model for future use
model.save('models/4_20_immi')

"""Merge LDA output and DF"""
#Make LDA corpus of our Data
lda_corpus = model[corpus]
#make dense numpy array of the topic proportions for each document
doc_topic_matrix = matutils.corpus2dense(lda_corpus, num_terms=n_topics).transpose()
df = df.reset_index(drop=True).join(pd.DataFrame(doc_topic_matrix))

"""Use stratifier to create various comparisions of topic distributions"""
strat_col = 'sent'
lda_output.rfc_distribution(df, n_topics, strat_col)
lda_output.compare_topics_distribution(df, n_topics, strat_col)
mean_diff = lda_output.summarize_on_stratifier(df, n_topics, strat_col)

"""Produces useful output of topics and example texts"""
lda_output.text_output(df, text_col='text', filepath='output/immi_4_20.txt', model=model, strat_col='sent')
