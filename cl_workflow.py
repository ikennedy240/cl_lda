"""
This Workflow takes in cleaned text labeled with one or more stratifiers
that the researcher is interested in using to break up text Analysis
"""
import preprocess
from datetime import datetime
import lda_output
import pandas as pd
import numpy as np
import logging
import cl_census
from gensim import corpora, models, similarities, matutils
from importlib import reload
reload(preprocess)

"""Start Logging"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


"""Import text as DataFrame"""
data_path = 'data/processed4_22.csv'
df = pd.read_csv(data_path, index_col=0,dtype = {'GEOID10':object,'blockid':object})

"""Import New Data if Needed"""
new_data_path = "data/chcl.csv"
new_df = pd.read_csv(new_data_path, index_col=0, dtype = {'GEOID10':object,'blockid':object})

"""Preprocess Raw Data"""
# remove duplicates
df.shape
df = df.drop_duplicates(subset='body_text')
df = preprocess.clean_duplicates(df)
# add census info
df = cl_census.mergeCLandCensus(df)
# clean text
df['clean_text'] = preprocess.cl_clean_text(df.body_text)
df.body_text = preprocess.cl_clean_text(df.body_text, body_mode=True)

# remove empty text
df = df[~df.clean_text.str.len().isna()]
df = df[df.clean_text.str.len()>100]
df = df.reset_index(drop=True)

#matching colnames with chris
# colnames = ['GEOID10', 'address', 'blockid', 'body_text', 'latitude', 'longitude','postid', 'price','scraped_month','scraped_day']
# merge = new_df[['GEOID10', 'matchAddress', 'GISJOIN', 'listingText','lat','lng','postID', 'scrapedRent', 'listingMonth', 'listingDate']]
# name_dict = dict(zip(merge.columns,colnames))
# #merging with chris
# merge = merge.rename(name_dict, axis=1)
# merge.scraped_day = merge.scraped_day.str.extract(r'(\d+$)')
# df_merged = pd.concat([df[merge.columns], merge]).reset_index(drop=True)
# df_merged.shape
# df_merged.to_csv('data/merged_raw.csv')


"""Make New Demo Columsn"""
prop_list = ['white','black','aindian','asian','pacisland','other','latinx']
df['poverty_proportion'] = df.under_poverty/df.total_poverty
# making census vars into proportions
for proportion in prop_list:
    df[proportion+'_proportion'] = df[proportion]/df.total_RE

"""Make Stratifying Column"""
df.columns

strat_list = []
new_col_list = []
for proportion in prop_list:
    strat_list.append(proportion+'_proportion')
    new_col_list.append('high'+proportion)

preprocess.make_stratifier(x, 'income', 'high_income')
df['race_income'] = df.highwhite
df = df.assign(race_income =  np.where(df.high_income==1, df.race_income+2, df.race_income))

labels = {3:"high income and high white", 2:"high income low white", 1:"low income high white", 0:"low income low white"}
df['race_income_label'] = [labels[x] for x in df.race_income]
df[['race_income', 'race_income_label']]
pd.crosstab(df.race_income_label, [df.high_income, df.highwhite])

df.to_csv('data/processed4_22.csv')
"""Make Corpus and Dictionary"""
with open('resources/hoods.txt') as f:
    neighborhoods = f.read().splitlines()
from sklearn.feature_extraction import stop_words
hood_stopwords = neighborhoods + list(stop_words.ENGLISH_STOP_WORDS)
corpus, dictionary = preprocess.df_to_corpus([str(x) for x in df.clean_text], stopwords=hood_stopwords)

"""Run Lda Model"""
n_topics = 50
n_passes = 50
#Run this if you have access to more than one core set workers=n_cores-1
model = models.ldamulticore.LdaMulticore(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0, workers=3)
#otherwise run this
#model = models.LdaModel(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0)
#save the model for future use
now = datetime.now()
model.save('models/model'+str(now.month)+'_'+str(now.day))

"""Merge LDA output and DF"""
#Make LDA corpus of our Data
lda_corpus = model[corpus]
#make dense numpy array of the topic proportions for each document
doc_topic_matrix = matutils.corpus2dense(lda_corpus, num_terms=n_topics).transpose()
df = df.reset_index(drop=True).join(pd.DataFrame(doc_topic_matrix))



"""Use stratifier to create various comparisions of topic distributions"""
strat_col = 'total_poverty'
lda_output.rfc_distribution(df, n_topics, strat_col)
lda_output.compare_topics_distribution(df, n_topics, strat_col)
mean_diff = lda_output.summarize_on_stratifier(df, n_topics, strat_col)
mean_diff = mean_diff.sort_values('difference', ascending=False)

"""Try a Multinomial LogisticRegression"""
from sklearn.linear_model import LogisticRegression
X = df[list(range(50))]
y = df.race_income_label
LR = LogisticRegression()
LR.fit(X,y)
highwhite_coefs = pd.DataFrame(LR.coef_).rename(labels).transpose()

lr_coefs.sort_values("low income low white")
lr_coefs['summed_diff'] = abs((lr_coefs['low income low white']-lr_coefs['low income high white']) + (lr_coefs['high income low white']-lr_coefs['high income and high white']))
mean_diff = lr_coefs.sort_values('summed_diff', ascending=False)
lda_output.get_formatted_topic_list(model)

"""Produces useful output of topics and example texts"""
reload(lda_output)
lda_output.text_output(df, text_col='body_text', filepath='output/cl'+str(now.month)+'_'+str(now.day)+'.txt', model= model, sorted_topics=mean_diff)
