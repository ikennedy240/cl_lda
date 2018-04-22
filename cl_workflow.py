"""
This Workflow takes in cleaned text labeled with one or more stratifiers
that the researcher is interested in using to break up text Analysis
"""
import preprocess
import lda_output
import pandas as pd
import logging
import cl_census
from gensim import corpora, models, similarities, matutils

"""Start Logging"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


"""Import text as DataFrame"""
data_path = 'data/seattle_cleaned4_22.csv'
df = pd.read_csv(data_path, index_col=0,dtype = {'GEOID10':object,'blockid':object})


"""Preprocess and Merge New Data if Needed"""
new_data_path = "data/chcl.csv"
new_df = pd.read_csv(new_data_path, index_col=0, dtype = {'GEOID10':object,'blockid':object})
"""preprocess"""
new_df['body_text_clean'] = preprocess.cl_prep_for_lda(new_df.body)
df = pd.concat([df,new_df[df.columns]]).reset_index(drop=True)
reload(preprocess)
df.shape
df_300 = df
df_300 = preprocess.clean_duplicates(df.reset_index(drop=True), method=300)
df_300
#matching colnames with chris
#colnames = ['GEOID10', 'address', 'blockid', 'body_text', 'latitude', 'longitude','postid', 'price','scraped_month','scraped_day']
#merge = new_df[['GEOID10', 'matchAddress', 'GISJOIN', 'listingText','lat','lng','postID', 'scrapedRent', 'listingMonth', 'listingDate']]
#name_dict = dict(zip(merge.columns,colnames))
#merging with chris
#merge = merge.rename(name_dict, axis=1)
#merge.scraped_day = merge.scraped_day.str.extract(r'(\d+$)')
#merge = cl_census.mergeCLandCensus(merge)
#df = pd.concat([df[merge.columns], merge]).reset_index(drop=True)
merge['body_text_clean'] = preprocess.cl_prep_for_lda(merge.body_text)
df.to_csv('data/seatle_complete4_22.csv')
df.body_text_clean.dropna(inplace=True)
df_300.shape
df_300.body_text_clean.dropna(inplace=True)
df_300 = df_300.reset_index(drop=True)


"""Make Corpus and Dictionary"""
corpus, dictionary = preprocess.df_to_corpus([str(x) for x in df_300.body_text_clean])

"""Run Lda Model"""
n_topics = 50
n_passes = 20
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
