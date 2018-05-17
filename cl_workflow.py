"""
This Workflow takes in cleaned text labeled with one or more stratifiers
that the researcher is interested in using to break up text Analysis

### What if we collected ACS rent estimates and looked at places where the difference
between average rents in the sample were most different
"""
import preprocess
from datetime import datetime
import lda_output
import pandas as pd
import numpy as np
import regex as re
import logging
import cl_census
from gensim import corpora, models, similarities, matutils
from importlib import reload
reload(preprocess)

"""Start Logging"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


"""Import text as DataFrame"""
data_path = 'data/full_extract_5-6-18.csv'
df_full = pd.read_csv(data_path,dtype = {'GEOID10':object,'blockid':object})
df_full.columns
"""Creating Train-Test Split"""
# We did some preliminary analysis with some data that was separately scraped from cl
# So this just makes sure that any postids from that set end up in the training set (not the test set)

# #import the old data
df_old = pd.read_csv('archive/processed5_14.csv', index_col=0,dtype = {'GEOID10':object,'blockid':object, 'postid':object})
# # clean df_full's postIDs
df_full.postID = df_full.postID.str.replace('\D','')
df_full.shape
df_full.listingDate.unique().shape
# # Check how many matches we get
sum(df_full.postID.isin(set(df_old.postid)))
# # Start the training set with just those matched
# train = df_full[df_full.postID.isin(set(df_old.postid))]
# # Then making a capital X just those that didn't match
# X = df_full[~df_full.postID.isin(x.postID)]
# # Then a train test split on the X, set to end with even dfs, random_state 24 for repeatability
# from sklearn.model_selection import train_test_split
# X_train, X_test = train_test_split(X,train_size =0.4653708963603937, random_state=24)
# # add the matached listings to the full train set
# X_train = pd.concat([train, X_train])
# # compare sizes
# print(X_train.shape, X_test.shape)
# # make sure the test set doesn't have any matches with old data
# sum(X_test.postID.isin(set(x.postID)))
# # save to CSV
# X_test.to_csv('data/cl_test_set.csv')
# X_train.to_csv('data/cl_train_set.csv')

"""Import Training Data for Analysis"""

df = pd.read_csv('data/cl_train_set.csv')

# add census info
df = cl_census.mergeCLandCensus(df)
# clean text
df = df.assign(body_text = df.listingText)
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

"""Make Stratifying Columns"""
strat_list = []
new_col_list = []
for proportion in prop_list:
    strat_list.append(proportion+'_proportion')
    new_col_list.append('high_'+proportion)

"""Make Categorical Race_Income Variable"""
df = preprocess.make_stratifier(df, 'income', 'high_income')
df = preprocess.make_stratifier(df, 'poverty_proportion', 'high_poverty')
df = preprocess.make_stratifier(df, strat_list, new_col_list)

df['race_income'] = df.high_white

df = df.assign(race_income =  np.where(df.high_income==1, df.race_income+2, df.race_income))

labels = {3:"high income and high white", 2:"high income low white", 1:"low income high white", 0:"low income low white"}
df['race_income_label'] = [labels[x] for x in df.race_income]
df[['race_income', 'race_income_label']]
pd.crosstab(df.race_income_label, [df.high_income, df.high_white])

"""Make Other Context Columns"""
df['log_income'] = np.log(df.income)
df['clean_price'] = pd.to_numeric(df.price.str.replace(r'\$|,',''), errors = 'coerce')
df['log_price'] = np.log(df.clean_price)
df = df.dropna().reset_index(drop=True)
#seems like I had a lot of missing dates, so I set them to an arbitrary fake early date. This shouldn't be a problem for
#the later data set
df[df.scraped_month.isnull()] = df[df.scraped_month.isnull()].assign(scraped_day=32, scraped_month=13, scraped_year=2016)
sum(df.scraped_month.isnull())
# check missing values by column
for x in df.columns:
    print(x,sum(df[x].isnull()))
#check the shape if we drop all nulls
df.dropna().shape
# drop all nulls
df = df.dropna()
# save processed data
df_full = df_full.drop_duplicates('postid')
df_full.to_csv('data/processed5_14.csv')

"""Break off the section to use the LDA"""
# break off a df with unique addresses
df_full = df
df = df_full.drop_duplicates('address')
# check for other dupes
df.shape
preprocess.clean_duplicates(df[df.address.str.len()>2], text_col='address', method = 'lsh', char_ngram=2, thresh=.75, bands=20).shape
df = preprocess.clean_duplicates(df[df.address.str.len()>2], text_col='address', method = 'lsh', char_ngram=2, thresh=.75, bands=20)
preprocess.clean_duplicates(df, text_col='clean_text', method='lsh').shape
# drop dupes with jaccard distance > .9
df= preprocess.clean_duplicates(df, text_col='clean_text', method='lsh')
df.to_csv('data/dropped_address5_14b.csv')
df = pd.read_csv('data/dropped_address5_14b.csv',index_col=0,dtype = {'GEOID10':object,'blockid':object})
"""Make Corpus and Dictionary"""
with open('resources/hoods.txt') as f:
    neighborhoods = f.read().splitlines()
from sklearn.feature_extraction import stop_words
hood_stopwords = neighborhoods + list(stop_words.ENGLISH_STOP_WORDS)
corpus, dictionary = preprocess.df_to_corpus([str(x) for x in df.clean_text], stopwords=hood_stopwords)

"""Run Lda Model"""
n_topics = 30
n_passes = 50
#Run this if you have access to more than one core set workers=n_cores-1
model = models.ldamulticore.LdaMulticore(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0, workers=3)
#otherwise run this
#model = models.LdaModel(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0)
#save the model for future use
now = datetime.now()
model.save('models/model'+str(now.month)+'_'+str(now.day))
#model = models.LdaModel.load('models/model4_23')
"""Merge LDA output and DF"""
#Make LDA corpus of our Data
lda_corpus = model[corpus]
#make dense numpy array of the topic proportions for each document
doc_topic_matrix = matutils.corpus2dense(lda_corpus, num_terms=n_topics).transpose()
df = df.reset_index(drop=True).join(pd.DataFrame(doc_topic_matrix))
# Make top_topic for each document
df = df.assign(top_topic=df[list(range(n_topics))].idxmax(1))

#do the same process for the full_df
full_corpus, dictionary = preprocess.df_to_corpus([str(x) for x in df_full.clean_text], stopwords=hood_stopwords, dictionary=dictionary)
full_lda = model[full_corpus]
full_doc_topic_matrix = matutils.corpus2dense(full_lda, num_terms=n_topics).transpose()
df_full = df_full.reset_index(drop=True).join(pd.DataFrame(full_doc_topic_matrix))
df_full = df_full.assign(top_topic=df_full[list(range(n_topics))].idxmax(1))
df_full.shape
df_full= df_full.drop(list(range(30)),1)
len(full_corpus)
"""Look at the distributions of the different topics"""
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(df[[26]].sort_values(by=26, ascending=False).head(400).values)
plt.hist(df.groupby('GEOID10').mean()[[4]].dropna().values)
df.columns
df.groupby('race_income_label').mean()[26]
plt.hist(df[df.race_income==0][12].values)

plt.axis([.1, 1, 0, 500])
plt.hist(df.groupby('GEOID10').mean()[['income']].dropna().values)


"""Use stratifier to create various comparisions of topic distributions"""
strat_col = 'high_white'
lda_output.compare_topics_distribution(df, n_topics, strat_col)
mean_diff = lda_output.summarize_on_stratifier(df, n_topics, strat_col)
mean_diff = mean_diff.sort_values('difference', ascending=False)
mean_diff

"""Try a Multinomial LogisticRegression"""
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import MNLogit
# we use the full df for the regression because we want to weight results by the
# existence of different ads in different neighborhoods, not just unique addresses
X = df_full[["black_proportion","asian_proportion","latinx_proportion","log_income","log_price"]]
y = df_full.top_topic
LR = LogisticRegression()
LR.fit(X,y)
lr_coefs = pd.DataFrame(LR.coef_).rename({0:"black_proportion",1:"asian_proportion",2:"latinx_proportion",3:"log_income",4:"log_price"}, axis=1)
lr_coefs
lr_coefs = lr_coefs.assign(abs_black = abs(lr_coefs.black_proportion)).sort_values("abs_black", ascending=False)
mean_diff = lr_coefs.merge(lda_output.summarize_on_stratifier(df, n_topics, 'high_black'), left_index=True, right_index=True)
mean_diff.drop(['abs_black','difference', 'proportion'], axis=1, inplace=True)

"""Compute Standard Errors"""
from statsmodels.discrete.discrete_model import Logit, MNLogit

top_topics = mean_diff.topic.values
tmp = y.copy()
for i in range(10):
    y_31 = pd.Series(np.where(tmp==top_topics[i],1,0))
    sm_logit = Logit(endog=y_31, exog=X.reset_index(drop=True))
    print(top_topics[i], '\n', sm_logit.fit().summary())
MN_logit = MNLogit(endog=y.astype(str), exog=X)
MN_logit.fit(method='nm',maxiter=5000, maxfun=5000).summary()

"""Make Predictions at the means"""
sample_data = pd.DataFrame(index=range(0,50),  columns=["asian_proportion","latinx_proportion","log_income","log_price"])
sample_data[["asian_proportion","latinx_proportion","log_income","log_price"]] = df[["asian_proportion","latinx_proportion","log_income","log_price"]].mean().values.reshape(1,-1)
b_min = df.black_proportion.min()
b_max = df.black_proportion.max()
sample_data["black_proportion"] = range(1,51)
sample_data.black_proportion = ((b_max-b_min)/50*sample_data.black_proportion)
sample_data = sample_data[["black_proportion","asian_proportion","latinx_proportion","log_income","log_price"]]
predcited_values = LR.predict_proba(sample_data.values)
[np.argmax(x) for x in predcited_values]

df.to_csv('data/data514.csv')


"""Produces useful output of topics and example texts"""

mean_diff
now = datetime.now()
reload(preprocess)
lda_output.text_output(df, text_col='body_text', filepath='output/cl'+str(now.month)+'_'+str(now.day)+'.txt', model= model, sorted_topics=mean_diff, strat_col='race_income_label', cl=True, print_it = False)








pairs = list(preprocess.candidate_duplicates(df[df.address.str.contains('missing')], char_ngram=5))
droplist = set([min(pair[0],pair[1]) for pair in pairs])
missing = df[df.address.str.contains('missing')].reset_index()
df = df.drop(missing['index'])
df.shape
df =  preprocess.clean_duplicates(df[df.address.str.len()>2], text_col='address', method = 'lsh', char_ngram=2, thresh=.75, bands=20)

address_dupes = preprocess.candidate_duplicates(df[df.address.str.len()>2],text_col='address', char_ngram=2, seeds=100, bands=20, hashbytes=4)
address_dupes.sort_values(0)

len(address_dupes)

def shingles(text, char_ngram=5):
    return set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram))

from lsh import cache, minhash
# calculate jaccard similarity between two lists of shingles
def jaccard(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)
lines = df.address
similarities = []
hasher = minhash.MinHasher(seeds=1000, char_ngram=1, hashbytes=4)
lshcache = cache.Cache(bands=10, hasher=hasher)
for (line_a, line_b) in address_dupes:
    doc_a, doc_b = lines.iloc[line_a], lines.iloc[line_b]
    shingles_a = shingles(lines.iloc[line_a], char_ngram)
    shingles_b = shingles(lines.iloc[line_b], char_ngram)
    try:
        jaccard_sim = jaccard(shingles_a, shingles_b)
    except:
        print("jaccard failed at \n", line_a, line_b,"with \n", shingles_a, shingles_b)
        jaccard_sim = 0
    fingerprint_a = set(hasher.fingerprint(doc_a.encode('utf8')))
    fingerprint_b = set(hasher.fingerprint(doc_b.encode('utf8')))
    minhash_sim = len(fingerprint_a & fingerprint_b) / len(fingerprint_a | fingerprint_b)
    similarities.append((lines.iloc[line_a], lines.iloc[line_b], jaccard_sim, minhash_sim))
df.address.iloc[2975]
jaccard(shingles(lines[3183], char_ngram=1), shingles(lines[3466], char_ngram=1))
[shingles(x, char_ngram=2) for x in df['address'].str.lower().values]

list(zip(list(y.index[1:]),list(y.index[0:])))
