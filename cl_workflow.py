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
reload(lda_output)

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

df = pd.read_csv('data/cl_train_set.csv',index_col=0,dtype = {'GEOID10':object,'blockid':object, 'postID':object})
df.shape
(64750, 39)
# add census info
df = cl_census.mergeCLandCensus(df)
# clean text
df = df.assign(body_text = df.listingText)
df['clean_text'] = preprocess.preprocess(df.body_text)
df.body_text = preprocess.cl_clean_text(df.body_text, body_mode=True)


# remove empty text
df = df[~df.clean_text.str.len().isna()]
df = df[df.clean_text.str.len()>100]
df = df.reset_index(drop=True)
#at this point training data is reduced from 64750 listings to 64578 listings

#matching colnames with chris
colnames = ['address', 'blockid', 'latitude', 'longitude','postid', 'price']
chris_cols = ['matchAddress', 'GISJOIN','lat','lng','postID', 'scrapedRent']
name_dict = dict(zip(chris_cols,colnames))
df = df.rename(name_dict, axis=1)



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


"""Make Two-Way Race_Income Categorical Variable"""
# df['race_income'] = df.high_white
#
# df = df.assign(race_income =  np.where(df.high_income==1, df.race_income+2, df.race_income))
#
# labels = {3:"high income and high white", 2:"high income low white", 1:"low income high white", 0:"low income low white"}
# df['race_income_label'] = [labels[x] for x in df.race_income]
# df[['race_income', 'race_income_label']]
# pd.crosstab(df.race_income_label, [df.high_income, df.high_white])

"""Make Other Context Columns"""
df['log_income'] = np.log(df.income)
df['clean_price'] = df.cleanRent
df['log_price'] = np.log(df.clean_price)
#df = df.dropna().reset_index(drop=True)
# Check Missing Values:
df.isnull().sum()
# drop missing values in those columns we'll use later. 'postOrigin' has many
# missing values, but we won't drop them because that variable doesn't matter
df = df.dropna(subset=['postid','income'])
df.shape
(64187, 61)
# with this preprocessing, we've lost a few more listings and the total set should be
# 64290 listings in the training data
# save processed data
df.to_csv('data/train_processed5_25.csv')

"""Break off the section to use the LDA"""
# break off a df with unique addresses
df_full = df
df = df_full.drop_duplicates('clean_text')
df.shape
(54055, 61)
# dropping duplicated addresses gives us 10320 listings for the LDA model
57946 - 34206
# check for other dupes
df_full.shape

df = preprocess.clean_duplicates(df, text_col='clean_text', method = 'lsh', char_ngram=5, thresh=.5)
df.shape
(32816, 61)
32816 - 17523
df.to_csv('data/lda_processed5_17.csv')
df_full = pd.read_csv('data/train_processed5_17.csv',index_col=0,dtype = {'GEOID10':object,'blockid':object})
df = pd.read_csv('data/no_dupes_lda_fit5_18.csv',index_col=0,dtype = {'GEOID10':object,'blockid':object})

""" Very Robust Dupe Routine That Uses Repeated LDA Sorting"""
total_dupes = 0
cycle_dupes = 200
n_topics = 30
n_passes = 1
model_runs = 0
df_test = df
df_test = df_test.drop(list(range(n_topics)),1)
while cycle_dupes>50:
    start = len(df_test)
    # fit LDA model
    corpus, dictionary = preprocess.df_to_corpus([str(x) for x in df_test.clean_text], stopwords=hood_stopwords)
    model = models.LdaModel(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0)
    lda_corpus = model[corpus]
    #make dense numpy array of the topic proportions for each document
    doc_topic_matrix = matutils.corpus2dense(lda_corpus, num_terms=n_topics).transpose()
    df_test = df_test.reset_index(drop=True).join(pd.DataFrame(doc_topic_matrix))
    # clean duplicates
    df_test = preprocess.post_lda_drop(df_test, n_topics, n_texts=50, char_ngram=5, thresh=.5, continuous=True, start = 0)
    df_test = preprocess.post_lda_drop(df_test, n_topics, thresh=.5, n_texts=50, slice_at=100, continuous=True)
    # rinse
    cycle_dupes = start - len(df_test)
    total_dupes += cycle_dupes
    df_test = df_test.drop(list(range(n_topics)),1)
    model_runs += 1
    print("Ran ", model_runs, "LDA models and dropped an additional ",cycle_dupes, " for a total of ", total_dupes)
    # repeat

df_test.shape
(14382, 61)
df = df_test
df.to_csv('data/no_dupes_5_25.csv')
df =  pd.read_csv('data/no_dupes_5_25.csv', index_col=0,dtype = {'GEOID10':object,'blockid':object, 'postid':object})


"""Make Corpus and Dictionary"""
with open('resources/hoods.txt') as f:
    neighborhoods = f.read().splitlines()
from sklearn.feature_extraction import stop_words
# we add a long list of seattle neighborhoods to our stop words lis
hood_stopwords = neighborhoods + list(stop_words.ENGLISH_STOP_WORDS)
corpus, dictionary = preprocess.df_to_corpus([str(x) for x in df.clean_text], stopwords=hood_stopwords)

"""Run Lda Model"""
n_topics = 10
n_passes = 50
#Run this if you have access to more than one core set workers=n_cores-1
model = models.ldamulticore.LdaMulticore(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0, workers=3)
#otherwise run this
model = models.LdaModel(corpus, id2word = dictionary, num_topics=n_topics, passes = n_passes, iterations = 100, minimum_probability=0)
#save the model for future use
now = datetime.now()
model.save('models/model'+str(now.month)+'_'+str(now.day))
model = models.LdaModel.load('models/model5_18')
model.log_perplexity(corpus)

"""Merge LDA output and DF"""
#Make LDA corpus of our Data
lda_corpus = model[corpus]
#make dense numpy array of the topic proportions for each document
doc_topic_matrix = matutils.corpus2dense(lda_corpus, num_terms=n_topics).transpose()
df = df.reset_index(drop=True).join(pd.DataFrame(doc_topic_matrix))
# Make top_topic for each document
df = df.assign(top_topic=df[list(range(n_topics))].idxmax(1))
df.to_csv('data/no_dupes_lda_fit5_26.csv')
df = pd.read_csv('data/no_dupes_lda_fit5_18.csv', index_col=0,dtype = {'GEOID10':object,'blockid':object, 'postid':object})
df = df.drop([str(x) for x in list(range(30))],1)


#do the same process for the full_df
full_corpus, dictionary = preprocess.df_to_corpus([str(x) for x in df_full.clean_text], stopwords=hood_stopwords, dictionary=dictionary)
full_lda = model[full_corpus]
full_doc_topic_matrix = matutils.corpus2dense(full_lda, num_terms=n_topics).transpose()
df_full = df_full.reset_index(drop=True).join(pd.DataFrame(full_doc_topic_matrix))
df_full = df_full.assign(top_topic=df_full[list(range(n_topics))].idxmax(1))
df_full.shape
df= df.drop(list(range(30)),1)
df= df.drop([str(x) for x in list(range(30))],1)
len(full_corpus)

df_full.to_csv('data/full_model_fit5_21.csv')

"""Look at the distributions of the different topics"""
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(df[[17]].sort_values(by=17, ascending=False).head(500).values)
plt.hist(df[df.high_black==1][[17]].sort_values(by=17, ascending=False).head(500).values)
plt.hist(df[df.high_black==0][[17]].sort_values(by=17, ascending=False).head(500).values)

plt.hist(df.groupby('GEOID10').mean()[[4]].dropna().values)
df.columns
df.groupby('race_income_label').mean()[26]
plt.hist(df[df.race_income==0][12].values)
df[df.clean_text.str.contains(' bear ')].body_text.values

plt.axis([.1, 1, 0, 500])
plt.hist(df.groupby('GEOID10').mean()[['income']].dropna().values)


df[df['4']>.01][[str(x) for x in list(range(30))]].idxmax(1).value_counts()

"""Use stratifier to create various comparisions of topic distributions"""
strat_col = 'high_white'
lda_output.compare_topics_distribution(df, n_topics, strat_col)
df = df.rename(dict(zip([str(x) for x in range(1,13)], range(1,13))), axis=1)
mean_diff = lda_output.summarize_on_stratifier(df, 12, 'high_white', topic_cols = list(range(1,13)))
mean_diff = mean_diff.sort_values('difference', ascending=False)
mean_diff

sorted_topics = pd.Series(range(n_topics))
df.groupby('top_topic').count()
"""Try a Multinomial LogisticRegression"""
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Ridge
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.regression.linear_model import OLS
# we use the full df for the regression because we want to weight results by the
# existence of different ads in different neighborhoods, not just unique addresses
X = df[["black_proportion","log_income","asian_proportion","latinx_proportion","log_price"]]
y = df.white_proportion
df_tmp = df.copy()
df_tmp[list(range(30))] = df_tmp[list(range(30))].where(df_tmp[list(range(30))]>.1,0)
topic_0 + topic_7 + topic_8 + topic_9 + topic_12  + topic_14 + topic_16 + topic_17+ topic_20 + topic_23 + topic_24 + topic_25  + topic_28
X = df[[str(x) for x in [0,7,8,9,12,14,16,17,20,23,24,25,28]]+["black_proportion","log_income","log_price","total_RE"]]
y = np.where(df['white_proportion']>np.median(df['white_proportion']),1,0)
y= df['income']
OLR = OLS(y,X).fit()
OLR.summary()
OLR.predict(exog=X)

df_full_results.params.sort_values()
df_results.params.sort_values()
df_results.summary()
EN = ElasticNet(alpha = .02, l1_ratio=.001)
EN.fit(X,y)
EN.score(X,y)
EN.predict(X)
LinR = LinearRegression()
LinR.fit(X,y)
LinR.score(X,y)

RR = Ridge()
RR.fit(X,y).score(X,y)
pd.Series(RR.coef_)
from sklearn.svm import SVR, SVC
supportR = SVR()
supportR.fit(X,y)

supportC = SVC()
supportC.fit(X,y)
supportC.score(X,y)

from sklearn.metrics import explained_variance_score, r2_score
explained_variance_score(y, OLR.predict(X))
r2_score(y, RR.predict(X))
predict_df = pd.DataFrame({'predictions':OLR.predict(X), 'real_values':y, 'tractID':df.GEOID10})
predict_df = predict_df.assign(abs_diff = abs(predict_df.predictions - predict_df.real_values))
predict_df.sum()

predict_df.groupby('tractID').mean()
X = df[["black_proportion","log_income","asian_proportion","latinx_proportion","log_price"]]
y = df.top_topic
LR = LogisticRegression()
LR.fit(X,y)
LR.score(X,y)
lr_coefs = pd.DataFrame(LR.coef_).rename({0:"black_proportion",1:"log_income",2:"asian_proportion",3:"latinx_proportion",4:"log_price"}, axis=1)
lr_coefs
lr_coefs = lr_coefs.assign(abs_black = abs(lr_coefs.black_proportion)).sort_values("abs_black", ascending=False)
mean_diff = lr_coefs.merge(lda_output.summarize_on_stratifier(df, n_topics, 'high_black'), left_index=True, right_index=True)
mean_diff.drop(['abs_black','difference', 'proportion'], axis=1, inplace=True)
mean_diff

""" Mess Around with Machine Learning """
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train, y_train)
predictions = rf.predict_proba(X_test)[:,1]

rf.score(X_test, y_test)


sorted_import_index = rf.feature_importances_.argsort()

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
df = pd.read_csv('data/stm_twelve_5_29.csv',dtype = {'GEOID10':object, 'postid':object})
df[list(range(n_topics))]
mean_diff = list(range(n_topics))
sorted_topics = mean_diff
now = datetime.now()
reload(lda_output)
lda_output.text_output(df, text_col='body_text', filepath='output/cl'+str(now.month)+'_'+str(now.day)+'jitter.txt', model= model, sorted_topics=sorted_topics, cl=True, print_it = False, sample_topics=30, sample_texts=10, jitter=False)
lda_output.multi_file_output(df, text_col='body_text', filepath='output/test', sorted_topics=mean_diff, sample_topics=12, sample_texts=10, topics=mean_diff)

lda_output.get_formatted_topic_list(model, formatting="summary", n_topics=-1, n_terms=20)

df[df.postid=='6436256628']
