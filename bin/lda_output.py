"""
This module includes some helper functions to produce useful output from
lda models trained with gensim
"""

from gensim import corpora, models, matutils
import regex as re
import pandas as pd
import numpy as np
import logging
import sys
import codecs

module_logger = logging.getLogger('cl_lda.lda_output')

# makes a nicely formatted list of topics given an LDA model
def get_formatted_topic_list(model, formatting="summary", n_topics=-1, n_terms=20):
    if formatting == "summary":
        topics = ["Topic "+str(x[0])+": "+re.sub(r'" \+ \d.\d+\*"|\d.\d+\*"',', ',x[1])[2:-1] for x in model.print_topics(n_topics,n_terms)]
    if formatting == "keywords":
        topics = ["Top keywords are: "+re.sub(r'" \+ \d.\d+\*"|\d.\d+\*"',', ',x[1])[2:-1] for x in model.print_topics(n_topics,n_terms)]
    if formatting == "blank":
        topics = [re.sub(r'" \+ \d.\d+\*"|\d.\d+\*"',', ',x[1])[2:-1] for x in model.print_topics(n_topics,n_terms)]
    return(topics)

#this function takes a df that has the texts, the toipc distributions, and a binary stratifieer
#if topic_cols isn't specified, defaults to list(range(n_topics))
def summarize_on_stratifier(df_merged, n_topics, strat_col, topic_cols=None, thresh = 0.01, method="mean"):
    # given some stratifier, compare the topic distributions
    #Count the occurence of each topic by high_white
    if topic_cols is None:
        topic_cols = list(range(n_topics))
    # make a limted df of just the topic_cols and the stratifier
    try:
        #topic cols are usually numeric if it comes from the LDA
        no_text = df_merged[topic_cols].join(df_merged[strat_col])
    except:
        #but sometimes they're not, if the df was saved and then reloaded
        no_text = df_merged[[str(x) for x in topic_cols]].rename(int,axis='columns').join(df_merged[strat_col])
    # makes the distributions binary based on the threshold, 0.01 by default
    # values less than the threshold are set to 0, values higher are set to one
    # as long as they are positive!
    no_text[no_text<thresh] = 0
    no_text = np.sign(no_text)
    #I've run this with means too, but I'll eventually want to do a chi-squared test
    #so I think counts are better
    if method == "mean":
        all = no_text.mean()
        high = no_text[no_text[strat_col]==1].mean()
        low = no_text[no_text[strat_col]==0].mean()
    if method == "sum":
        all = no_text.sum()
        high = no_text[stratifier==1].sum()
        low = no_text[stratifier==0].sum()
        strat_col='high_white'
    columns = {0:'all_r', 1:strat_col, 2:'not_'+strat_col}
    #make a df of that data
    mean_diff = pd.DataFrame(data=[all, high, low]).transpose().rename(columns = columns).drop(strat_col)
    #calculate the absolute value of the difference between the two categories)
    mean_diff = mean_diff.assign(difference = abs(mean_diff[columns[1]] - mean_diff[columns[2]]), proportion = mean_diff[columns[1]]/mean_diff[columns[2]], topic = mean_diff.index)
    return mean_diff

def compare_topics_distribution(df_merged, n_topics, strat_col, topic_cols=None, thresh = 0.01, method="mean"):
    mean_diff = summarize_on_stratifier(df_merged, n_topics, strat_col, topic_cols, thresh, method)
    columns = {0:'all_r', 1:strat_col, 2:'not_'+strat_col}
    topic_comparison = pd.DataFrame(data=[mean_diff[columns[0]].sort_values(ascending=False).index, mean_diff[columns[1]].sort_values(ascending=False).index, mean_diff[columns[2]].sort_values(ascending=False).index]).transpose().rename(columns = columns)
    return topic_comparison

def rfc_distribution(df_merged, strat_col, n_topics=None, topic_cols=None, thresh = 0.01, method="mean", return_model = False):

        # given some stratifier, fit an RFC model to identify topic importance
        if topic_cols is None:
            try:
                topic_cols = list(range(n_topics))
            except:
                raise ValueError("You must supply either topic_cols or n_topics")
        #do a quick Random Forest Classification to see which topics are most useful for distinguishing
        #beetween high and low white neighborhoods
        #SKIP DOWN AND USE top_ten_mean TO AVOID SKLEARN
        from sklearn.model_selection import train_test_split
        try:
            #topic cols are usually numeric if it comes from the LDA
            X_train, X_test, y_train, y_test = train_test_split(df_merged[topic_cols], df_merged[strat_col], random_state=0)
        except:
            #but sometimes they're not, if the df was saved and then reloaded
            X_train, X_test, y_train, y_test = train_test_split(df_merged[[str(x) for x in topic_cols]], df_merged[strat_col], random_state=0)
        from sklearn.ensemble import RandomForestClassifier
        try:
            rf = RandomForestClassifier(n_estimators = 1000, n_jobs = 3).fit(X_train,y_train)
        except:
            rf = RandomForestClassifier(n_estimators = 1000).fit(X_train,y_train)
        predictions = rf.predict_proba(X_test)[:,1]
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_test, predictions)
        score = rf.score(X_test, y_test)
        #make a summary dataframe of the relative counts for the 10 most important topics
        #this one is orders them based off the RFC (ie these topics are good for sorting
        #based on their proportion in a document)
        mean_diff = summarize_on_stratifier(df_merged, n_topics, strat_col, topic_cols, thresh, method)
        sorted_rfc = mean_diff.iloc[list(rf.feature_importances_.argsort())]
        sorted_rfc['rfc'] = range(1,n_topics+1)
        if return_model:
            return rf, sorted_rfc
        return sorted_rfc




# helper function to print topics and example texts
# sorted topics must have the
#need to make some changes here
def text_output(df, text_col, filepath, sample_topics=10, sample_texts=5, sorted_topics=None, strat_col=None, topics=None, model=None, print_it = False, cl = False, jitter=False):
    if topics is None:
        if model is None:
            return "You must provide either model or topics"
        else:
            module_logger.info("Automatically generating topic list")
            topics = get_formatted_topic_list(model, formatting='keywords', n_topics=-1)
    n_topics = len(topics)
    try:
        df[1]
    except:
        df = df.rename(dict(zip([str(x) for x in range(n_topics)], range(n_topics))), axis=1)
    if sorted_topics is None:
        if strat_col is None:
            return "You must provide either sorted_topics or strat_col"
        # if no sorted_topics are provided, they're calculated with default values
        else:
            module_logger.info("Using default paramaters to produce topics sorted by stratifier")
            mean_diff = summarize_on_stratifier(df, n_topics, strat_col)
            sorted_topics = mean_diff.sort_values('proportion', ascending=False)[1:].proportion
    if print_it:
        print("sorted_topics.index",sorted_topics.index)
        for j in sorted_topics.index:
            print("Topic #", j,' occurred in \n', sorted_topics.loc[j], '\n', topics[int(j)])
        print("\n ------- Sample Documents ------- \n\n")
        for j in sorted_topics.index[0:sample_topics]:
            print("Topic #", j,' occurred in \n', sorted_topics.loc[j], '\n', topics[int(j)])
            print("\n Top ",sample_texts," texts fitting topic", j, "are: \n \n")
            if jitter:
                tmp_top = df.sort_values(by=j, ascending=False).head(100).sample(10)
            else:
                tmp_top = df.sort_values(by=j, ascending=False)
            for i in range(sample_texts):
                tmp = tmp_top.iloc[i]
                print("Topic", j, "Example #", i+1, ':\n')
                if strat_col:
                    print("This text rated a ", tmp[strat_col], " in ", strat_col, '\n')
                if cl:
                    print("This text had post id:", tmp.postid, " was connected to a listing in tract #", tmp.GEOID10,'and cost $', tmp.clean_price, '\n')
                print("was ",round(tmp.loc[j]*100,2),"percent topic", j, ':\n', tmp[text_col], '\n')
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            for j in sorted_topics.index:
                print("Topic #", j,' occurred in \n', sorted_topics.loc[j], '\n', topics[int(j)], file=f)
            print("\n ------- Sample Documents ------- \n\n", file=f)
            for j in sorted_topics.index[0:sample_topics]:
                print("Topic #", j,' occurred in \n', sorted_topics.loc[j], '\n', topics[int(j)], file=f)
                print("\n Top ",sample_texts," texts fitting topic", j, "are: \n \n", file=f)
                if jitter:
                    tmp_top = df.sort_values(by=j, ascending=False).head(100).sample(10)
                else:
                    tmp_top = df.sort_values(by=j, ascending=False)
                for i in range(sample_texts):
                    tmp = tmp_top.iloc[i]
                    print("Topic", j, "Example #", i+1, ':\n', file=f)
                    if strat_col:
                        print("This text rated a ", tmp[strat_col], " in ", strat_col, '\n', file=f)
                    if cl:
                        print("This text had address:", tmp.address, " was connected to a listing in tract #", tmp.GEOID10,'and cost $', tmp.clean_price, '\n', file=f)
                    print("was ",round(tmp.loc[j]*100,2),"percent topic", j, ':\n', tmp[text_col], '\n', file=f)
        module_logger.info("Saved output to "+filepath)
    module_logger.info("Completed Output")

# an output function which makes individual files
def multi_file_output(df, text_col, filepath, sample_topics=10, sample_texts=5, sorted_topics=None, strat_col=None, topics=None, model=None):
    if topics is None:
        if model is None:
            return "You must provide either model or topics"
        else:
            module_logger.info("Automatically generating topic list")
            topics = get_formatted_topic_list(model, formatting='keywords', n_topics=-1)
    n_topics = len(topics)
    try:
        df[1]
    except:
        df = df.rename(dict(zip([str(x) for x in range(n_topics)], range(n_topics))), axis=1)
    if sorted_topics is None:
        if strat_col is None:
            return "You must provide either sorted_topics or strat_col"
        # if no sorted_topics are provided, they're calculated with default values
        else:
            module_logger.info("Using default paramaters to produce topics sorted by stratifier")
            mean_diff = summarize_on_stratifier(df, n_topics, strat_col)
            sorted_topics = mean_diff.sort_values('proportion', ascending=False)[1:].proportion
    # What we need to do is make a separate file with meta-data for each files
    # Meta data should be: topic, topic proportion, postID, price
    for j in sorted_topics.index[0:sample_topics]:
        tmp_top = df.sort_values(by=j, ascending=False)
        for i in range(sample_texts):
            tmp = tmp_top.iloc[i]
            filename = "topic_"+str(j)+"_prop_"+str(round(tmp[j],2))+"_postid_"+str(tmp.postid)+"_price_"+str(tmp.clean_price)+".txt"
            #module_logger.info("Trying"+str(filename))
            with open(filepath+'/'+filename, 'w', encoding='utf-8') as f:
                print("Topic", j, "Example #", i+1, ':\n', file=f)
                print("This text had address:", tmp.address, " was connected to a listing in tract #", tmp.GEOID10,' with post ID ',tmp.postid, ' and cost $', tmp.clean_price, '\n', file=f)
                print("was ",round(tmp.loc[j]*100,2),"percent topic", j, ':\n', tmp[text_col], '\n', file=f)
    module_logger.info("Saved output to "+filepath)



if __name__ == "__main__":
    df = pd.read_csv('data/no_dupes_lda_fit5_18.csv', index_col=0,dtype = {'GEOID10':object,'blockid':object, 'postid':object})
    model = models.LdaModel.load('models/model5_18')
    df.clean_price
