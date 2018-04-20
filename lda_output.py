"""
This module includes some helper functions to produce useful output from
lda models trained with gensim
"""

from gensim import corpora, models, matutils
import regex as re


# makes a nicely formatted list of topics given an LDA model
def get_formatted_topic_list(model, formatting, n_topics=10):
    if formatting == "summary":
        topics = ["Topic "+str(x[0])+": "+re.sub(r'\s+', ', ',re.sub(r'\d|\W',' ',x[1]))[2:-2] for x in model.print_topics(n_topics,20)]
    if formatting == "keywords":
        topics = ["Top keywords are: "+re.sub(r'\s+', ', ',re.sub(r'\d|\W',' ',x[1]))[2:-2] for x in model.print_topics(n_topics,20)]
    return(topics)

#this function takes a df that has the texts, the toipc distributions, and the stratifieer
#if topic_cols isn't specified, defaults to list(range(n_topics))
def summarize_on_stratifier(df_merged, n_topics, stratifier, topic_cols=none, thres = 0.01, method="mean"):
    # given some stratifier, compare the topic distributions
    #Count the occurence of each topic by high_white
    if not(topic_cols):
        topic_cols = list(range(n_topics))
    # make a limted df of just the topic_cols and the stratifier
    no_text = df_merged[topic_cols].join(stratifier)
    # makes the distributions binary based on the threshold, 1% by default
    no_text[no_text<.01] = 0
    no_text = np.sign(no_text)
    #I've run this with means too, but I'll eventually want to do a chi-squared test
    #so I think counts are better
    if method == "mean":
        all = no_text.mean()
        high = no_text[stratifier==1].mean()
        low = no_text[stratifier==0].mean()
    if method == "sum":
        all = no_text.sum()
        high = no_text[stratifier==1].sum()
        low = no_text[stratifier==0].sum()
    #make a df of that data
    mean_diff = pd.DataFrame(data=[all, high, low]).transpose().rename(columns = {0:'all_r', 1:'high_white', 2:'low_white'})
    #calculate the absolute value of the difference between the two categories)
    mean_diff['difference']  = abs(mean_diff.high_white - mean_diff.low_white)
    mean_diff['prop'] = abs(mean_diff.high_white/mean_diff.low_white)
    mean_diff.sort_values(by='prop', ascending=False)
    topic_comparison
    sorted_mean = mean_diff.sort_values('prop', ascending=False)[1:]

    topic_comparison = pd.DataFrame(data=[all.drop('high_white').sort_values(ascending=False).index, high.drop('high_white').sort_values(ascending=False).index, low.drop('high_white').sort_values(ascending=False).index]).transpose().rename(columns = {0:'all_r', 1:'high_white', 2:'low_white'})


# given some stratifier, fit an RFC model to identify topic importance
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


# helper function to print topics and example texts

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


if __name__ == "__main__":
    model = models.LdaModel.load('models/4_12model')
