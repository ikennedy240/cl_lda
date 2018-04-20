"""
This Module includes functions for processing and cleaning scraped craigslist
data to prepare for text analysis using LDA or other NLP analysis
"""
import pandas as pd
import logging
from gensim import corpora, models
from six import iteritems

module_logger = logging.getLogger('cl_lda.preprocess')

# given a folder of scraped CL data, returns a datafame of combined listings with duplicates


#Get neighborhoods
def make_neighborhood_list(hoodseries, save=True):
    #clean up some basic stuff in hoods
    hoods = pd.Series(hoodseries.dropna()).str.replace(r'\(|\)','').str.lower().str.replace(', ?wa$','').str.replace(r'.*\d.*','')
    #split the hoods on commas and slashes and then add those to the list and count frequency
    hoods = hoods[hoods!=''].str.split(r',|/|-|&|#|\|', expand=True).stack().str.strip().reset_index(drop=True)
    #check the length
    hoods = pd.concat([hoods, hoods.str.len()], axis = 1)
    #clear those with less than 3 chars and more than 25 chars
    hoods = hoods[hoods[1]>3]
    hoods = hoods[hoods[1]<25]
    #only inluce hoods that show up more than 3 times
    hoods = list(hoods[0].value_counts()[hoods[0].value_counts()>1].index)
    #writes to file
    if save:
        with open('resources/hoods.txt', 'w') as f:
            [f.write(x+', ') for x in hoods]
    #returns list of neighborhood names with counts
    return hoods


#Clean body text
def cl_prep_for_lda(text_series):
    text_series = text_series.str.replace("QR Code Link to This Post", '').str.replace(r'\n|\r|\t','')
    text_series = text_series.str.replace(r'\S*(\.com|\.net|\.gov|\.be|\.org)\S*',' #URL ').str.replace(r'http\S*', ' #URL ').str.replace(r'\d+', ' #NUMBER ')
    text_series = text_series.str.replace(r'^,+','').str.replace(r',,+','').str.strip()
    text_series = text_series.str.replace(r'!!!!+',' shii ').str.replace(r'!!!', ' mitsu ').str.replace(r'!!', ' nii ').str.replace(r'!', ' ichi ')
    text_series = text_series.str.replace('\$\$+',' dollasigns ').str.replace('\$', ' dollasign ').str.strip()
    return text_series

#take out neighborhood Names
def clean_neighborhoods(text_df, neighborhoods, text_col='body_text'):
    for name in neighborhoods:
        text_df[text_col] = text_df[text_col].str.replace(r' ?'+name+' ?', " #HOOD ", case=False)
        if neighborhoods.index(name) % 100 == 0:
            module_logger.info("Replaced "+str(neighborhoods.index(name))+" neighborhoods")

#Deal with duplicates
def clean_duplicates(text_df, text_col='body_text',method = 100):
    if method == 'latlon':
        #must have 'latitude', 'longitude','price' colums to use this method
        text_df = text_df.drop_duplicates(['latitude','longitude','price'])
    if type(method)==int:
        text_df['body_100']=text_df[text_col].str.slice(stop=method).copy()
        text_df = text_df.body_100.unique()
    return text_df

# make a corpus and dictionary from a list of texts
def df_to_corpus(documents, stopwords=None):
    if not(stopwords):
        from sklearn.feature_extraction import stop_words
        stopwords = stop_words.ENGLISH_STOP_WORDS
    #turns each tweet into a list of words
    texts = [[word for word in document.lower().split() if word not in stopwords] for document in documents]
    #makes a dictionary based on those texts (this is the full df) and saves it
    dictionary = corpora.Dictionary(texts)
    #applies a bag of words vectorization to make the texts into a sparse matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    return(corpus, dictionary)

# make a corpus that doesn't hold everythin in memory
class MyMemoryCorpus(object):
    def __init__(self, text_file, dictionary=None, stopwords=None):
        self.file_name = text_file
        if dictionary is None:
            self.prepare_dictionary()
        else:
            self.dictionary = dictionary
        #Checks if a dictionary has been given as a parameter.
        #If no dictionary has been given, it creates one and saves it in the disk.
    def __iter__(self):
        for line in open(self.file_name, encoding = 'utf-8'):
            # assume there's one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(line.lower().split())

    def prepare_dictionary(self):
        from sklearn.feature_extraction import stop_words
        # List of stop words which can also be loaded from a file.
        stop_list = stop_words.ENGLISH_STOP_WORDS

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


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    df = pd.read_csv("data/immi_rated.csv")
    seattle = pd.read_csv('data/seattle_cleaned')
    df = df.iloc[0:20000]
    df.text = df.text.str.replace('\n','')
    df['text'].to_csv('data/immi_text.txt', sep=' ', index=False, header=False)
    #test df_to_corpus
    df_to_corpus(list(df.text))
    #test clean_duplicates
    seattle.shape
    clean_duplicates(seattle, text_col='body_text', method = 1000).shape
    clean_neighborhoods(seattle, ["ballard", "sand point"])
    cl_prep_for_lda(df.text)
    seattle.body_text.sample(10)
    seattle.columns
    make_neighborhood_list(seat.neighborhood)
    seat = pd.read_csv('data/seattle4_14.csv')
    seat.neighborhood
