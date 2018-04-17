"""
This Module includes functions for processing and cleaning scraped craigslist
data to prepare for text analysis using LDA or other NLP analysis
"""
import pandas as pd
import logging

module_logger = logging.getLogger('cl_lda.preprocess')

# given a folder of scraped CL data, returns a datafame of combined listings with duplicates


#Get neighborhoods
def makeNeighborhoodList(hoodseries, save=True):
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
def prepForLda(text):
    test = seattlefull.body_text
    test = test.str.replace("QR Code Link to This Post", '').str.replace(r'\n','').str.replace(r'\r',' ').str.strip()
    test = test.str.replace(r'\S*(\.com|\.net|\.gov|\.be|\.org)\S*','#URL').str.replace(r'http\S*', '#URL').str.replace(r'\d+', '#NUMBER')
    test = test.str.replace(r'^,+','').str.replace(r',,+','')str.strip()
    df.body_text = df.body_text.str.replace(r'!!!!+',' shii ').str.replace(r'!!!', ' mitsu ').str.replace(r'!!', ' nii ').str.replace(r'!', ' ichi ')
    df.body_text = df.body_text.str.replace('\$\$+','dollasigns').str.replace('\$', 'dollasign')
    df.body_text = test

#take out neighborhood Names
def cleanNeighborhoods(cltext, neighborhoods):
for name in neighborhoods:
    df['body_text'] = df.body_text.str.replace(r' ?'+name+' ?', " #HOOD ", case=False)
    if hoods.index(name) % 100 == 0:
        print(hoods.index(name))

#Deal with duplicates
def clean_duplicates(cltext):
    cl_full = pd.read_csv('data/full_cl_dump.csv')
    cl_full.shape
    cl_dropped = df.drop_duplicates(['latitude','longitude','price'])
    cl_dropped.shape
    cl_dropped['body_100']=cl_dropped.body_text.str.slice(stop=1000).copy()
    cl_dropped.body_100.unique().shape
    cl_dropped.to_csv('data/cl_dropped.csv')

# make a corpus and dictionary from a list of texts
def df_to_corpus(documents, stopwords=None):
    if stopwords:
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
    def __init__(self, text_file, dictionary=None, stopwords=None:
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
        for line in open(self.file_name, encoding = 'utf-8'):
            # assume there's one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(line.lower().split())

    def prepare_dictionary(self):
        stop_list =   # List of stop words which can also be loaded from a file.

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
    df = pd.read_csv("data/immi_rated.csv")
