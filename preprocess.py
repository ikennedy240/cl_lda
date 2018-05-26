"""
This Module includes functions for processing and cleaning scraped craigslist
data to prepare for text analysis using LDA or other NLP analysis
"""
import pandas as pd
import logging
from gensim import corpora, models
from six import iteritems
import re
import numpy as np
import itertools
from lsh import cache, minhash
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
            f.write('\n'.join(hoods))
    #returns list of neighborhood names with counts
    return hoods


#Clean body text
def cl_prep_for_lda(text_series, neighborhoods=None):
    if neighborhoods is None:
        module_logger.info("No neighborhoods provided, using 'resources/hoods.txt'")
        with open('resources/hoods.txt', 'r') as f:
            neighborhoods= f.read().splitlines()
    module_logger.info("Cleaning urls and multiple commas")
    clean_text = cl_clean_text(text_series)
    module_logger.info("Cleaning neighborhoods")
    clean_text = clean_neighborhoods(clean_text, neighborhoods)
    module_logger.info("Success: returning cleaned text")
    return clean_text

#remove unwanted signs
def cl_clean_text(text_series, clean_punct=True, body_mode=False):
    if body_mode:
        text_series = text_series.str.replace("QR Code Link to This Post", '').str.replace(r'\n|\r|\t',' ').str.replace(r',,+',',').str.replace(r'^,+','').str.replace(r' +',' ').str.strip()
        return text_series
    text_series = text_series.str.replace("QR Code Link to This Post", '').str.replace(r'\n|\r|\t','')
    text_series = text_series.str.replace(r'\S*(\.com|\.net|\.gov|\.be|\.org)\S*',' #URL ').str.replace(r'http\S*', ' #URL ').str.replace(r'\d+', ' ')
    text_series = text_series.str.replace(r'^,+','').str.replace(r',,+',' ')
    if clean_punct:
        module_logger.info("Cleaning punctuation, but retaining exclamations and dollarsigns")
        # We want to clean all non-word characters, but we think certain punctuation might be important, so this retains combinations of $ and !
        text_series = text_series.str.replace(r'!!!!+',' shii ').str.replace(r'!!!', ' mitsu ').str.replace(r'!!', ' nii ').str.replace(r'!', ' ichi ')
        text_series = text_series.str.replace(r'\$\$+',' dollasigns ').str.replace(r'\$', ' dollasign ').str.strip()
        text_series = text_series.str.replace(r'[^\w\s]+','')
        # later on we'll drop all tokens with less than 4 characters, so this makes the interesting punctuation have more
        text_series = text_series.str.replace('shii', '!!!!_').str.replace('mitsu', '!!!_').str.replace('nii', '!!__').str.replace('ichi', '!___').str.replace('dollasigns', '$$__').str.replace('dollasign', '$___').str.strip()
    text_series = text_series.str.replace(r' +',' ').str.strip()
    return text_series

def preprocess(text):
    with open('resources/seattle_stop_words.txt') as f:
        neighborhoods = f.read().splitlines() # read stopwrod file
    from sklearn.feature_extraction import stop_words # get sklearn stop_words
    stop_words = neighborhoods + list(stop_words.ENGLISH_STOP_WORDS) # join our stopwords and the sklearn ones
    stopword_pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*', flags=re.IGNORECASE) # turn the stopwords into a compiled regex
    punctuation_pattern = r"[#\w]+|[!?]+" # compile a regex that saves certain characters
    url_pattern = r'(http)?(www)?\S*(\.com|\.net|\.gov|\.be|\.org)\S*'
    short_pattern = r"(^| |')\b\w{1,3}\b"
    text = (text.str.lower() # make lowercase
           .str.replace(stopword_pattern, '') # drop neighborhoods and other stopwords
           .str.replace(url_pattern, '') # drop urls
           .str.replace(r'\d+', '') # drop digits
           .str.findall(punctuation_pattern) # drop most punctuation
           .str.join(' ') # join after punctuation drop
           .str.replace(short_pattern, '')) # drop words with less than 3 characters
    return text


#take out neighborhood Names
def clean_neighborhoods(text_series, neighborhoods=None):
    if neighborhoods is none:
        with open('resources/hoods.txt') as f:
            neighborhoods = f.read().splitlines()
    for name in neighborhoods:
        text_series = text_series.str.replace(r' ?'+name+' ?', " #HOOD ", case=False)
        if neighborhoods.index(name) % 100 == 0:
            module_logger.info("Replaced "+str(neighborhoods.index(name))+" neighborhoods")
    return text_series

# helper function to make shingles from a text
def shingles(text, char_ngram=5):
    return set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram))

# calculate jaccard similarity between two lists of shingles
def jaccard(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)

def jaccard_sim(line_a, line_b, char_ngram=5):
    shingles_a = shingles(line_a, char_ngram)
    shingles_b = shingles(line_b, char_ngram)
    return jaccard(shingles_a, shingles_b)

def sims_all(lines, n_texts, char_ngram=5):
    sims_all = np.zeros((n_texts, n_texts), dtype=np.float64)
    for i, line in enumerate(lines):
        for j in range(i+1, len(lines)):
            sims_all[i, j] = jaccard_sim(lines[i], lines[j], char_ngram)
    return sims_all

# make a list of possible duplicates
def candidate_duplicates(text_df, text_col, char_ngram=5, seeds=100, bands=5, hashbytes=4):
    hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, hashbytes=hashbytes)
    if seeds % bands != 0:
        raise ValueError('Seeds has to be a multiple of bands. {} % {} != 0'.format(seeds, bands))
    lshcache = cache.Cache(num_bands=bands, hasher=hasher)
    for i in range(len(text_df)):
        line = text_df[text_col].iloc[i]
        lshcache.add_fingerprint(hasher.fingerprint(line),doc_id = i)
    candidate_pairs = set()
    for b in lshcache.bins:
        for bucket_id in b:
            if len(b[bucket_id]) > 1:
                pairs_ = set(itertools.combinations(b[bucket_id], r=2))
                candidate_pairs.update(pairs_)
    return candidate_pairs

#Deal with duplicates
def clean_duplicates(text_df, text_col='body_text',method = 100,char_ngram=5, seeds=100, bands=5, hashbytes=4, thresh = .9):
    start = len(text_df)
    if method == 'latlon':
        #must have 'latitude', 'longitude','price' colums to use this method
        text_df = text_df.drop_duplicates(['latitude','longitude','price'])
    if type(method)==int:
        text_df['body_100']=text_df[text_col].str.slice(stop=method).copy()
        text_df = text_df.drop_duplicates(subset = 'body_100').drop('body_100', axis = 1)
    #use LSH minhash to clean dupes requires LSH and minhash3 to run!!!
    if method=='lsh':
        try:
            text_df = text_df.sort_values('listingDate').reset_index()
        except(KeyError):
            try:
                module_logger.info('text_df has insufficient date information, dropping by month/day')
                text_df=text_df.sort_values(['scraped_month','scraped_day']).reset_index()
            except:
                module_logger.info('text_df has insufficient date information, dropping by index')
                text_df=text_df.reset_index()
        # first get candidate_pairs
        candidate_pairs = candidate_duplicates(text_df, text_col, char_ngram, seeds, bands, hashbytes)
        module_logger.info("Found "+str(len(candidate_pairs))+" possible duplicate pairs")
        # then we make sure jaccard similarity is above .9
        lines = text_df[text_col].str.lower().values
        hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, hashbytes=hashbytes)
        lshcache = cache.Cache(bands=bands, hasher=hasher)
        similarities = []
        for (line_a, line_b) in candidate_pairs:
            doc_a, doc_b = lines[line_a], lines[line_b]
            similarities.append((line_a, line_b, jaccard_sim(doc_a, doc_b, char_ngram)))
            if len(similarities) % 10000 == 0:
                module_logger.info("Processed "+str(len(similarities))+" possible duplicates")
        # reduce to only jaccards above .9 and check which pair is older
        drop_list = [min(pair[0],pair[1]) for pair in similarities if pair[2]>=thresh]
        text_df = text_df.drop(set(drop_list)).set_index("index")
        dupes = start - len(text_df)
        module_logger.info("Dropped "+str(dupes)+" duplicates")
    return text_df

def post_lda_drop(df_test, n_topics, text_col='clean_text', n_texts=40, char_ngram=5, thresh=.9, continuous=False, return_max=False, slice_at=None, start=0):
    if continuous:
        dropped = n_texts+1
        while dropped>n_texts:
            dropped = len(df_test)
            df_test, max_list = post_lda_drop(df_test, n_topics, text_col, n_texts, char_ngram, thresh, continuous=False, return_max=True, slice_at=slice_at, start=start)
            start+= n_texts - max(max_list) - 1
            dropped = dropped - len(df_test)
            print(dropped, start)
        return df_test
    max_list = []
    for i in range(n_topics):
        tmp_top = df_test.sort_values(by=i, ascending=False).iloc[start:n_texts+start] # grab the top listings for the topic
        if slice_at is None:
            sims = sims_all(tmp_top[text_col].values, n_texts, char_ngram) # calculate the jaccard similarity between all of them
        else:
            sims = sims_all(tmp_top[text_col].str.slice(stop=slice_at).values, n_texts, char_ngram) # calculate the jaccard similarity between all of them
        x = np.transpose((sims>thresh).nonzero()) # find those texts where the similarity is above a threshold
        pairs = [(tmp_top.iloc[pair[0]].name, tmp_top.iloc[pair[1]].name) for pair in x]
        drop_list = list(set([min(pair[0],pair[1]) for pair in pairs])) # make a unique list of those, keeping only the newest dupe
        module_logger.info("dropped " + str(len(drop_list))+" listings from topic "+str(i)+'\n') # output some info about the drop
        max_list.append(len(drop_list))
        df_test = df_test.drop(drop_list) #drop them from the test set
    if return_max:
        return df_test, max_list
    return df_test

# make a corpus and dictionary from a list of texts
def df_to_corpus(documents, stopwords=None, dictionary=None):
    if stopwords is None:
        from sklearn.feature_extraction import stop_words
        stopwords = stop_words.ENGLISH_STOP_WORDS
    #compiles a regex of the stopwords (this is to include mulit-word stopwords)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
    # cleans stopwords and drops words with fewer than three characters
    texts = [[word for word in pattern.sub('', document.lower()).split() if len(word)>3] for document in documents]
    #makes a dictionary based on those texts (this is the full df) and saves it
    if dictionary is None:
        dictionary = corpora.Dictionary(texts)
    #applies a bag of words vectorization to make the texts into a sparse matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    return(corpus, dictionary)

# make a corpus that doesn't hold everythin in memory
class MyMemoryCorpus(object):
    def __init__(self, text_file, dictionary=None, stopwords=None):
        self.file_name = text_file
        #Checks if a dictionary has been given as a parameter.
        #If no dictionary has been given, it creates one and saves it in the disk.
        if dictionary is None:
            self.prepare_dictionary()
        else:
            self.dictionary = dictionary

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

#make a binary variable in a new column which breaks down based on a given threshold
#defaults to > the median of strat_col
def make_stratifier(df, strat_col, new_col, thresh=None):
    df = df.copy()
    import numpy as np
    #check if we got lists for both columns
    if type(strat_col) is list and type(new_col) is list:
        #throw an error if lenghts don't match
        if len(strat_col)!=len(new_col):
            raise ValueError("strat_col and new_col must be the same size")
        #loop through list of cols and run the function for each pair
        else:
            for i in range(len(strat_col)):
                df = make_stratifier(df, strat_col[i], new_col[i], thresh)
            return df
    # if no threshold is given, use the median of strat_col
    if thresh is None:
        thresh = df[strat_col].median()
    #make a new binary column that's one above a threshold of the old column and zero below it
    df[new_col]=np.where(df[strat_col]>thresh, 1, 0)
    return df

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    new_df = pd.read_csv('data/seattle_4_22.csv', index_col=0, dtype = {'GEOID10':object,'blockid':object})
    new_df['clean_text'] = cl_clean_text(new_df.body_text, clean_punct=True, body_mode=False)
    with open('resources/hoods.txt') as f:
        neighborhoods = f.read().splitlines()
    from sklearn.feature_extraction import stop_words
    hood_stopwords = neighborhoods + list(stop_words.ENGLISH_STOP_WORDS)
    corpus, dictionary = df_to_corpus(["11111 !!!!!! ????? words things alphabets"], stopwords=hood_stopwords)
    print([dictionary[i] for i in range(121)])
    documents = [str(x) for x in new_df.clean_text]
    documents[20]
    char_ngram = 5
    text_col='clean_text'
    text_df2 = pd.read_csv('data/5_13_all_vars.csv', index_col=0, dtype = {'GEOID10':object,'blockid':object})
    len(text_df2)-len(set(drop_list))

    test = clean_duplicates(text_df2, text_col='clean_text', method='lsh')
    test.shape
