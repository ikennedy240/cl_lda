import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/Users/ikennedy/Documents/GitHub/torchMoji')

from __future__ import print_function, unicode_literals
from torchmoji.create_vocab import extend_vocab, VocabBuilder
from torchmoji.word_generator import WordGenerator
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
import numpy as np
import pandas as pd
import csv
import datetime
import json

emoji_codes = pd.read_json('emojicodes.json', orient='values', typ = 'series').str.extract(':(\w+):', expand = False).sort_index()

#import labeled twitter data
df =  pd.read_csv('Sentiment Analysis Dataset.csv', index_col=0, names = ['sentiment', 'source', 'text'], error_bad_lines=False)[['sentiment','text']]
#pulls a random sample of 100000, changes sentiment to numeric
df_sample = df.sample(200000).reset_index(drop=True)
df_sample.sentiment = pd.to_numeric(df_sample.sentiment)
#checks that the sample mean is reasonable
np.mean(df_sample.sentiment)


#import tweets and replace text with full text if 'tweet' is a retweet
df = pd.read_json('immigrationTweets.json')
df.text[~df.retweeted_status.isnull()] = df[~df.retweeted_status.isnull()].retweeted_status.apply(lambda x: x.get('text'))
df = df[['id','user','text','lang','reply_count','retweet_count','retweeted_status','term']]


#import and parse emoji codes

#import vocab and model, define sentence tokenizer, set chunk_size
with open('/Users/ikennedy/Documents/GitHub/torchMoji/model/vocabulary.json') as f: vocab = json.load(f)
model = torchmoji_emojis('pytorch_model.bin')
st = SentenceTokenizer(vocab, 30)

#specifiy colums for full df for:
#twitter pull
df_full = pd.DataFrame(columns=['id','user','text','lang','reply_count','retweet_count','retweeted_status','term']+list(emoji_codes))
#Twitter sample
df_full = pd.DataFrame(columns=['sentiment', 'text']+list(emoji_codes))
#runn in a loops of 5000 to avoid overusing computational resources
chunk_size = 5000
i = 1000
chunk_size = 1000
for i in range(chunk_size,len(df)+chunk_size,chunk_size):
    if(i>len(df)):
        i = len(df)
        chunk_size = len(df) % chunk_size
    #grab the subset of documents
    documents = list(df.text[i-chunk_size:i])
    #tokenize them
    tokens, infos, stats = st.tokenize_sentences(documents)
    #fit the probabilities
    prob = model(tokens)
    #append the results to df_full
    df_full = df_full.append(df[i-chunk_size:i].join(pd.DataFrame(prob, columns=emoji_codes, index=range(i-chunk_size,i))))
    #print the status
    print("finished processing", i, "documents")

#confirm shape


#save as csv
filename = 'rated_tweet_set.csv'
df_full.to_csv(filename)
df_full.tail()
