import os
import re
import pandas as pd
import numpy as np
from scipy.stats import threshold
#import emojilist
emoji_codes = pd.read_json('emojicodes.json', orient='values', typ = 'series').str.extract(':(\w+):', expand = False).sort_index()


#import imdb
imdb = pd.read_csv('imdb_emoji.csv', index_col=0).dropna()
imdb['binary'] = np.sign(np.sign(imdb.sentiment - 5)+1)
imdb['scaled'] = (imdb.sentiment-np.mean(imdb.sentiment))/np.std(imdb.sentiment)
imdb_dummies = pd.read_csv('imdb_emoji.csv', index_col=0).dropna()
imdb_dummies.iloc[:,2:67] = threshold(imdb_dummies.iloc[:,2:67], 0, .05,0)
x = np.mean(imdb.iloc[:,2:67]>.05)
print(np.mean(x), np.max(x), np.min(x))
x
imdb_dummies

#import rated_tweet_set
tweets = pd.read_csv('rated_tweet_set.csv', index_col=0)

#import immigration
immi = pd.read_csv('immigration_emoji.csv', index_col=0)
immi = immi.dropna(subset = emoji_codes)
immi['predict'] = rfc.predict_proba(immi[emoji_codes])
immi['sent'] = rfc.predict(immi[emoji_codes])

immi[['id','user','text','lang','reply_count','retweet_count','retweeted_status','term','sent']].to_csv('immi_rated.csv')

sample =  immi.sample(100).sort_values('predict').reset_index()
for i in range(100):
    print('Proba', sample.predict.iloc[i], '\n Text:', sample.text.iloc[i], '\n')


immi[immi.predict<.2].sample(10)['text'].values
immi.text[immi.predict==1].sample(20).values

#feature model_selection
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.0002))
X = imdb[list(emoji_codes)]
X.shape
X = sel.fit_transform(X)
#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(tweets[list(emoji_codes)],tweets['sentiment'])

#train classifier
class_scores = tweets[['text','sentiment']].copy()

'''Multiclassifier'''
from sklearn.svm import LinearSVC
svc = LinearSVC(dual=False, C=100).fit(X_train, y_train)

svc.score(X_train,y_train)
svc.score(X_test,y_test)
class_scores['svc'] = svc.predict(tweets[list(emoji_codes)])
class_scores.sample(10)
class_scores.text.iloc[147489]


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 9, weights='distance').fit(X_train, y_train)
knn.score(X_train,y_train)
knn.score(X_test,y_test)

class_scores['knn'] = knn.predict(imdb[list(emoji_codes)])

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
rfc.score(X_train,y_train)
rfc.score(X_test,y_test)

class_scores['rfc'] = rfc.predict(tweets[list(emoji_codes)])


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(X_train,y_train)
lr.score(X_train,y_train)

class_scores['lr'] = lr.predict(imdb[list(emoji_codes)])

class_scores.sample(10)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(verbose=1)
gbc.fit(X_train,y_train)
gbc.score(X_train,y_train)
gbc.score(X_test,y_test)


from sklearn.model_selection import GridSearchCV
param_grid = {"learning_rate":[1,.5,0.1,.01], "n_estimators":[100, 300, 1000], "max_leaf_nodes":[5,10,None]}
clf = GridSearchCV(gbc, param_grid, verbose=2)
clf.fit(X_train, y_train)

clf.best_estimator_.score(X_train, y_train)
clf.best_estimator_.score(X_test, y_test)

class_scores['gbc_norm'] = np.sign((gbc-np.mean(gbc))/np.std(gbc))
gbc = clf.best_estimator_.predict_proba(imdb[list(emoji_codes)])[:,0]

class_scores

pd.DataFrame(clf.cv_results_)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

'''Regression'''
regress_scores = imdb[['text','sentiment','binary', 'scaled']].copy()
X_train, X_test, y_train, y_test = train_test_split(imdb_dummies[emoji_codes],imdb['scaled'])

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=.01).fit(X_train,y_train)
ridge.score(X_train,y_train)
ridge.score(X_test,y_test)
regress_scores['ridge'] = ridge.predict(imdb_dummies[emoji_codes])

np.mean(np.sign(regress_scores.scaled)==np.sign(regress_scores.ridge))

from sklearn.svm import SVR
svr = SVR().fit(X_train,y_train)
svr.score(X_train,y_train)
svr.score(X_test,y_test)

regress_scores['svr'] = svr.predict(imdb_dummies[emoji_codes])
np.mean(np.sign(regress_scores.scaled)==np.sign(regress_scores.svr))

regress_scores['diff'] = abs(regress_scores.scaled-regress_scores.ridge)

wrong = regress_scores[regress_scores['diff']>2]
#predict for immigration
wrong[wrong.sentiment>8].text.iloc[2]
