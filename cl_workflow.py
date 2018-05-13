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
data_path = 'data/processed4_22.csv'
df = pd.read_csv(data_path, index_col=0,dtype = {'GEOID10':object,'blockid':object})

df.shape
df = df.dropna().reset_index(drop=True)




# """Import New Data if Needed"""
# new_data_path = "data/chcl.csv"
# new_df = pd.read_csv(new_data_path, index_col=0, dtype = {'GEOID10':object,'blockid':object})
#
# """Preprocess Raw Data"""
# remove duplicates
df.shape
df = df.drop_duplicates(subset='body_text')
df_small = preprocess.clean_duplicates(df, method=80)
df_small.shape
# add census info
df = cl_census.mergeCLandCensus(df)

# clean text
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

"""Make Other Contex Columns"""
df['top_topic'] = df[list(range(50))].idxmax(1)
df['log_income'] = np.log(df.income)
df['clean_price'] = pd.to_numeric(df.price.str.replace(r'\$|,',''), errors = 'coerce')
df['log_price'] = np.log(df.clean_price)
df = df.dropna().reset_index(drop=True)

#df.to_csv('data/processed4_22.csv')
"""Make Corpus and Dictionary"""
with open('resources/hoods.txt') as f:
    neighborhoods = f.read().splitlines()
from sklearn.feature_extraction import stop_words
hood_stopwords = neighborhoods + list(stop_words.ENGLISH_STOP_WORDS)
corpus, dictionary = preprocess.df_to_corpus([str(x) for x in df.clean_text], stopwords=hood_stopwords)

"""Run Lda Model"""
n_topics = 50
n_passes = 10
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

"""Look at the distributions of the different topics"""
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(df[[26]].sort_values(by=26, ascending=False).head(400).values)
plt.hist(df.groupby('GEOID10').mean()[[4]].dropna().values)
df.columns
df.groupby('race_income_label').mean()[26]
plt.hist(df[df.race_income==0][41].values)
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
X = df[["black_proportion","asian_proportion","latinx_proportion","log_income","log_price"]]
y = df.top_topic
LR = LogisticRegression()
LR.fit(X,y)
lr_coefs = pd.DataFrame(LR.coef_).rename({0:"black_proportion",1:"asian_proportion",2:"latinx_proportion",3:"log_income",4:"log_price"}, axis=1)
lr_coefs_black = pd.DataFrame(LR.coef_).rename({0:"high_black",1:"high_income"}, axis=1).sort_values('high_black')
lr_coefs_white = pd.DataFrame(LR.coef_).rename({0:"white_proportion",1:"log_income",2:"log_price"}, axis=1)

lr_coefs
lr_coefs = lr_coefs.assign(abs_black = abs(lr_coefs.black_proportion)).sort_values("abs_black", ascending=False)
mean_diff = lr_coefs.merge(lda_output.summarize_on_stratifier(df, n_topics, 'high_black'), left_index=True, right_index=True)
mean_diff.drop(['abs_black','difference', 'proportion'], axis=1, inplace=True)
mean_diff





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



"""Produces useful output of topics and example texts"""
reload(lda_output)
mean_diff
now = datetime.now()
lda_output.text_output(df, text_col='body_text', filepath='output/cl'+str(now.month)+'_'+str(now.day)+'.txt', model= model, sorted_topics=mean_diff, strat_col='race_income_label', cl=True)

x = lda_output.get_formatted_topic_list(model, n_terms=50)
['Topic 0: community, !___, apartments, pool, bedroom, home, access, center, park, court, offer, housing, opportunity, place, indoor, perfect, bath, approximately, fitness, equal, located, area, close, details, fireplace, monday, storage, just, contact, today, parking, friday, dishwasher, minutes, basketball, office, amenities, life, features, clubhouse, shopping, lake, available, internet, fields, link, personal, lounge, location, corner',
 'Topic 1: center, home, !___, indoor, private, patio, bedroom, closets, apartment, basketball, fitness, pool, storage, pets, offer, package, large, court, fireplace, complimentary, washer, internet, walkin, playground, restrictions, homes, dryer, balcony, naval, equal, ready, disposal, welcome, opportunity, housing, bath, wifi, state, clubhouse, dishwasher, fully, living, approximately, space, service, info, tanning, station, free, apartments',
 'Topic 2: community, apartment, homes, amenities, just, pool, nearby, live, bedroom, center, outdoor, living, apartments, lease, access, fitness, residents, easy, !___, home, heated, minutes, appointment, shopping, stroll, exclusive, entertainment, unique, provides, selection, picture, away, life, thoughtful, comfortable, warm, like, chance, town, coming, petfriendly, landscaping, communities, wide, embrace, favorite, dining, feature, furry, visit',
 'Topic 3: apartment, bedroom, community, contact, home, appliances, homes, views, center, modern, apartments, friendly, amenities, offers, opportunity, housing, place, spacious, energy, pines, city, floor, rent, deck, studio, bath, perfect, redhill, equal, rooftop, apply, efficient, welcome, approximately, light, features, broadstone, plans, fitness, select, beautiful, green, office, youll, dining, environment, info, access, options, restrictions',
 'Topic 4: room, homes, vintage, community, !___, $___, home, bedroom, minutes, amenities, access, located, luxury, center, social, spacious, views, youll, friendly, blocks, bath, senior, approximately, just, housing, apartment, offers, opportunity, controlled, right, welcome, fitness, flats, elevator, style, apartments, rent, parking, equal, dining, street, restrictions, garage, living, link, deposit, onsite, entertain, allowed, kitchen',
 'Topic 5: !___, home, today, contact, apartment, community, floor, info, bedroom, space, apartments, kitchen, tour, living, large, room, spacious, open, !!__, located, come, great, lease, beautiful, center, available, appliances, schedule, closet, free, just, offer, office, love, enjoy, storage, month, stainless, hour, subject, area, plan, access, change, features, view, washer, dryer, hours, bathroom',
 'Topic 6: community, park, outdoor, apartment, bedroom, onsite, private, online, pool, center, friendly, available, maintenance, indoor, fitness, close, housing, playground, homes, home, rent, resident, management, origin, equal, washerdryer, opportunity, access, service, heated, just, storage, dishwasher, emergency, minutes, offers, clubhouse, easy, amenities, free, entertainment, shopping, apartments, fireplaces, space, area, wood, included, courtesy, bath',
 'Topic 7: $___, pricing, subject, change, !___, availability, equal, center, opportunity, housing, monthly, deposit, access, park, bath, time, fitness, storage, level, contact, sqft, month, notice, homes, daily, best, price, post, details, craigslist, property, shopping, plan, stated, verify, accurate, amenities, clubhouse, easy, breed, ceilings, flatsingle, restrictions, private, community, friendly, bedroom, garage, advertised, guaranteed',
 'Topic 8: !___, community, living, apartments, flats, center, timbers, bedroom, studio, friendly, kitchens, appliances, time, home, laundry, bath, homes, features, hours, rooftop, policy, efficient, contact, housing, amenities, opportunity, equal, office, saturday, fitness, parking, storage, apartment, link, energy, info, space, private, convenience, windows, free, $___, residential, welcome, street, sunday, feel, unit, common, kitchen',
 'Topic 9: minutes, center, street, apartments, !___, blocks, friendly, community, access, easy, urban, field, fixtures, kitchen, link, away, space, windsor, pratt, city, patio, fitness, jackson, equal, dishwasher, park, opportunity, just, housing, galleries, safeco, upgraded, grounds, onsite, restrictions, parks, neighborhoods, lines, freeways, famous, elevator, shopping, surrounding, century, residents, seattles, history, popular, living, convenient',
 'Topic 10: bedroom, community, apartments, views, areas, park, room, appliances, studio, home, area, metro, modern, office, rooftop, bath, living, dryer, sound, washer, puget, fitness, !___, homes, available, housing, link, equal, opportunity, stainless, main, details, center, bedrooms, lighting, blvd, fully, equipped, storage, approximately, life, designer, deck, contact, mountains, parking, bike, lounge, features, twotone',
 'Topic 11: homes, bedroom, community, private, pool, outdoor, apartment, home, room, center, bath, large, fitness, onsite, storage, renovated, covered, access, features, size, select, area, kitchens, fireplaces, station, housing, opportunity, fireplace, !___, complimentary, pets, equal, unit, amenities, parking, park, available, dining, closets, washer, ceilings, welcome, office, electric, wood, walkin, entertainment, approximately, spacious, living',
 'Topic 12: bike, community, rooftop, units, select, stainless, countertops, quartz, blocks, flooring, modern, storage, live, steel, appliances, opportunity, lounge, floor, station, spaces, features, apartments, private, views, parking, amenities, expansive, bedroom, center, park, urban, tile, open, windows, fitness, room, housing, closets, interior, equal, !___, walking, social, plank, lockers, studio, info, kitchen, balconies, screening',
 'Topic 13: $___, deposit, rent, application, !___, month, refundable, monthly, income, available, security, allowed, apartment, pets, cats, apply, dogs, credit, rental, nonrefundable, bedroom, home, person, adult, apartments, lease, free, contact, months, history, details, laundry, small, parking, restrictions, additional, screening, bath, online, community, property, welcome, unit, located, breed, utilities, address, info, floor, time',
 'Topic 14: apartments, apartment, home, bedroom, community, internet, make, luxury, amenities, floor, studio, washington, !___, features, wildreed, live, high, speed, living, offer, kitchens, welcome, best, plans, stainless, appliances, tivalli, enjoy, fitness, rent, washer, pool, opportunity, bath, housing, need, convenient, available, approximately, ceilings, services, lifestyle, life, comfortable, court, access, private, area, modern, cable',
 'Topic 15: $___, month, bedroom, community, apartment, home, hill, pool, amenities, homes, center, patio, !___, additional, onsite, rent, friendly, appliances, housing, welcome, opportunity, place, apartments, wood, deposit, bath, fitness, offer, subject, availability, prices, maintenance, style, change, fireplace, renovated, private, beautiful, equal, park, swimming, pets, approximately, features, located, storage, clubhouse, enjoy, balcony, parking',
 'Topic 16: apartment, court, available, apartments, pointe, center, bedroom, !___, homes, home, living, amenities, fitness, heated, days, staff, size, pool, features, package, maintenance, property, welcome, basketball, lifestyle, indoor, entertainment, $___, james, closets, management, date, week, onsite, bath, offer, harbour, located, storage, atrium, professional, park, private, free, contact, dining, fireplace, info, options, business',
 'Topic 17: room, homes, views, community, deck, windows, amenities, sound, martin, kitchen, kitchens, design, center, floor, city, rooftop, area, luxury, residents, glass, equal, finishes, sophisticated, storage, contemporary, appliances, pets, offer, puget, feature, water, loft, floors, floortoceiling, home, energy, guest, features, fitness, enjoy, cabinetry, open, roof, walk, suite, inspired, opportunity, spectacular, housing, bedroom',
 'Topic 18: center, $___, storage, just, park, fitness, bedroom, home, room, community, exit, trails, open, access, roundabout, left, parking, builtin, shopping, available, covered, additional, town, closets, fullsize, rent, lounge, deposit, washer, location, clubhouse, appliances, housing, dryer, opportunity, turn, equal, link, microwave, monday, pool, floorplans, large, marymoor, city, hour, continue, master, creek, beach',
 'Topic 19: $___, unit, building, parking, month, rent, available, apartment, street, walk, water, deposit, bedroom, restaurants, included, lease, !___, storage, floor, kitchen, location, large, garbage, located, private, condo, blocks, bathroom, sewer, quiet, light, laundry, block, park, shops, space, great, walking, includes, distance, utilities, just, easy, appliances, close, security, pets, access, washerdryer, carpet',
 'Topic 20: !___, living, center, community, bedroom, lounge, home, fitness, storage, area, room, amenities, city, apartments, outdoor, bike, housing, access, apartment, hour, opportunity, equal, homes, bath, views, steel, rooftop, stainless, studio, hours, luxury, approximately, pool, package, natural, available, come, quartz, units, exit, appliances, select, features, closets, countertops, office, entertainment, style, policy, youre',
 'Topic 21: apartments, community, home, bedroom, living, access, rent, floor, amenities, modern, large, energy, plans, available, parking, kitchens, $___, easy, area, views, room, features, pets, appliances, center, fitness, designer, alaire, covered, rooms, storage, youll, gourmet, ceilings, opportunity, housing, complimentary, equal, offer, select, restaurants, online, come, bath, include, avana, resident, contemporary, landscaped, dining',
 'Topic 22: !___, apartment, center, bedroom, access, stainless, homes, steel, storage, living, appliances, renovated, home, parking, kitchen, community, convenient, amenities, internet, change, wood, subject, latitude, closets, hour, shopping, washer, fitness, maintenance, apartments, location, availability, bath, outdoor, friendly, garage, extra, opportunity, housing, floor, area, oversized, brand, dryer, offer, located, welcome, approximately, cabinetry, private',
 'Topic 23: homes, !___, select, community, apartment, home, storage, bedroom, tower, fitness, views, hour, lounge, garage, access, center, renovated, deck, equal, housing, opportunity, contact, resident, bath, appliances, parking, friendly, newly, information, amenities, approximately, features, info, spacious, office, flooring, decks, breed, living, high, apartments, outdoor, $___, pool, restrictions, studio, building, best, rent, internet',
 'Topic 24: !___, $___, home, apartments, apartment, enjoy, flooring, views, just, appliances, quartz, shopping, windows, stainless, steel, bedroom, kitchen, spacious, today, living, away, rooftop, floor, contact, deck, view, building, steps, center, parking, access, dining, wood, location, city, located, homes, amenities, community, available, features, tops, including, counter, beautiful, market, large, tour, washer, brand',
 'Topic 25: right, room, turn, community, wood, housing, center, style, home, bedroom, fitness, flooring, storage, apartment, bath, restrictions, left, upgraded, opportunity, equal, $___, !___, office, breed, fireplace, approximately, countertops, modern, pool, available, pets, exit, welcome, outdoor, link, dryer, washer, homes, cabinetry, living, apartments, come, designer, paint, lighting, steel, parking, kitchen, closets, onsite',
 'Topic 26: home, apartment, access, bedroom, breed, center, storage, make, bull, open, community, school, welcome, $___, !___, restrictions, spacious, onsite, woodcreek, year, pool, room, animal, breeds, terriers, park, easy, private, office, friendly, pets, management, fitness, subject, dogs, outdoor, change, internet, availability, area, mark, love, hour, american, minutes, details, fireplace, homes, apply, team',
 'Topic 27: center, !___, edge, century, access, home, space, close, onsite, shops, waters, plenty, station, fabulous, restaurants, fitness, bedroom, community, just, residential, pool, location, internet, youll, favorite, feel, views, needle, door, features, looking, friendly, setting, arena, unit, place, right, offer, sits, movie, great, availability, cozy, bath, wonderful, visit, opportunity, housing, dishwasher, lake',
 'Topic 28: pets, available, access, community, panorama, !___, minutes, apartment, large, modern, enjoy, $___, right, steel, stainless, package, appliances, home, deposit, pool, kitchen, located, building, parking, bedroom, bath, opportunity, housing, left, amenities, area, center, private, dollar, excellent, allowed, great, window, weight, equal, pointe, residents, harbour, restrictions, fitness, beautiful, closets, buildings, valley, covered',
 'Topic 29: !___, building, views, floor, unit, amenities, months, bike, leasing, windows, parking, free, access, storage, rooftop, tour, contact, deck, info, appliances, units, modern, city, notes, online, features, park, internet, property, dogs, book, community, onsite, apartments, availability, available, controlled, room, refrigerator, space, apartment, light, dishwasher, schedule, open, roof, management, month, housing, microwave',
 'Topic 30: home, bedroom, homes, park, community, private, center, pets, apartments, apartment, bedrooms, access, shopping, available, ridge, policy, $___, onsite, minutes, views, friendly, opportunity, bath, equal, housing, breed, village, pool, easy, miles, management, fitness, just, approximately, restrictions, local, located, additional, large, mall, link, town, dogs, washer, closets, dryer, renovated, welcome, dining, sound',
 'Topic 31: !___, home, $___, apartment, access, community, center, pets, apartments, park, contact, housing, bedroom, shopping, opportunity, bath, link, fitness, office, equal, private, approximately, text, close, restrictions, features, room, rent, amenities, details, homes, closets, info, apply, credit, welcome, breed, onsite, policy, information, wood, storage, parking, place, walk, dishwasher, resident, courtyard, fireplace, spacious',
 'Topic 32: community, home, !___, access, apartments, welcome, bedroom, homes, center, apartment, living, available, opportunity, park, housing, views, bath, village, approximately, allowed, room, policy, features, equal, hours, appliances, office, amenities, located, stainless, pets, space, link, parking, closets, fitness, friendly, crossing, windows, aspira, floor, mountain, rent, luxury, information, monday, $___, dining, steel, info',
 'Topic 33: community, friendly, center, maintenance, park, !___, located, yauger, access, amenities, pool, friends, lease, term, clubhouse, lane, brand, becket, internet, features, garages, fitness, apartments, conveniently, fairways, washer, site, housing, text, opportunity, nonrefundable, dining, make, family, state, info, speed, âbusiness, cooper, storage, apply, shopping, area, indoor, workout, home, âonsite, âpet, entire, contact',
 'Topic 34: $___, community, bedroom, !___, apartment, center, heights, apartments, fitness, rent, allowed, washer, pool, deposit, minutes, access, clubhouse, home, area, bath, amenities, onsite, spacious, located, dryer, storage, homes, room, closets, opportunity, living, housing, dishwasher, pets, feature, friendly, parking, residents, time, fireplace, approximately, weight, outdoor, private, townhomes, enjoy, features, equal, shopping, dining',
 'Topic 35: home, !___, $___, center, community, pool, available, fitness, room, pets, contact, office, storage, unit, homes, equal, housing, opportunity, bath, closets, info, lease, amenities, large, bedroom, maintenance, information, ceilings, apartment, leasing, clubhouse, hour, onsite, property, just, parking, features, mercer, located, access, apartments, dishwasher, washerdryer, months, living, kitchen, hours, high, patiobalcony, pond',
 'Topic 36: community, access, parking, rent, available, garbage, appliances, disposal, pool, dogs, bedroom, center, fitness, bath, ready, contact, homes, kitchen, area, $___, allowed, office, !___, apartments, oven, refrigerator, details, floor, canterbury, features, maintenance, equal, trash, service, dishwasher, online, home, opportunity, housing, cats, carpet, wall, heated, garage, maker, approximately, income, fireplace, range, amenities',
 'Topic 37: property, !___, available, $___, kitchen, lake, information, contact, questions, large, floors, walk, hardwood, agent, required, rent, park, tenants, unit, office, space, lighting, access, appliances, just, email, parking, bedrooms, home, application, showing, area, light, short, room, lease, living, great, info, natural, months, remodeled, easy, beautiful, washington, granite, applications, stainless, market, applicant',
 'Topic 38: hour, center, rooftop, flooring, fitness, appliances, bedroom, views, energy, countertops, quartz, apartments, lounge, lighting, !___, circa, community, modern, steel, stainless, bath, housing, equal, opportunity, windows, home, access, located, approximately, youll, kitchen, parking, living, deck, cabinetry, room, like, amenities, homes, accept, services, comprehensive, urban, features, reusable, link, tenant, high, reports, defined',
 'Topic 39: room, $___, contact, info, deposit, large, home, bath, lease, bedrooms, kitchen, living, garage, parking, pets, bedroom, floor, rent, features, month, available, property, master, yard, fireplace, !___, unit, house, space, laundry, year, dining, deck, months, area, storage, security, bathroom, tenant, appliances, dishwasher, great, bathrooms, hardwood, closet, floors, rental, close, smoking, fenced',
 'Topic 40: access, apartments, community, !___, apartment, center, amenities, fitness, living, home, shelby, lounge, resident, bedroom, opportunity, onsite, near, housing, enjoy, equal, bath, available, club, cable, palisades, parking, main, just, private, location, controlled, walk, modern, clubhouse, patio, courtyard, approximately, friendly, located, electric, games, restrictions, experience, package, dishwasher, minutes, city, door, studio, stunning',
 'Topic 41: $___, animal, !___, community, home, apartment, office, located, bedroom, income, animals, just, center, policy, proof, spacious, refundable, including, housing, rent, weight, access, springs, indigo, deposit, resident, breeds, homes, prior, amenities, following, minutes, fitness, limit, chow, features, freeway, equal, property, right, opportunity, state, month, conveniently, hour, large, closets, months, current, bath',
 'Topic 42: apartment, $___, rent, apartments, community, !___, spacious, center, youll, fitness, home, unit, pool, right, living, large, court, cambridge, room, housing, opportunity, love, black, windows, bath, equal, restrictions, bedroom, check, clubhouse, homes, need, lake, kitchen, kitchens, beautiful, outdoor, quality, amenities, value, available, affordable, washerdryer, close, grounds, access, blvd, deposit, dishwasher, size',
 'Topic 43: home, !___, homes, center, turn, left, exit, bedroom, street, restrictions, fitness, pool, community, right, access, equal, housing, bath, opportunity, spacious, breed, outdoor, floor, approximately, parking, amenities, wood, covered, shopping, available, friendly, washer, fireplace, dryer, burning, court, park, contact, link, apply, info, location, large, seasonal, plans, close, closets, weight, easy, pets',
 'Topic 44: community, !___, opportunity, bedroom, link, apartment, appliances, equal, housing, private, friendly, available, bath, green, approximately, home, located, contact, lake, energy, amenities, office, area, views, center, resident, lounge, restaurants, features, parking, hours, living, city, apartments, hour, minutes, policy, efficient, washer, fitness, tenant, reports, screening, reusable, accept, windows, homes, comprehensive, washington, info',
 'Topic 45: access, community, managed, parking, private, features, view, apartment, website, appliances, !___, lounge, area, closet, easy, storage, dryer, washer, kitchen, balcony, extra, electric, dishwasher, available, outdoor, patio, stove, controlled, fitness, bedroom, apartments, walkin, professionally, refrigerator, garage, insurance, studio, required, shopping, select, unit, microwave, stainless, restaurants, steel, minutes, room, maintenance, grill, emergency',
 'Topic 46: available, pool, center, unit, fitness, apartments, fireplace, amenities, clubhouse, closets, $___, dishwasher, today, washer, community, outdoor, dryer, contact, bath, info, parking, bedroom, onsite, housing, home, court, swimming, patio, !___, playground, storage, access, apartment, opportunity, maintenance, equal, large, homes, ceilings, disposal, public, features, balcony, high, park, covered, room, area, washerdryer, package',
 'Topic 47: !___, income, home, center, bedroom, $___, community, person, just, minutes, housing, friendly, equal, equinox, bath, amenities, opportunity, fitness, approximately, homes, located, welcome, city, dishwasher, views, washer, dryer, kitchen, storage, stainless, link, contact, business, pool, info, park, modern, life, living, breed, steel, apartment, offers, turn, stay, south, billiards, range, restrictions, accept',
 'Topic 48: miles, $___, community, center, school, !___, club, park, right, apartment, home, washington, lake, bedroom, parking, square, access, golf, hospital, apartments, additional, !!__, located, high, homes, closets, deposit, college, outdoor, city, public, fees, town, highgrove, overlake, friendly, free, fitness, monthly, refundable, details, apply, rent, main, university, great, courtyards, admiralty, house, building',
 'Topic 49: community, kitchen, room, center, bike, storage, courtyard, views, onsite, rooftop, electric, fitness, garage, apartments, access, controlled, amenities, located, friendly, public, appliances, parking, transportation, efficient, !___, housing, green, equal, opportunity, leed, bath, available, space, including, resident, decks, maintenance, view, floors, approximately, certified, experience, closets, accessgated, building, lifestyle, life, lounge, bedroom, energy']
