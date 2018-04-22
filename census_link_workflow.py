import pandas as pd
import logging
import cl_census
from datetime import datetime
#start logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load old data:
seattle = pd.read_csv('data/seattlefull.csv', index_col=0, dtype = {'GEOID10':object,'blockid':object})

#prep any scraped files in 'data/new_data' (can set different path)
#this adds some metadata, like the scraped datetime and the important GEOID10 variable
seattle_new = cl_census.prep_scraped_data(import_all = True)

#merge with census data
seattle_new = cl_census.mergeCLandCensus(seattle_new)

#merge new and old data
seattle_full = pd.concat([seattle, seattle_new])

#write to files
now = datetime.now()
seattle_new.to_csv('data/seattle_'+str(now.month)+'_'+str(now.day)+'.csv')
seattle_full.to_csv('data/seattle_full.csv')
