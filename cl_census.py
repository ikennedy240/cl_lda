"""
This module supports merging CL data with census data
"""

import pandas as pd
import numpy as np
import os
from census import Census
from us import states
import regex as re
import logging
import googlemaps
from datetime import datetime
import json

#gmaps = googlemaps.Client(key='Add Your Key here')

module_logger = logging.getLogger('cl_lda.cl_census')
# Get census codes given latitude and longitude

def getCensusCode(cldata):
    import urllib.request
    from urllib.parse import urlencode
    #Adds census 'blockid' and 'GEOID10' columns to CL Data
    #cldata must have columns called 'latitude' and 'longitude' as floats
    cldata['blockid']=0
    for x in range(cldata.shape[0]):
        row = cldata.iloc[x].copy()
        url = 'https://geocoding.geo.census.gov/geocoder/geographies/coordinates?'+urlencode({'x':str(row.longitude), 'y':str(row.latitude), 'benchmark':'4', 'vintage':'4', 'format':'json'})
        try:
            tmp = urllib.request.urlopen(url, timeout=60).read()
            cldata.loc[x,'blockid'] = json.loads(tmp)['result']['geographies']["2010 Census Blocks"][0]["GEOID"]
            module_logger.info("First Try: "+str(x))
        except:
            try:
                tmp = urllib.request.urlopen(url, timeout=60).read()
                cldata.loc[x,'blockid'] = json.loads(tmp)['result']['geographies']["2010 Census Blocks"][0]["GEOID"]
                module_logger.info("Second Try: "+str(x))
            except:
                cldata.loc[x,'blockid'] =  np.nan
                module_logger.info("Set to Nan: "+str(x))
    cldata['GEOID10'] = cldata.blockid.str.slice(0,11).copy()
    return cldata


# Get state tract data

def StateTractData(st):
    try:
        tmp = pd.read_csv("resources/"+st+"tracts.csv", dtype = {'GEOID10':object,'blockid':object})
        module_logger.info('read file')
    except:
        #Census API code
        with open('resources/censusapikey.txt', 'r') as f:
            census_key = f.readlines()[0].strip()
        c = Census(census_key)
        help(c.acs)
        statefips = eval("states."+st+".fips")
        c.acs.state(fields = ['B02001_001E', 'B02001_002E','B02001_003E'],state_fips=statefips)
        tmp = pd.DataFrame(c.acs.get(['B02001_001E', 'B02001_002E','B02001_003E'], geo={'for': 'tract:*','in': 'state:{} county:*'.format(statefips)}))
        #construct column with tract code
        tmp['GEOID10']= tmp.state+tmp.county+tmp.tract
        #give it understandable columns, and created percent white column
        tmp.rename(columns={'B02001_001E': "total_pop", 'B02001_002E': 'white_pop','B02001_003E': 'black_pop'}, inplace=True)
        tmp['percent_white'] = tmp.white_pop/tmp.total_pop*100
        #Write to CSV
        tmp.to_csv("resources/"+st+"tracts.csv")
        module_logger.info('gened file')
    return tmp


# merge state tracts with cl data by GEOID10
def mergeCLandCensus(cldata,state='WA',strat_col=None,thresh=None,geocode=False):
    #merge with state tract data
    if geocode:
        cldata = getCensusCode(cldata)
    try:
        cl_withtracts = cldata.merge(StateTractData(state),how='left',on='GEOID10')
    except:
        print("Looks like this data is missing a GEOID10 column.\n Should we create one?\n NOTICE: This could take some time~~")
        if input("Continue? y/n") =='y':
            cldata = getCensusCode(cldata)
            cl_withtracts = cldata.merge(StateTractData(state),how='left',on='GEOID10')
        else:
            return cldata
    #create a dummy variable '1' for neighborhoods with white population over a certain percentage
    if strat_col is None:
        return cl_withtracts
    else:
        if thresh is None:
            thresh = cl_withtracts[strat_col].median()
        cl_withtracts[strat_col]=np.where(cl_withtracts[strat_col]>=thresh, 1, 0)
        return cl_withtracts

# import prepped data
def import_scraped_data(path = 'data/new_data', archive = 'archive', save=True):
    full_df = pd.DataFrame()
    files = os.listdir(path)
    files.remove('.DS_Store')
    files = [i for i in files if re.match('prepped',i)]
    for i in files:
        filepath = path+'/'+i
        new_file = pd.read_csv(filepath, index_col=0)
        new_file = new_file
        full_df = pd.concat([full_df,new_file])
        os.rename(filepath, archive+'/'+i)
        module_logger.info("Imported "+i)
    if save:
        now = datetime.now()
        full_df.to_csv('data/import_'+str(now.month)+'_'+str(now.day)+'.csv')
        module_logger.info("Completed input and wrote import_"+str(now.month)+'_'+str(now.day)+'.csv to file')
    return full_df

# prep newly scraped files by adding date, geoid, and census info
def prep_scraped_data(path = 'data/new_data', archive = 'archive', import_all=False):
    files = os.listdir(path)
    files.remove('.DS_Store')
    files = [i for i in files if not re.match('prepped',i)]
    for i in files:
        #take in new file, give it a date
        filepath = path+'/'+i
        month, day = re.search(r'(\d+)_(\d+)', i).group(1,2)
        new_file = pd.read_csv(filepath, index_col=0, dtype = {'GEOID10':object,'blockid':object}).assign(scraped_month = month, scraped_day = day)
        module_logger.info("Imported "+i+", added date\n COMMENCING CENSUS GEOCODING")
        #run getCensusCode on it,
        # merge mergeCLandCensus
        new_file = mergeCLandCensus(new_file, geocode=True)
        # rename file: prepped+file
        new_file.to_csv(path+'/'+'prepped'+i)
        os.rename(filepath, archive+'/'+i)
        module_logger.info("Merged with census and wrote "+i+" to prepped file")
    if import_all:
        return import_scraped_data(path,archive)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    seattle = pd.read_csv('data/seattlefull.csv', index_col=0, dtype = {'GEOID10':object,'blockid':object})
    seattle_new = import_scraped_data()
    seattle_new.to_csv('data/seattle_new.csv')
    tmp = mergeCLandCensus(seattle_new)
    st = 'WA'
    small = seattle_new.iloc[:10].copy()
    small = getCensusCode(small[:10])
    small = mergeCLandCensus(small.iloc[:10])
    cldata = small.iloc[:10]
    seattle_census = getCensusCode(seattle_new)
    #take in new file, give it a date
    #run getCensusCode on it,
    # merge mergeCLandCensus
    # rename file: prepped+file
    files =  os.listdir('data')
    filtered = [i for i in files if not re.match('seattle',i)]
    prep_scraped_data()
    x = import_scraped_data()
    seattle = mergeCLandCensus(seattle)
    seattle = seattle.drop([0], axis=0).drop(['Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1)
    x.drop(['Unnamed: 0.1'], axis=1, inplace=True)
    seattle_full = pd.concat([x,seattle])
    seattle.to_csv('seattle_old_4_22.csv')
    seattle_full.to_csv('seattle_full.csv')
