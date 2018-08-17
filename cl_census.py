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

def StateTractData(st, var_dict=None, save=False, year=2015):
    if var_dict is None:
        try:
            tmp = pd.read_csv("resources/"+st+"tracts.csv", index_col=0, dtype = {'GEOID10':object,'blockid':object})
            module_logger.info('read file')
            return tmp
        except:
            var_dict = {'B03002_001E': 'total_RE','B03002_003E': 'white', 'B03002_004E': 'black', 'B03002_005E': 'aindian', 'B03002_006E': 'asian', 'B03002_007E': 'pacisland', 'B03002_009E': 'other', 'B03002_012E': 'latinx', 'B17001_001E': 'total_poverty', 'B17001_002E': 'under_poverty', 'B19013_001E': 'income', 'B06009_011E': 'col_degree', 'B08015_001E':'commute'}
        #Census API code
    with open('resources/censusapikey.txt', 'r') as f:
        census_key = f.readlines()[0].strip()
    c = Census(census_key)
    statefips = eval("states."+st+".fips")
    tmp = pd.DataFrame(c.acs.state_county_tract(fields = list(var_dict.keys()),state_fips=statefips, county_fips="*", tract="*", year=year))
    #construct column with tract code
    tmp['GEOID10']= tmp.state+tmp.county+tmp.tract
    #give it understandable columns
    tmp.rename(columns=var_dict, inplace=True)
    tmp['acs_year'] = year
    #Write to CSV
    if save:
        tmp.to_csv("resources/"+st+"tracts.csv")
        module_logger.info('saved '+st+'census data to resources/'+st+"tracts.csv")
    module_logger.info('gened file')
    return tmp


# merge state tracts with cl data by GEOID10
def mergeCLandCensus(cldata,state='WA',strat_col=None,thresh=None, var_dict= None, geocode=False):
    #merge with state tract data
    if geocode:
        cldata = getCensusCode(cldata)
    try:
        cl_withtracts = cldata.merge(StateTractData(state, var_dict),how='left',on='GEOID10')
    except:
        print("Looks like this data is missing a GEOID10 column.\n Should we create one?\n NOTICE: This could take some time~~")
        if input("Continue? y/n") =='y':
            cldata = getCensusCode(cldata)
            cl_withtracts = cldata.merge(StateTractData(state, var_dict),how='left',on='GEOID10')
        else:
            return cldata
    #create a dummy variable '1' for neighborhoods with strat_col over a margin
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
        new_file = pd.read_csv(filepath, index_col=0, dtype = {'GEOID10':object,'blockid':object})
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
