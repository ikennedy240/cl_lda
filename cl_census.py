"""
This module supports merging CL data with census data
"""

# Get census codes given latitude and longitude

def getCensusCode(CLdata):
    #Adds census 'blockid' and 'GEOID10' columns to CL Data
    #CLdata must have columns called 'latitude' and 'longitude' as floats
    with pd.option_context('mode.chained_assignment', None):
        CLdata['blockid']=0
        for x in range(CLdata.shape[0]):
            row = CLdata.iloc[x]
            url = 'https://geocoding.geo.census.gov/geocoder/geographies/coordinates?'+urlencode({'x':str(row.longitude), 'y':str(row.latitude), 'benchmark':'4', 'vintage':'4', 'format':'json'})
            try:
                tmp = urllib.request.urlopen(url, timeout=60).read()
                CLdata.blockid.iloc[x] = json.loads(tmp)['result']['geographies']["2010 Census Blocks"][0]["GEOID"]
                print("First Try: ", x)
            except:
                try:
                    tmp = urllib.request.urlopen(url, timeout=60).read()
                    CLdata.blockid.iloc[x] = json.loads(tmp)['result']['geographies']["2010 Census Blocks"][0]["GEOID"]
                    print("Second Try: ", x)
                except:
                    CLdata.blockid.iloc[x] =  np.nan
                    print("Set to Nan: ", x)
    CLdata['GEOID10'] = CLdata.blockid.str.slice(0,11)
    return CLdata


# Get state tract data

def StateTractData(st):
    try:
        x = pd.read_csv("resources/"+st+"tracts.csv", dtype = {'GEOID10':object,'blockid':object})
        print('read file')
    except:
        #Census API code
        c = Census("a36d29f80d1e867eb35fba5f935294928c1320be")
        statefips = eval("states."+st+".fips")
        x = pd.DataFrame(c.acs.get(['B02001_001E', 'B02001_002E','B02001_003E'], geo={'for': 'tract:*','in': 'state:{} county:*'.format(statefips)}))
        #construct column with tract code
        x['GEOID10']= x.state+x.county+x.tract
        #give it understandable columns, and created percent white column
        x.rename(columns={'B02001_001E': "total_pop", 'B02001_002E': 'white_pop','B02001_003E': 'black_pop'}, inplace=True)
        x['percent_white'] = x.white_pop/x.total_pop*100
        #Write to CSV
        x.to_csv(st+"tracts.csv")
        print('gened file')
    return x


# merge state tracts with cl data by GEOID10
def mergeCLandCensus(cldata,state,thresh=67):
    #merge with state tract data
    cl_withtracts = cldata.merge(StateTractData(state),how='left',on='GEOID10')
    #create a dummy variable '1' for neighborhoods with white population over a certain percentage
    cl_withtracts['high_white']=np.where(cl_withtracts['percent_white']>=thresh, 1, 0)
    return cl_withtracts
