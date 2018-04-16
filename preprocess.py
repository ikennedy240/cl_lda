"""
This Module includes functions for processing and cleaning scraped craigslist
data to prepare for text analysis using LDA or other NLP analysis
"""

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

test = seattlefull.body_text
test = test.str.replace("QR Code Link to This Post", '').str.replace(r'\n','').str.replace(r'\r',' ').str.strip()
test = test.str.replace(r'\S*(\.com|\.net|\.gov|\.be|\.org)\S*','#URL').str.replace(r'http\S*', '#URL').str.replace(r'\d+', '#NUMBER')
test = test.str.replace(r'^,+','').str.replace(r',,+','')str.strip()
df.body_text = df.body_text.str.replace(r'!!!!+',' shii ').str.replace(r'!!!', ' mitsu ').str.replace(r'!!', ' nii ').str.replace(r'!', 'ichi')
df.body_text = df.body_text.str.replace('\$\$+','dollasigns').str.replace('\$', 'dollasign')
df.body_text = test

#take out neighborhood Names

for name in hoods:
    df['body_text'] = df.body_text.str.replace(r' ?'+name+' ?', " #HOOD ", case=False)
    if hoods.index(name) % 100 == 0:
        print(hoods.index(name))

#Deal with duplicates

cl_full = pd.read_csv('data/full_cl_dump.csv')
cl_full.shape

cl_dropped = df.drop_duplicates(['latitude','longitude','price'])
cl_dropped.shape
cl_dropped['body_100']=cl_dropped.body_text.str.slice(stop=1000).copy()
cl_dropped.body_100.unique().shape
cl_dropped.to_csv('data/cl_dropped.csv')