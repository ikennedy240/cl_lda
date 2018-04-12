# cl_lda
I'm working with a test sample of about 30,000 craigslist adds to look for lexical and discursive differences by neighborhood racial composition. The data here is just the text of the ad and a dummy variable 'high_white' indicating if the census block of the ad address in is a neighborhood above or below median white proportion for census blocks in the sample (this is around 66% white). I picked that just to have equal numbers of each type of text in my sample, it's probably not the best proportion. 

The code here cleans up the data, including removing about half the texts for being too similar (I struggled with this all day, so if you have a better solution that would be awesome (: ). Then it fits an LDA model with 50 topics, orders those topics by how different they are in relation to the high_white dummy. I sort the topics by difference in how many texts include at least some percentage of a topic (1% currently), and by how useful they are to a random forest classifier. 

I'm really interested in understanding what the topics are, so then I print topic details and example texts to a file.
