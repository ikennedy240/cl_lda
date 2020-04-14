# Racialized Language and Discourse in Seattle Rental Ad Texts

This repo is the code for the paper with the above title currently under review at *Social Forces*. Below find the authors and their affiliations and a summary of findings.

Ian Kennedy<sup>1</sup>, Chris Hess<sup>1</sup>, Amandalynne Paullada<sup>2</sup>, Sarah Chasins<sup>3</sup>

University of Washington, Department of Sociology<sup>1</sup>

University of Washington, Department of Linguistics<sup>2</sup>

University of California at Berkeley, Department of Computer Science<sup>3</sup>

## Summary of Findings

### Abstract
Racial discrimination has been a central part of residential segregation for many decades, in the Seattle area as well as in the United States as a whole. In addition to redlining and restrictive housing covenants, housing advertisements included explicit racial language until 1968. Since then, housing patterns have remained racialized, despite overt forms of racial language becoming less prevalent. We use Structural Topic Models (STM) to investigate whether and how contemporary rental listings from the Seattle-Tacoma Craigslist page differ in association with neighborhood racial composition. STM and qualitative analysis show that listings from White neighborhoods emphasize trust and connections to neighborhood history and culture, while listings from non-White neighborhoods offer more incentives and focus on transportation and development features, sundering these units from their surroundings. Analysis of security discourse reveals that not only is language about security more common in less White neighborhoods, but administrative data show that actual security systems are less common. Without explicitly mentioning race, these listings reveal that racialized neighborhood discourse, which might impact neighborhood decision-making in ways that contribute to housing segregation, is present in Seattle’s contemporary rental housing advertisements. 

## Background

While scholars often engage with the housing market through large datasets, most people do so through housing searches, often reading real estate advertisements. These short descriptions of housing units tell home seekers about units and their surroundings, but also potentially provide insight into what the landlord or property manager believes is most salient to their potential tenants. Though listings almost certainly include bedroom size, rent, and square footage, they also describe the surrounding neighborhood. The history of housing in the United States is entwined with racial exclusion and motivates our investigation of whether legacies of segregation may also be present in rental advertisements themselves. 

During the era of de jure discrimination through mechanisms like redlining, exclusionary housing covenants and racial violence, historical advertisements were often explicitly discriminatory. For instance, an advertisement from The Seattle Times on August 9, 1934 offered four rooms for eighteen dollars ($341 in 2019 dollars) and specified “whites only.” Over time, language became more coded, such that by the late 1940s, racial exclusion was described in veiled language as a ‘restriction,’ as in one advertisement posted in The Seattle Times on December 23, 1947, which read “Property values will continue to increase in this new restricted district on Magnolia Bluff.” This district, now a wealthy Seattle neighborhood, was made restricted through the use of exclusionary housing covenants, discriminatory agreements used across the United States that excluded people of certain races from owning or living in a home (Rothstein 2017). In turn, these covenants prevented non-White households in Seattle and other metropolitan areas from building wealth through homeownership and also limited their access to certain public schools (Fernald 2019, Denton 1995).  

By 1968, rents had increased so that a three-bedroom home in the Southend of Seattle was advertised in the same paper on February 18 of that year for two hundred and fifty dollars a month (over $1800 in 2019 dollars). The listing was short, and shown in Figure 1: 

![Newspaper clipping](/plots/figure1.png)

In contrast to a couple decades prior, landlords rarely included such strongly racial language in the late 60s . For one thing, most people likely already knew which neighborhoods were friendly for which people. The Southend, though, bordered the Central District, where most Black households in Seattle lived. Perhaps the person who posted this listing wanted to make sure that prospective tenants knew who was welcome and may have worried that the neighborhood name might discourage White tenants. At the same time, the writer of this listing might have worried that without this kind of coded exclusion, they would need to turn away prospective Black tenants trying to rent the unit. 

In this study, we use rental housing listings from Craigslist in the Seattle-Tacoma-Bellevue, WA metropolitan area as a case study to understand how the description of housing opportunities varies systematically with the racial and ethnic composition of neighborhoods. We first use unsupervised topic modeling, a method that recognizes groups of words that often appear together, to identify a set of topics prevalent across our dataset of rental listings. Next, we estimate regression models to identify how much each topic is associated with neighborhood racial and ethnic composition. Finally, we describe the topics associated with variation across neighborhoods using qualitative analysis and deep reading. There are differences in rental listings across neighborhoods that are both statistically and practically significant. We call these observed differences racialized neighborhood discourse. This concept aligns with psychological and geographical work on the racialization of place (Bonam, Taylor, and Yantis 2017; Bonam et al. 2017; Bonam, Yantis, and Taylor 2018; Bonds and Inwood 2016; Weheliye 2014) and the way new White residents often fail to integrate into existing community structures when moving to less White neighborhoods (Walton 2018). Our analysis contributes insight into Seattle’s landscape of racialized neighborhood discourse and explores how variations in language used to describe neighborhood locations could be implicated in the reproduction of residential segregation.

## Findings

Without considering neighborhood differences, Craigslist rental listings tend to have a consistent format and present similar types of information. Since the purpose of these texts is to attract renters, the texts need to include information that home seekers might use to select a place to live. Accordingly, almost all advertisements include details about the unit for rent, like the size, number of bedrooms, amenities, and monthly costs. Based on our readings of historical listings, similar details were present in listings from the 1930s-1960s, which differ from contemporary listings primarily in length because newspapers charged posters by the letter. 

Information about the surrounding area is common, but not universal. If a listing includes explicit information about the neighborhood, that information is generally focused on nearby places to shop, eat, or visit, and local transportation options. We consider it unlikely for a listing to mention an amenity, nearby attraction, or unit feature that is not present, as that would be misleading. However, the opposite case where such details are omitted is more imaginable, and likely quite common. Even if a particular unit or neighborhood feature, say a café or a security system, is present in reality, it may not show up in a text. Results from each step of our analysis focus on how these discursive patterns are different on average for advertisements from neighborhoods with differently racialized populations. 
The STM produced 40 topics, defined by a high probability of containing certain groups of words, and a vector of topic proportions for each document. We label each topic by examining the words most associated with it and reading example texts, and refer to the topic by that label and, in parentheses, the number it was assigned by STM. We focus on the way aspects of a listing, like the description of the unit and neighborhood, discourse about commuting and access, and the marketing techniques, vary across neighborhood types. Table 1 shows key topics in each of those groupings, the words most associated with those topics and the titles we assigned through qualitative coding. The top words are based on STM’s ‘FR-EX’ measure which uses word frequency and exclusivity within a topic (Roberts et al. 2014).

Topic | Title	Top words using ‘FR-EX’, stemmed
------------- | -------------
**Access** |		
Short and Central (Topic 7) |	unit, plex, apt, triplex, build, coin
Driving and Bus Times (Topic 30) |	hospit, safeway, westwood, meyer, colleg, express, mile
Commuting Distance (Topic 39) |	burk, trail, gilman, microsoft, campus, googl
Convenience and Ease (Topic 40) |	locat, great, conveni, beauti, easi, open
**Marketing** |			
Shared Units (Topic 1) |	mother, cottag, mil, share, entranc, law
Developments as Communities (Topic 4) |	afford, emerg, site, onsit, mainten, paperless
Pools and More (Topic 31) |	court, tenni, pool, swim, burn, sauna
Subleases (Topic 32) |	someon, subleas, leav, sinc, renew, know
**Neighborhood Descriptions** |	
High Class Surroundings (Topic 8) |	cours, golf, jefferson, height, point, sand
Condos with Views (Topic 20) |	tower, elliott, skylin, stun, concierg, penthous
Safe and Friendly (Topic 26) |	breed, restrict, patrol, weight, select, friday
Vintage Charm (Topic 28) |	studio, vintag, classic, summit, brick, rail
**Unit Descriptions** |		
Cozy and Comfortable (Topic 16) |	winter, warm, morn, nearest, keep, climat
Elegant Homes (Topic 17) |	formal, bonus, piec, upstair, master, famili
Tenant Restrictions (Topic 19) |	report, incom, reusabl, evict, histori, comprehens
Homes with Personality (Topic 23) |	craftsman, basement, driveway, backyard, unfinish, furnac

While the titles of most topics are apparent simply from the ‘FR-EX’ words, some, like “Short and Central” (Topic 7) relied more on the qualitative coding process to produce the title.

We regress the log of the topic proportions for each document on the neighborhood typology and other covariates  to assess that topic’s association with neighborhood racial composition. We report the results of log-level OLS regression of estimated topic proportions on a held-out test set of documents using predominantly White neighborhoods as the reference category and include covariates with information about the units for rent, the neighborhood, and demographic context, listed above.  Log-level coefficients of β on a neighborhood type dummy, for example, majority non-White, suggest that a switch from a listing in a predominantly White neighborhood to a majority non-White neighborhood is associated with an increase in the topic proportion of βe^β×100%. Figures 2-5 report results from this model. Appearing on the left side of these plots indicates that the relevant neighborhood type is associated with less of that topic, compared to predominantly White neighborhoods, while appearing on the right indicates an association with relatively more of that topic. 

For example, results for “Commuting Distance” (Topic 39) and “Convenience and Ease” (Topic 40) are shown in Figure 2. The coefficient associating “Commuting Distance” (Topic 39) with majority non-White neighborhoods is 0.51. That means that we expect that, on average, listings for a unit in majority non-White neighborhoods will include this topic 0.51e^0.51×100% or 84.9% more than listings in predominantly White neighborhoods. We use this method to calculate the percentage increases reported below. In contrast to the large effect for “Commuting Distance” (Topic 39), “Convenience and Ease” (Topic 40), a topic which centered on easy to access storage and convenient parking, had only small differences in its prevalence across neighborhood types.

![regression plot](/plots/figure2.png)

The quantitative results identify the topics which vary with neighborhood racial proportion. Topics which have to do with trust (Topics 1 and 32), with personality and other less quantifiable positive qualities (Topics 23, 28), and centrality (Topics 7 and 20) are associated more with predominantly White neighborhoods. Topics associated with travel (Topics 30 and 39), safety (Topic 26), and property amenities (as opposed to unit amenities) (Topics 8, 4, and 31) are more associated with less White neighborhoods. 

For example, consider two topics concerning trust, shown in Figure 3. “Shared Units” (Topic 1) listings advertise units that are attached to the landlord’s home or property, often called accessory units, and “Subleases” (Topic 32) are requests for new tenants to assume a lease or sublet for a short period of time. Both of these arrangements require high levels of trust between the two parties. Our model estimates that, compared to a listing from a predominantly White neighborhood, a listing from a less White tract contains 15.0% (for White Mixed neighborhoods) to 23.7% (for majority non-White neighborhoods) less of “Shared Units” (Topic 1) and from a non-significant 3.1% increase (for majority non-White neighborhoods) to a 13.4% decrease (for White Latinx neighborhoods) of “Subleases” (Topic 32). This suggests that these high-trust arrangements may be most commonly advertised in the Whitest neighborhoods. We can also arrive at this same inference by examining Figure 3 and noticing that Topics 1 and 32 are arranged on the left side of the plot, indicating their association with predominantly White neighborhoods.

![regression plot](/plots/figure3.png)

We also compare regression results and spatial distribution, with an example shown in Figures 4 and 5. 

![map plot](/plots/figure4.png)
![regression plot](/plots/figure5.png)

We can see from Figure 5 that “Vintage Charm” (Topic 28) is clustered in central Seattle. This topic focused on smaller units in older buildings built with natural materials, and included words like ‘vintage’, ‘charm’, ‘brick’, ‘hardwood’, and ‘studio’, and was associated with predominantly White neighborhoods. Contrastingly, “Safe and Friendly” (Topic 26), includes a number of central listings, but has many more peripheral listings than “Vintage Charm” (Topic 28). This pattern—that topics associated with more White neighborhoods are also more central—occurs for other topics as well. In part, this reflects the long-standing spatial and racial demographic order in Seattle. The oldest, most central, and most established neighborhoods have been Whiter because of explicitly racially motivated redlining and racial covenants enforced by real estate agents who could have been expelled from the Real Estate Board for non-compliance (Rothstein 2017). These historical patterns have been exacerbated by changes in Seattle over the past decades, as non-White and poorer populations have been pushed out of more desirable central areas by rising rents and evictions (Thomas 2017, Hess 2020). In other words, neighborhoods’ racial composition and their peripheral status are intertwined. 

![map plot](/plots/figure6.png)

Both “Driving and Bus Times” (30) and “Commuting Distance” (39) were more associated with less White neighborhood types (see Figure 2). Given that more central neighborhoods have also been whiter in Seattle, it would be reasonable to consider that this association was a product of less White neighborhoods also requiring more talk about transportation options simply because they were further from the center of the city. However, while the spatial distribution of these topics, shown in Figure 6, does include listings in peripheral areas, both topics also include significant numbers of listings in central areas, especially less White central areas. While listings including “Vintage Charm” (28) are more common in Whiter, more central neighborhoods, less White neighborhoods are more likely to mention transportation options—ways to leave the area—regardless of whether they are central or peripheral.

This is true especially with mentions of Seattle’s light rail, which occurred in 6.3% of listings from neighborhoods with a Black proportion above the median for the sample, but only 2.0% of other listings. However, the light rail in Seattle passes through more traditionally Black neighborhoods which could account for some of that difference. To measure that connection, we examined the prevalence of the term ‘light rail’ only for listings geocoded to within one mile of the train’s route, leaving 5,691 listings from above-median Black tracts and 1,774 listings from other tracts. In that subset, 21.5% of listings from high-Black neighborhoods included ‘light rail’, while only 13.0% of listings from other neighborhoods did, a significantly larger prevalence with t=8.77,p<2.2e^(-16). Listings from tracts with higher Black proportion are much more likely to mention the light rail than listings from other areas that are equally close to that transportation option.

## Conclusion

The racialized neighborhood discourse described above is markedly different from the explicit racial exclusion seen in the 1930s or the coded racial language of ‘restricted districts’ seen until 1968. However, that difference cannot be called progress, and it certainly does not suggest the attainment of a post-racial present. Instead, our analysis shows pervasive thematic differences in the way listings in predominantly White neighborhoods treat community, trust, transportation, and safety when compared to less White neighborhoods. White neighborhoods are depicted with ties to history and community, while listings in less White neighborhoods enclose community in developments with their pools and fitness centers. Advertisements for situations with high trust, like where the renter and landlord live on the same property are more common in predominantly White neighborhoods. Even accounting for distance to transportation options, listings in less White neighborhoods seem more focused on how to leave a neighborhood than what you can do there. These differences in listing text, which are associated with the racial composition of a neighborhood, constitute racialized neighborhood discourse. Overall, these findings support critical race accounts of a racialized social system insidiously woven into the fabric of American social life (Bonilla-Silva 1997; Reskin 2012; Golash-Boza 2016). Listing texts include racialized discourse, in this account, not because text writers hold prejudicial beliefs, but because historical forces have produced racialized understandings of neighborhoods as having community or not, as being safe or unsafe, as being sufficient or insufficient.

For more information, please contact Ian Kennedy(ikennedy@uw.edu)

References
Besbris, Max, Jacob William Faber, Peter Rich, and Patrick Sharkey. 2015. “Effect of Neighborhood Stigma on Economic Transactions.” Proceedings of the National Academy of Sciences 112(16):4994–98.

Boeing, Geoff and Paul Waddell. 2017. “New Insights into Rental Housing Markets across the United States: Web Scraping and Analyzing Craigslist Rental Listings.” Journal of Planning Education and Research 37(4):457–476.

Boeing, Geoff. 2019. “Online Rental Housing Market Representation and the Digital Reproduction of Urban Inequality.” Environment and Planning A: Economy and Space 0308518X1986967.

Bonam, Courtney M., Hilary B. Bergsieker, and Jennifer L. Eberhardt. 2016. “Polluting Black Space.” Journal of Experimental Psychology: General 145(11):1561–82.

Bonam, Courtney M., Valerie J. Taylor, and Caitlyn Yantis. 2017. “Racialized Physical Space as Cultural Product.” Social and Personality Psychology Compass 11(9):e12340.

Bonam, Courtney, Caitlyn Yantis, and Valerie Jones Taylor. 2018. “Invisible Middle-Class Black Space: Asymmetrical Person and Space Stereotyping at the Race–Class Nexus.” Group Processes & Intergroup Relations 136843021878418.

Bonds, Anne and Joshua Inwood. 2016. “Beyond White Privilege: Geographies of White Supremacy and Settler Colonialism.” Progress in Human Geography 40(6):715–33.

Bonilla-Silva, Eduardo. 1997. “Rethinking Racism: Toward a Structural Interpretation.” American Sociological Review. 62(3):465-480.

Bonilla-Silva, Eduardo. 2006. Racism without Racists: Color-Blind Racism and the Persistence of Racial Inequality in the United States. Lanham, MD: Rowman & Littlefield Publishers.

Bonilla-Silva, Eduardo. 2019. “Feeling Race: Theorizing the Racial Economy of Emotions.” American Sociological Review 84(1):1–25.

Bracey, Glenn E. 2015. “Toward a Critical Race Theory of State.” Critical Sociology 41(3):553–572.

Casas, Andreu, Tianyi Bi, and John Wilkerson. 2018. “A Robust Latent Dirichlet Allocation Approach for the Study of Political Text.” First draft presented at Ninth Conference on New Directions in Analyzing Text as Data, Seattle, August, 2018.

Chasins, Sarah and Rastislav Bodik. 2017. “Skip Blocks: Reusing Execution History to Accelerate Web Scripts.” Proceedings of the ACM on Programming Languages 1(OOPSLA):1–28.

Crowder, Kyle, Jeremy Pais, and Scott J. South. 2012. “Neighborhood Diversity, Metropolitan Constraints, and Household Migration.” American Sociological Review 77(3):325–53.

Cryer, Jennifer. 2018. “Navigating Identity in Campaign Messaging: The Influence of Race & Gender on Strategy in U.S. Congressional Elections.” SSRN (2863215).

Denton, Nancy A. "The persistence of segregation: Links between residential segregation and school segregation." Minn. L. Rev. 80 (1995): 795.

Desmond, Matthew. 2017. “How Housing Dynamics Shape Neighborhood Perceptions.” Pp. 151–74 in Evidence and Innovation in Housing Law and Policy, edited by Lee Anne Fennell and Benjamin J. Keys. Cambridge: Cambridge University Press.

Dijk,Teun.A. van. 1993. “Principles of critical discourse analysis.” Discourse and Society. 4(2): 249–283. 

Dijk, Teun A. van. 2018. “Socio-cognitive discourse studies.” in John Flowerdew and John E. Richardson eds. The Routledge Handbook of Critical Discourse Studies. London: Routledge.

DiMaggio, Paul, Manish Nag, and David Blei. 2013. “Exploiting Affinities between Topic Modeling and the Sociological Perspective on Culture: Application to Newspaper Coverage of U.S. Government Arts Funding.” Poetics 41(6):570–606.

Egami, Naoki, Christian J. Fong, Justin Grimmer, Margaret E. Roberts, and Brandon M. Stewart. 2018. “How to Make Causal Inferences Using Texts.” ArXiv:1802.02163 [Stat.ML] 45.

Emirbayer, Mustafa and Matthew Desmond. 2015. The Racial Order. University of Chicago Press.

Fernald, Marcia. "The State of the Nation’s Housing." Cambridge, MA: Joint Center for Housing Studies at Harvard University (2019).

Golash-Boza, Tanya. 2016. “A Critical and Comprehensive Sociological Theory of Race and Racism.” Sociology of Race and Ethnicity 2(2):129–141.

Greif, Meredith. 2018. “Regulating Landlords: Unintended Consequences for Poor Tenants.” City & Community 17(3):658–74.
Guillem, Susana Martínez. 2018. “Race/ethnicity.” in John Flowerdew and John E. Richardson eds. The Routledge Handbook of Critical Discourse Studies. London: Routledge.

Hall, Matthew, Kyle Crowder, and Amy Spring. 2015. “Neighborhood Foreclosures, Racial/Ethnic Transitions, and Residential Segregation.” American Sociological Review 80(3):526–49.

Haney López, Ian. 2015. Dog Whistle Politics: How Coded Racial Appeals Have Reinvented Racism and Wrecked the Middle Class. Oxford: Oxford University Press.

Hartman, Saidiya V. 1997. Scenes of Subjection: Terror, Slavery, and Self-Making in Nineteenth-Century America. Oxford: Oxford University Press.

Hess, Christian. 2020. "Light-rail investment in Seattle: Gentrification pressures and trends in neighborhood ethnoracial composition." Urban Affairs Review 56, no. 1: 154-187.

Hess, Christian, Rebecca J. Walter, Arthur Acolin, and Sarah Chasins. 2019 "Comparing Small Area Fair Market Rents With Other Rental Measures Across Diverse Housing Markets." Cityscape 21, no. 3: 159-186.

Hogan, Bernie and Brent Berry. 2011. “Racial and Ethnic Biases in Rental Housing: An Audit Study of Online Apartment Listings.” City & Community 10(4):351–72.

Howell, Junia. 2018. “The Unstudied Reference Neighborhood: Towards a Critical Theory of Empirical Neighborhood Studies.” Sociology Compass 13(1):e12649.

Howell, Junia and Michael O. Emerson. 2018. “Preserving Racial Hierarchy amidst Changing Racial Demographics: How Neighbourhood Racial Preferences Are Changing While Maintaining Segregation.” Ethnic and Racial Studies 41(15):2770–89.

Korver-Glenn, Elizabeth. 2018. “Compounding Inequalities: How Racial Stereotypes and Discrimination Accumulate across the Stages of Housing Exchange.” American Sociological Review 83(4):627–56.

Krysan, Maria and Kyle Crowder. 2017. Cycle of Segregation: Social Processes and Residential Stratification. New York: Russell Sage Foundation.

Light, Ryan and Colin Odden. 2017. “Managing the Boundaries of Taste: Culture, Valuation, and Computational Social Science.” Social Forces 96(2):877–908.

Massey, Douglas S. and Nancy A. Denton. 1993. American Apartheid: Segregation and the Making of the Underclass. Cambridge, MA: Harvard University Press.

Murchie, Judson and Jindong Pang. 2018. “Rental Housing Discrimination across Protected Classes: Evidence from a Randomized Experiment.” Regional Science and Urban Economics 73:170–79.

Nelson, Laura K. 2017. “Computational Grounded Theory: A Methodological Framework.” Sociological Methods & Research 004912411772970.

Pager, Devah and Hana Shepherd. 2008. “The Sociology of Discrimination: Racial Discrimination in Employment, Housing, Credit, and Consumer Markets.” Annual Review of Sociology 34(1):181–209.

Quillian, Lincoln and Devah Pager. 2001. “Black Neighbors, Higher Crime? The Role of Racial Stereotypes in Evaluations of Neighborhood Crime.” American Journal of Sociology 107(3):717–67.

Ray, Victor Erik, Antonia Randolph, Megan Underhill, and David Luke. 2017. “Critical Race Theory, Afro-Pessimism, and Racial Progress Narratives.” Sociology of Race and Ethnicity 3(2):147–58.

Reskin, Barbara. 2012. “The Race Discrimination System.” Annual Review of Sociology 38(1):17–35.

Roberts, Margaret E., Brandon M. Stewart, and Dustin Tingley. 2014. “Stm: R Package for Structural Topic Models.” CRAN. R Foundation for Statistical Computing, Vienna, Austria.

Rothstein, Richard. 2017. The Color of Law: A Forgotten History of How Our Government Segregated America. New York: Liveright Publishing.

Schofield, Alexandra, Laure Thompson, and David Mimno. 2017. “Quantifying the Effects of Text Duplication on Semantic Models.” Pp. 2737–47 in. Association for Computational Linguistics.

Seamster, Louise. 2015. “The White City: Race and Urban Politics: Race and Urban Politics.” Sociology Compass 9(12):1049–65.

Seamster, Louise and Victor Ray. 2018. “Against Teleology in the Study of Race: Toward the Abolition of the Progress Paradigm.” Sociological Theory 36(4):315–42.

Sharkey, Patrick. 2013. Stuck in Place: Urban Neighborhoods and the End of Progress toward Racial Equality. Chicago: University of Chicago Press.

Spillers, Hortense J. 1987. “Mama’s Baby, Papa’s Maybe: An American Grammar Book.” Diacritics 17(2):64.

Subramanian, Subu V., Dolores Acevedo-Garcia, and Theresa L. Osypuk. 2005. “Racial Residential Segregation and Geographic Heterogeneity in Black/White Disparity in Poor Self-Rated Health in the US: A Multilevel Statistical Analysis.” Social Science & Medicine 60(8):1667–1679.

Thomas, Timothy A. 2017. “Forced Out: Race, Market, and Neighborhood Dynamics of Evictions.” Ph.D. Diss., Department of Sociology, University of Washington.

Tvinnereim, Endre, Xiaozi Liu, and Eric M. Jamelske. 2017. “Public Perceptions of Air Pollution and Climate Change: Different Manifestations, Similar Causes, and Concerns.” Climatic Change 140(3–4):399–412.

Walton, Emily. 2018. “Habits of Whiteness: How Racial Domination Persists in Multiethnic Neighborhoods.” Sociology of Race and Ethnicity 233264921881523.

Weheliye, Alexander G. 2014. Habeas Viscus: Racializing Assemblages, Biopolitics, and Black Feminist Theories of the Human. Durham, NC: Duke University Press.

Wynter, Sylvia. 2001. “Towards the Sociogenic Principle: Fanon, Identity, the Puzzle of Conscious Experience, and What It Is Like to Be ‘Black.’” National Identities and Sociopolitical Changes in Latin America 30–66.

Wynter, Sylvia. 2006. “On How We Mistook the Map for the Territory, and Reimprisoned Ourselves in Our Unbearable Wrongness of Being, of Desêtre: Black Studies toward the Human Project.” A Companion to African-American Studies 107–118.

Zuberi, Tukufu. 2011. “Critical Race Theory of Society.” Connecticut Law Review 43(5):21.

Zuberi, Tukufu and Eduardo Bonilla-Silva, eds. 2008. White Logic, White Methods: Racism and Methodology. Lanham, MD: Rowman & Littlefield Publishers.



