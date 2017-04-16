# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:52:26 2017

@author: Le Yan
"""

#C:\Users\Le Yan\Documents\KCL 2016-17\Applied Geocomputation\Assignment 1
import pysal as ps 
import numpy as np 
import pandas as pd
import geopandas as gpd 
import seaborn as sns 
import matplotlib.pyplot as plt
import os


if os.path.isdir('outputs') is not True: #create directory for output images
    os.mkdir('outputs')

#Read files 
qualifications = pd.read_csv(os.path.join('Data', '246541426 - Qualifications.csv'), header=6, skip_blank_lines=True, skipfooter=7, engine='python')
job = pd.read_csv(os.path.join('Data', '2570418033 - NSSeC.csv'), header=5, skip_blank_lines=True, skipfooter=7, engine='python')
hprice = pd.read_csv(os.path.join('Data', 'house-prices-LSOAs.csv'), header=0, skip_blank_lines=True, engine='python')
hhincome = pd.read_csv(os.path.join('Data', 'modelled-household-income-estimates-lsoa.csv'), header=0, skip_blank_lines=True, engine='python')

df_pct = pd.concat([qualifications['mnemonic']], axis=1, keys=['mnemonic']) #initiate new df to store proportions 
###Cleaning data
#1. Qualifications
#drop LSOA name column (is this method more reproducible?)
if np.where(qualifications.columns.values=='2011 super output area - lower layer')[0] >= 0:
    qualifications = qualifications.drop('2011 super output area - lower layer', 1)
qualifications.columns #check columns

for c in qualifications.columns[1:8].values: #append proportions to df_pct
    df_pct[c] = pd.Series(qualifications[c].astype(float)/qualifications["All categories: Highest level of qualification"].astype(float))
#2. Jobs
#I prefer this method
job.drop(["2011 super output area - lower layer", "L15 Full-time students"], axis=1, inplace=True)
for c in job.columns[1:9].values:
    df_pct[c] = pd.Series(job[c].astype(float)/job["All categories: NS-SeC"].astype(float))

#3. Median house price
#clean up columns
def strip_non_ascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

hprice.columns = [strip_non_ascii(x) for x in hprice.columns]
hprice.columns = pd.Series(hprice.columns).str.replace("\(\)-", "").str.strip() #remove parantheses and dashes for easier column access
hprice = hprice.replace('.', np.nan)
for c in hprice.columns[3:24].values:  #convert values from obj to float
    hprice[c] = hprice[c].astype(float)

#calculate % change in house price 
hprice = hprice[np.isfinite(hprice['Median2001'])] #remove rows with NaN values in 2001
#Adjust for inflation
#http://inflation.stephenmorley.org/
#According to the ONS, the inflation for index in 2001-2012 is 1.62. Hence 100 pounds back in 2001 would be worth 100x1.62 in 2012
hprice['Median2001Inf'] = hprice['Median2001'] * 1.62
hprice['Price Change'] = ((hprice['Median2012'].astype(float) - hprice['Median2001'].astype(float))/ hprice['Median2001'].astype(float))
hpricechange = hprice[['Price Change', 'Lower Super Output Area']].copy() #make new dataframe of only columns I want to merge
df_pct = pd.merge(hpricechange, df_pct, left_on="Lower Super Output Area", right_on="mnemonic", how="outer")

#4. Household Income                         
for c in hhincome.columns.values:            #remove all non-ascii characters from dataframe
    hhincome[c] = hhincome[c].apply(lambda x: x.decode('unicode_escape').\
                                                encode('ascii', 'ignore').\
                                                strip())
for c in hhincome.columns[4:].values:  #convert values from obj to float
    hhincome[c] = hhincome[c].str.replace(',', '').astype(float)
    
hhincome = hhincome[np.isfinite(hhincome["Median 2001/02"])] #check for nan values...looks like it's fine
hhincome['Median2001IncomeInf'] = hhincome["Median 2001/02"] * 1.62
        #Looks like taking inflation into account is a bad idea...everyone's income dropped!
hhincome['Income Change'] = ((hhincome['Median 2012/13'].astype(float) - hhincome['Median 2001/02'].astype(float)) / hhincome['Median 2001/02'].astype(float))

hhincomechange = hhincome[['Income Change', 'Code']]
#merge
df_pct = pd.merge(hhincomechange, df_pct, left_on="Code", right_on="mnemonic", how='outer')

#drop inneeded columns
df_pct.drop(["Code", "Lower Super Output Area"], axis=1, inplace=True)
#df_pct is now a nice clean table of data. yay.
#save it in case
df_pct.to_csv(os.path.join('outputs', 'cleaned_data.csv'))

#5. Exploratory Analysis
#Pretty much just copying practical codes here
df_pct = df_pct.set_index(['mnemonic'])
gdf = gpd.read_file(os.path.join('Data','Lower_Layer_Super_Output_Areas_December_2011_Generalised_Clipped__Boundaries_in_England_and_Wales.shp'))
gdf.set_index('lsoa11cd', drop=True, inplace=True)

sdf = gdf.join(df_pct, how='inner')
shp_link = os.path.join('lsoas.shp')
sdf.to_file(shp_link) #Oh. So this is how you convert your df back to a shpfile. This would've been useful to know earlier...

sdf.plot(column='Income Change', cmap='OrRd', scheme='quantiles') #Example of how to plot
g = sns.pairplot(df_pct)
