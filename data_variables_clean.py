# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 14:03:05 2017

@author: Le Yan
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os 


if os.path.isdir('outputs') is not True: #create directory for output images
    os.mkdir('outputs')
    
#load in cleaned dataset from CW1
data = pd.read_csv(os.path.join('Data', 'cleaned_data_noinf.csv'), header=0)

#Load in raw house prices and initial income levels 
hprice = pd.read_csv(os.path.join('Data', 'house-prices-LSOAs.csv'), header=0)
hhincome = pd.read_csv(os.path.join('Data', 'modelled-household-income-estimates-lsoa.csv'), header=0)

#clean data of ascii values // code taken from initial analysis files 
#1. house price 
def strip_non_ascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

hprice.columns = [strip_non_ascii(x) for x in hprice.columns]
hprice.columns = pd.Series(hprice.columns).str.replace("\(\)-", "").str.strip() #remove parantheses and dashes for easier column access
hprice = hprice.replace('.', np.nan)
for c in hprice.columns[3:24].values:  #convert values from obj to float
    hprice[c] = hprice[c].astype(float)
    
#2. hhincome
for c in hhincome.columns.values:            #remove all non-ascii characters from dataframe
    hhincome[c] = hhincome[c].apply(lambda x: x.decode('unicode_escape').\
                                                encode('ascii', 'ignore').\
                                                strip())
for c in hhincome.columns[4:].values:  #convert values from obj to float
    hhincome[c] = hhincome[c].str.replace(',', '').astype(float)
    
#keep only wanted columns
hprice12 = hprice[['Lower Super Output Area', 'Median2012']].copy()
hhincome12 = hhincome[['Code', 'Median 2012/13']].copy()

del hhincome, hprice #delete unneeded dataframes

#merge then drop unnecessary columns
data = data.merge(hprice12, left_on='mnemonic', right_on='Lower Super Output Area').merge(hhincome12, left_on='mnemonic', right_on='Code')
data.drop(["Unnamed: 0", "Lower Super Output Area", "Code"], axis=1, inplace=True)

data.dropna(inplace=True) #drop rows with missing data

#Possibly required variables: tenancy (Hamnett (2003) mentions housing in inner-city turning from rented to owned)
tenure = pd.read_csv(os.path.join('Data', "tenure.csv"), header=6, skip_blank_lines=True, engine='python')
tenure_pct = tenure[['mnemonic', '%.1', '%.2', '%.3', '%.4', '%.5']].copy()
tenure_pct.columns = ['mnemonic', 'Owned %', 'Shared %', 'Social Rented %', 'Private Rented %', 'Rent Free %']
cols = ['Owned %', 'Shared %', 'Social Rented %', 'Private Rented %', 'Rent Free %']
for c in cols[0:]:        #convert to decimal proportion
    tenure_pct[c] = tenure_pct[c] / 100

data = data.merge(tenure_pct, on='mnemonic')

"""
#Other possibly required variables: change in profession type and education levels in boroughs
#1. qualifications 
qual11 = pd.read_csv(os.path.join('Data', '246541426 - Qualifications.csv'), header=6, skip_blank_lines=True, engine='python')
qual01 = pd.read_csv(os.path.join('Data', '246541426 - Qualifications 2001.csv'), header=5, skip_blank_lines=True, engine='python')
#drop missing col from qual11
qual11.drop(['Highest level of qualification: Apprenticeship'], axis=1, inplace=True)
#calc prop. for qual10
new_cols = ['No qual','level 1', 'level 2', 'level 3', 'level 4', 'other qual']
qual01.rename(columns=dict(zip(qual01.columns[1:7], new_cols)), inplace=True) 
for cols in qual01.columns[1:7]:
    qual01[cols + "_pct"] = qual01[cols].astype(float) / qual01["All people aged 16-74"].astype(float)
#make df of pct change
"""

#save final file for later 
data.to_csv(os.path.join('outputs', '20170416_cleaned.csv'))
#Step 2 - Standardising/Normalising/Transforming
data.set_index('mnemonic', drop=True, inplace=True)
data.index.name = None
new_cols = ['level 1', 'level 2', 'apprenticeship', 'level 3', 'level 4', 'other qual']
data.rename(columns=dict(zip(data.columns[3:9], new_cols)), inplace=True) #renaming cols or creating files bring issues
#taken from practical 3 - checking distribution of info
col_pos=0
for c in data.columns.values:
    print("Creating chart for " + c)
    nm = c.replace("/", "-")
    fig, ax = plt.subplots()
    fig.set_size_inches(7,4)
    sns.distplot(data[c])
    fig.savefig(os.path.join('outputs', "Untransformed-" + str(col_pos) + "." + nm + '.png'))
    plt.close(fig)
    col_pos += 1
    