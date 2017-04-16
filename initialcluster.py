# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:15:41 2017

@author: Le Yan
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import pysal as ps 
import seaborn as sns
import os 
import geopandas as gpd


lsoa = pd.read_csv(os.path.join('outputs', 'cleaned_data_noinf.csv'), header=0)

    
lsoa.set_index(["mnemonic"], inplace=True)
lsoa.drop(["Unnamed: 0"], axis=1, inplace=True)
lsoa.dropna(inplace=True)
store = lsoa.columns #store old column names in case

for c in lsoa.columns.values:
    fig, ax = plt.subplots()
    fig.set_size_inches(7,4)
    sns.distplot(lsoa[c])
    fig.savefig(os.path.join('outputs','displot_' + c + '.png'))
    plt.close(fig)
    
#jon's code
from sklearn.cluster import KMeans
lsoa.drop(list(lsoa.columns[lsoa.isnull().any().values].values), axis=1, inplace=True)


k      = 7 # Number of clusters
k_var  = 'KMeans' # Variable name
kmeans = KMeans(n_clusters=k).fit(lsoa) # The process

print(kmeans.labels_) # The results

    
# Add it to the data frame
lsoa[k_var] = pd.Series(kmeans.labels_, index=lsoa.index) 

# How are the clusters distributed?
lsoa.KMeans.hist(bins=k)

colnames = ["IncomeChng", "PriceChng", "NoQual", "Qual1", "Qual2", "QualApp", "Qual3", "Qual4", "QualOther", "Group1", "Group2", "Group3", "Group4", "Group5", "Group6", "Group7", "Group8", "KMeans"]
lsoa.columns = colnames 
# Going to be a bit hard to read if 
# we plot every variable against every
# other variables, so we'll just pick a 
# few
sns.pairplot(lsoa, 
             vars=["PriceChng","Qual4", "Group1"], 
             hue=k_var, markers=".", size=3, diag_kind='kde')
###
lsoacopy = lsoa.copy(deep=True)


from scipy.stats import boxcox
for c in lsoacopy.columns.values:
    lsoacopy = lsoacopy.ix[lsoacopy[c] > 0]
col_pos = 0
for c in lsoacopy.columns:
    if lsoacopy[c].dtype.kind != 'O':
        print("Transforming " + c)
        x, _ = boxcox( lsoacopy[c] )
        nm = c.replace("/", "-")
        fig, ax = plt.subplots()
        fig.set_size_inches(7,4)
        sns.distplot(x, hist=True)
        fig.savefig(os.path.join('outputs', "Box-Cox-" + str(col_pos) + "." + nm + '.png'))
        plt.close(fig)
        col_pos += 1
        lsoacopy[c] = pd.Series(x, index=lsoacopy.index)
        
sns.set(style="whitegrid")
sns.pairplot(lsoacopy,
             vars=['PriceChng', 'Qual4', 'Group1'], 
             markers=".", size=4, diag_kind='kde')

#check skew
sk = lsoacopy.skew(axis=0, numeric_only=True)
to_drop = sk[sk >= 4].index
print("Dropping highly-skewed variables: " + ", ".join(to_drop.values))
lsoacopy.drop(to_drop.values, axis=1, inplace=True)


k      = 7 # Number of clusters
k_var  = 'KMeans' # Variable name
kmeans = KMeans(n_clusters=k).fit(lsoacopy) # The process

print(kmeans.labels_) # The results

    
# Add it to the data frame
lsoacopy[k_var] = pd.Series(kmeans.labels_, index=lsoacopy.index) 

# How are the clusters distributed?
lsoacopy.KMeans.hist(bins=k)

# Going to be a bit hard to read if 
# we plot every variable against every
# other variables, so we'll just pick a 
# few
sns.pairplot(lsoacopy, 
             vars=["PriceChng","Qual4", "Group1"], 
             hue=k_var, markers=".", size=3, diag_kind='kde')

# Quick sanity check in case something hasn't
# run successfully -- these muck up k-means
#mapping 
gdf = gpd.read_file(os.path.join('Data','Lower_Layer_Super_Output_Areas_December_2011_Generalised_Clipped__Boundaries_in_England_and_Wales.shp'))
gdf.set_index('lsoa11cd', drop=True, inplace=True)

lsoacopy.drop(list(lsoacopy.columns[lsoacopy.isnull().any().values].values), axis=1, inplace=True)

k_pref = 4
kmeans = KMeans(n_clusters=k_pref).fit(lsoacopy)
lsoacopy[k_var] = pd.Series(kmeans.labels_, index=lsoacopy.index)

sdf = gdf.join(lsoacopy, how='inner')

from pysal.contrib.viz import mapping as maps

# Where will our shapefile be stored
shp_link = os.path.join('lsoas_kde.shp')

# Save it!
sdf.to_file(shp_link)

# And now re-load the values from the DBF file 
# associated with the shapefile.
values = np.array(ps.open(shp_link.replace('.shp','.dbf')).by_col(k_var))

maps.plot_choropleth(shp_link, values, 'unique_values', 
                     title='K-Means ' + str(k_pref) + ' Cluster Analysis', 
                     savein=os.path.join('outputs', 'K-Means.png'), dpi=150, 
                     figsize=(8,6), alpha=0.9
                    )

sns.pairplot(lsoacopy, 
             vars=["PriceChng","Qual4", "Group1"], 
             hue=k_var, markers=".", size=3, diag_kind='kde')