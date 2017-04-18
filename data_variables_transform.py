# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:16:44 2017

@author: Le Yan
"""
import pysal as ps
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import seaborn as sns
import os 
import geopandas as gpd
import clusterpy as cpy
import sklearn

#read cleaed data
data = pd.read_csv(os.path.join('outputs', '20170416_cleaned.csv'), header=0)

#doing some initial cleaning
data.set_index('mnemonic', drop=True, inplace=True)
data.index.name = None
data.drop(["Unnamed: 0"], axis=1, inplace=True)

#house price and income level area  bit confusing
newcols = ['hprice12', 'hhincome12']
data.rename(columns=dict(zip(data.columns[17:19], newcols)), inplace=True) 
#renaming cols or creating files bring issues
new_cols = ['level 1', 'level 2', 'apprenticeship', 'level 3', 'level 4', 'other qual']
data.rename(columns=dict(zip(data.columns[3:9], new_cols)), inplace=True) 

#Time to clean data
#remove values which are not % - to re-add later
#also z-score standardise raw data as per Mohamad and Usman (2013)
hhincomeprice = data[["hprice12", "hhincome12"]].copy()
data.drop(newcols, axis=1, inplace=True)
for col in hhincomeprice.columns: 
    hhincomeprice[col + '_std'] = (hhincomeprice[col] - hhincomeprice[col].mean())/hhincomeprice[col].std()
hhincomeprice.drop(newcols, axis=1, inplace=True) 
#1. Remove skew
sk = data.skew(axis=0, numeric_only=True)
dropped = sk[sk >= 4].index
data.drop(dropped.values, axis=1, inplace=True)

numeric_cols = [col for col in data if data[col].dtype.kind != 'O']
data[numeric_cols] += 1

#2.Normalise, as per Brunsdon and Singleton (2015)

from scipy.stats import boxcox

for col in data.columns:
    if data[col].dtype.kind !="O": #checking if data type is not an object
        x, _= boxcox(data[col])
        fig, ax = plt.subplots()
        fig.set_size_inches(7,4)
        sns.distplot(x, hist=True)
        fig.savefig(os.path.join('outputs', 'Box-cox-' + str(col) + '.png'))
        plt.close(fig)

#re-add columns
data  = pd.concat([data, hhincomeprice], axis=1)
del hhincomeprice

#3. Remove correlations
###From this point the code is retrieved from JR's lecture 3###
data.corr()


corrs    = 0.50 #strongly correlated
corrh    = 0.70 #highly correlated

# Generate the matrix but capture the output this time
corrm = data.corr()
corrm['name'] = corrm.index # We need a copy of the index

num_corrs = []
hi_corrs  = []

for c in corrm.columns:
    if c != 'name':
        hits = corrm.loc[(corrm[c] >= corrs) & (corrm[c] < 1.0), c]
        print("=" * 20 + " " + c + " " + "=" * 20)
        print("Strongly correlated with : ")
        if hits.size > 0: 
            print("\t" + ", ".join(hits.index.values))
            num_corrs.append(hits.size)
            
            if hits[ hits > corrh ].size > 1:
                print("Highly correlated with: ")
                print("\t" + ", ".join(hits[ hits > corrh ].index.values))
                hi_corrs.append(hits[ hits > corrh ].size)

maxcorrs = 3 # What's our threshold for too many strong correlations?

to_drop = [] # Columns to drop
to_keep = [] # Columns to keep

for c in corrm.columns:
    if c != 'name':
        
        hits = corrm.loc[(corrm[c] >= corrs) & (corrm[c] < 1.0), c]
        
        print("=" * 12 + " " + c + " " + "=" * 12)
        print(hits)
        print(" ")
        
        hi_vals    = False
        multi_vals = False
        
        # Remove ones with very high correlations
        if hits[ hits > corrh ].size > 0:
            print(">>> Very high correlation...")
            s1 = set(to_keep)
            s2 = set(hits[ hits > corrh ].index.values)
            #print("Comparing to_keep(" + ", ".join(s1) + ") to hits(" + ", ".join(s2) + ")")
            s1 &= s2
            #print("Column found in 'very high correlations': " + str(s1))
            if len(s1) > 1: 
                hi_vals = True
                print("Will drop '" + c + "' because of very high correlation with retained cols: \n\t" + "\n\t".join(s1))
        
        # Remove ones with many correlations
        if hits.size >= maxcorrs: 
            print(">>> Many correlations...")
            s1 = set(to_keep)
            s2 = set(hits.index.values)
            #print("Comparing to_keep(" + ", ".join(s1) + ") to hits(" + ", ".join(s2) + ")")
            s1 &= s2
            #print("Column found in 'many correlations' :" + str(s1))
            if len(s1) > 1: 
                multi_vals = True
                print("Will drop '" + c + "' because of multiple strong correlations with retained cols: \n\t" + "\n\t".join(s1))
        
        if hi_vals==True or multi_vals==True:
            to_drop.append(c)
        else:
            to_keep.append(c)

print(" ")
print("To drop: " + ", ".join(to_drop))
print(" ")
print("To keep: " + ", ".join(to_keep))
######
#From this point, it seems that raw household income and housep price are not required
#As in Voas and Williamson (2000) we drop correlated values 'because they are partially predictable
#on the basis of [other variables]' 
#house price is indicated by group 1 occupation, high rate of house income change and high level of education
#household income is indicated by group 1-2 occupation and high level of education

data_clean = data.drop(to_drop, axis=1)

#4. Standardise
sns.pairplot(data_clean, vars=["Income Change", "Private Rented %", "level 4"], markers='.', size=4, diag_kind='kde')
data_std = data_clean.copy(deep=True)
#try z-standardising
#As per Brunsdon and Singleton (2015). Mohamad and Usman (2013) suggest that z-score standardisation
#is the most robust scaling standardisation method
for col in data_std.columns:
    data_std[col] = (data_std[col] - data_std[col].mean())/data_std[col].std()
    
sns.pairplot(data_std, vars=["Income Change", "Private Rented %", "1. Higher managerial, administrative and professional occupations"], markers='.', size=4, diag_kind='kde')

#5. determine optimal clusters with silhouette plot
from sklearn.cluster import KMeans   
from sklearn.metrics import silhouette_samples, silhouette_score 
#http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
for k in range(3,10):
    # Debugging
    print("Cluster count: " + str(k))
    
    #############
    # Do the clustering using the main columns
    clusterer = KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusterer.fit_predict(data_std)
    
    # Calculate the overall silhouette score
    silhouette_avg = silhouette_score(data_std, cluster_labels)
    print("For k =", k,
          "The average silhouette_score is :", silhouette_avg)
    
    # Calculate the silhouette values
    sample_silhouette_values = silhouette_samples(data_std, cluster_labels)
    
    #############
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(9, 5)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1
    ax1.set_xlim([-1.0, 1.0]) # Changed from -0.1, 1
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, data_std.shape[0] + (k + 1) * 10])
    
    y_lower = 10
    
    # For each of the clusters...
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        # Set the color ramp
        color = cm.spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks(range(-1, 1, 2)) # Was: [-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]

    # 2nd Plot showing the actual clusters formed --
    # we can only do this for the first two dimensions
    # so we may not see fully what is causing the 
    # resulting assignment
    colors = cm.spectral(cluster_labels.astype(float) / k)
    ax2.scatter(data_std[data_std.columns[0]], data_std[data_std.columns[1]], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("Visualization of the clustered data")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % k),
                 fontsize=14, fontweight='bold')

    plt.show()

#Silhouette analysis determines thatclusters higher than 5 tend not to be very robust.
#3-5 are generally in the same range of optimal clustering
#looking at the silhouette plots, it seems that 4 is the optimal number of clusters


#6. k-means clustering
gdf = gpd.read_file(os.path.join('Data','Lower_Layer_Super_Output_Areas_December_2011_Generalised_Clipped__Boundaries_in_England_and_Wales.shp'))
gdf.set_index('lsoa11cd', drop=True, inplace=True)
data_std.drop(list(data_std.columns[data_std.isnull().any().values].values), axis=1, inplace=True)


k_pref = 4
k_var  = 'KMeans' 

kmeans = KMeans(n_clusters=k_pref).fit(data_std)
data_std[k_var] = pd.Series(kmeans.labels_, index=data_std.index)

sdf = gdf.join(data_std, how='inner')

from pysal.contrib.viz import mapping as maps

# Where will our shapefile be stored
shp_link = os.path.join('outputs', 'lsoas_kde.shp')

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

#save pickle for later analysis
data_std.to_pickle(os.path.join("outputs","clusters.pickle"))
