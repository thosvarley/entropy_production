#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:41:28 2020

@author: thosvarley
"""
import numpy as np 
import matplotlib.pyplot as plt 
from library import cluster_nerve, cluster_kmeans, make_transmat, entropy_production

"""
This script will walk you through how to use the functions in the library 
to calcuate the entropy production of a time-series, given some clustering 
schema.

Assume you have 20 fMRI scans, each with 10 regions and 50 TRs (random choices)
and that you have already loaded them into a list. It should look like:
    
    data = [subject_0, subject_1 ... subject_19]

Each dataset is assumed to be in regions x TRs shape. It will mess up if you 
have it transposed. 
"""

#For data-storage reasons, I am generating 20 random signals from scratch
#from a uniform distribution on the interval [0,1).
data = [np.random.randn(10,50) for x in range(20)]

#Following Lynn et al., (2020) we can either concatenate all the scans...
concat = np.hstack(data)
#...or operate on each one individually, in which case you wrap everything in a for-loop
#and iterate over the elements of data. 
#I'll assume you chose to concat, for ease on demonstration. 

"""
First thing you have to do is cluster your data. You have two choices:
    
    cluster_kmeans(X, k) will use k-means clustering to perform the embedding.
        It requires pre-selecting k (as always). Lynn suggested using the largest k
        such that all transitions were observed once (i.e. in the transition probabilit
        matrix there is no M[x][y] = 0).
    
    cluster_nerve(X) will perform a Rips filtration on the data until the simplicial 
        complex becomes connected, and then use infomap to construct clusters from that. 
        It does not require free parameters but takes a *lot* longer. Also for very long 
        time series (i.e. if you choose to concat) it might blow up your computer. 
        (I plan to optimize this a little bit). 
    
Both of these functions return the same struture: a 1-dimensional numpy array 
where the value of the ith element is the state that the ith TR is assigned to. 
"""

#For speed purposes I'm assuming you choose k-means and and a k of 6 
cluster = cluster_kmeans(concat, 6)
    
#From that cluster array, you make the transition probability matrix
transmat = make_transmat(cluster)
#Note that the transmat has been normalized so that all out-going edges define
#a probability distribution.
print(transmat.sum(axis=1))
plt.imshow(transmat)
plt.colorbar(label="P(future | present)")
plt.xlabel("Future State")
plt.ylabel("Present State")
#And from there you calculate the entropy production
ent = entropy_production(transmat)

print(ent, "bit")
    

