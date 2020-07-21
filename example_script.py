#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:41:28 2020

@author: thosvarley
"""
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style="white")
from library import cluster_nerve, cluster_kmeans, make_transmat, entropy_production, determinism, degeneracy

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

k_range = [x for x in range(4,9)] #The range of K-values you sweep in the KNN
l_range = [x for x in range(1,25)] #The range of lags you sweep. 

k_list = [[] for x in range(len(l_range))] #A 2D list, with one sub-list for each lag. Each sublist will have one value for each K in ascending order.
deg_list = [[] for x in range(len(l_range))] #Same for determinism
det_list = [[] for x in range(len(l_range))] #Same for degeneracy

for l in range(len(l_range)): #The transmat function has a lag parameter - how far back in time is the "memory" of the system?
    for k in range(4,9):
        #For speed purposes I'm assuming you choose k-means and and a k of 6 
        cluster = cluster_kmeans(concat, k)
            
        #From that cluster array, you make the transition probability matrix
        transmat = make_transmat(cluster)
        #Note that the transmat has been normalized so that all out-going edges define
        #a probability distribution.
        
        """
        Uncomment to see a visualized TPM.
        
        plt.imshow(transmat)
        plt.colorbar(label="P(future | present)")
        plt.xlabel("Future State")
        plt.ylabel("Present State")
        plt.show()
        """
        #And from there you calculate the entropy production...
        k_list[l].append(entropy_production(transmat))
        
        #...the degeneracy...
        deg_list[l].append(degeneracy(transmat, norm=True))
        
        #...and the determinism. The effectiveness is just det - deg.
        det_list[l].append(determinism(transmat, norm=True))
        print(k)

#Turning everything into numpy arrays for plt.imshow()
k_array = np.array(k_list)
deg_array = np.array(deg_list)
det_array = np.array(det_list)

sns.set(style="darkgrid") #Making sure the styles are right. 
"""
Plots the entropy production for each lag, over the values of K. Sort of like collapsing a 
#manifold down like an accordian. 
"""
for i in range(len(l_range)):
    plt.plot(k_range, k_list[i], label = l_range[i])
plt.xlabel("K")
plt.ylabel("Bit")
plt.show()

sns.set(style="white")
"""
A better visualization of the manifold. An lag x k plane. Color is height. 
"""
plt.imshow(k_array)
plt.xticks([x for x in range(len(k_range))], k_range)
plt.xlabel("K")
plt.yticks([x for x in range(len(l_range))], l_range)
plt.ylabel("Lag")
plt.colorbar(label="Bits")
plt.show()

plt.imshow(deg_array)
plt.xticks([x for x in range(len(k_range))], k_range)
plt.xlabel("K")
plt.yticks([x for x in range(len(l_range))], l_range)
plt.ylabel("Lag")
plt.colorbar(label="Bits")
plt.show()

sns.set(style="darkgrid")
"""
Compare entropy production and degeneracy/effectiveness.
"""
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
for i in range(len(k_range)):
    plt.scatter(deg_array[:,i], k_array[:,i], label=k_range[i])
plt.legend(title="K")
plt.xlabel("Normalized Degeneracy")
plt.ylabel("Entropy Production")

plt.subplot(1,2,2)
for i in range(len(k_range)):
    plt.scatter((det_array[:,i] - deg_array[:,i]), k_array[:,i], label=k_range[i])
plt.legend(title="K")
plt.xlabel("Normalized Det. - Deg.")
plt.ylabel("Entropy Production")