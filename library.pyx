#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:38:40 2020

@author: thosvarley
"""
cimport cython 
import numpy as np 
cimport numpy as np
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform, pdist
from scipy.stats import zscore, entropy
import igraph as ig
from copy import deepcopy
from collections import Counter
import matplotlib.pyplot as plt
from scipy.io import loadmat
"""
in_dir = '/home/thosvarley/Data/HCP/rest/'
mat = loadmat(in_dir + '100307.mat')
X = np.vstack(np.squeeze(mat["parcel_time"])).T
"""
@cython.boundscheck(False)
@cython.wraparound(False)
def cluster_kmeans(double[:,:] X, int k):
    cdef np.ndarray cluster = k_means(X.T, k)[1]
    return cluster
"""
def cluster_nerve(X, method = "infomap"):
    dmat = squareform(pdist((X.T), metric="cosine"))
    mn = np.min(dmat)
    mx = np.max(dmat)/2
    space = np.linspace(mn, mx, 10)
    connected = False
    
    for i in range(1,space.shape[0]):
        if connected == False:
            filt = deepcopy(dmat)
            filt[filt > space[i]] = 0
            
            plt.imshow(filt)
            plt.show()
            G = ig.Graph.Weighted_Adjacency(filt.tolist())
            
            if G.is_connected() == True:
                connected = True
                
    if method == "infomap":
        comm = G.community_infomap(edge_weights = "weight")
    elif method == "walktrap":
        comm = G.community_walktrap(weights="weight").as_clustering()
    elif method == "labelprop":
        comm = G.community_label_propagation(weights = "weight")
    
    return np.array(comm.membership)
"""

@cython.boundscheck(False)
@cython.wraparound(False)
def make_transmat(long cluster, int lag=1):
    
    cdef int num_states = np.max(cluster)+1
    cdef double[:,:] mat = np.zeros((num_states, num_states))
    
    cdef list transitions = list(zip(cluster[:-lag], cluster[lag:]))
    cdef int i
    cdef double total
    
    for i in range(len(transitions)):
        mat[transitions[i][0], transitions[i][1]] += 1.0
    
    for i in range(num_states):
        total = np.sum(mat[i])
        if total != 0:
            mat[i] = mat[i]/total
        
    return mat

@cython.boundscheck(False)
@cython.wraparound(False)
def entropy_production(double[:,:] transmat):
    """
    Given a transition probability matrix (rows corresponding to out-going transitions),
    calculates the entropy production in bits. 
    """
    cdef double entropy = 0
    cdef int i, j
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[0]):
            if transmat[i][j] != 0 and transmat[j][i] != 0:
                entropy += transmat[i][j] * np.log2(transmat[i][j] / transmat[j][i])
    
    return entropy 

@cython.boundscheck(False)
@cython.wraparound(False)
def determinism(double[:,:] transmat):
    
    cdef double N = transmat.shape[0]
    cdef double det = 0
    cdef int i 
    
    for i in range(transmat.shape[0]):
        det += (entropy(transmat[i], base=2))

    return np.log2(N) - (det/N)

@cython.boundscheck(False)
@cython.wraparound(False)
def degeneracy(double[:,:] transmat):
    
    cdef double N = transmat.shape[0]
    cdef double[:] avg = np.mean(transmat, axis=0)
    cdef double deg = entropy(avg, base=2)
    
    return np.log2(N) - deg

def mutual_information(X, Y):
    
    mx = np.max((X,Y))
    joint_space = np.histogram2d(X, Y, bins = mx+1)[0] / X.shape[0]
    joint_ent = entropy(joint_space.flatten(), base=2)
    
    X_counts = Counter(X).values()
    Y_counts = Counter(Y).values()
    
    X_ent = entropy(list(X_counts), base=2)
    Y_ent = entropy(list(Y_counts), base=2)
    
    return X_ent + Y_ent - joint_ent


def auto_mutual_information(cluster, max_lag):
        
    auto_mi = np.zeros(max_lag)
    cluster_counts = list(Counter(cluster).values())
    auto_mi[0] = entropy(cluster_counts, base=2)
    
    for l in range(1,max_lag):
        auto_mi[l] = mutual_information(cluster[:-(l)], cluster[(l):])
        
    return auto_mi


'''
def local_flux(x, y, probmat, transitions):
    """
    A utility function for use in flux(). 
    Given a point (x, y), the associated probability matrix, and the list of points,
        the function returns the average flux u(x, y) for that point. 
    """
    
    flux_vector = np.zeros(2)
    
    #W_(x-1,y),(x,y)
    n_ij = len([p for p in transitions if p[0] == (x-1, y) and p[1] == (x, y)])
    n_ji = len([p for p in transitions if p[0] == (x, y) and p[1] == (x-1, y)])
    
    flux_vector[0] += ((n_ij - n_ji) / len(transitions))

    #W_(x,y),(x+1,y)
    n_ij = len([p for p in transitions if p[0] == (x, y) and p[1] == (x+1, y)])
    n_ji = len([p for p in transitions if p[0] == (x+1, y) and p[1] == (x, y)])
    
    flux_vector[0] += ((n_ij - n_ji) / len(transitions))
    
    #W_(x,y-1),(x,y)
    n_ij = len([p for p in transitions if p[0] == (x, y-1) and p[1] == (x, y)])
    n_ji = len([p for p in transitions if p[0] == (x, y) and p[1] == (x, y-1)])
    
    flux_vector[1] += ((n_ij - n_ji) / len(transitions))

    #W_(x,y),(x,y-1)
    n_ij = len([p for p in transitions if p[0] == (x, y) and p[1] == (x, y+1)])
    n_ji = len([p for p in transitions if p[0] == (x, y+1) and p[1] == (x, y)])
    
    flux_vector[1] += ((n_ij - n_ji) / len(transitions))    
    
    return (1/2) * flux_vector
    

def flux(X, nbins = 50):
    """
    Given a multi-dimensional time-series X, flux() calculates the first two principle components,
    and then plots the flow around an nbins x nbins digital surface. 
    This function isn't quite ready to go yet.
    Also plotting this is a beast. 
    """
    zX = zscore(X)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(zX.T).T
    
    mn_x, mx_x = np.min(comps[0]), np.max(comps[0])
    mn_y, mx_y = np.min(comps[1]), np.max(comps[1])
    
    grid_x = np.digitize(comps[0], bins=np.linspace(mn_x, mx_x, num=nbins))
    grid_y = np.digitize(comps[1], bins=np.linspace(mn_y, mx_y, num=nbins))
    
    points = list(zip(grid_x, grid_y))
    probmat = np.zeros((nbins+1, nbins+1))
    
    for i in range(grid_x.shape[0]):
        probmat[points[i][0], points[i][1]] += 1/grid_x.shape[0]
    
    transitions = list(zip(points[:-1], points[1:]))
    
    fluxes = []
    
    for i in range(probmat.shape[0]):
        for j in range(probmat.shape[1]):
            fluxes.append(local_flux(i, j, probmat, transitions))
    
    return fluxes
'''