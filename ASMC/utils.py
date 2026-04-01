import numpy as np 
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from tqdm import tqdm


# Inspection of Conformal Manifold
def triangular_plot(chains,save='None',xlim='None',ylim='None',figsize=(25,25),names=None):
    data=chains
    nsteps,ndim=chains.shape
    fig = plt.figure(figsize=figsize)
    fig.set(facecolor = "white")
    for i in range(ndim):
        ax = fig.add_subplot(ndim,ndim,i*ndim+i+1)
        ax.hist(data[:,i], 100, color="k", histtype="step")
        if names == 'None':
            ax.set_title(f"x{i+1}")
        else: 
            ax.set_title(str(names[i]))
    for i in range(ndim):
        for j in range(i):
            plt.subplot(ndim,ndim,ndim*i+j+1)
            counts,xbins,ybins,image = plt.hist2d(data[:,j],data[:,i],bins=100
                                      ,norm=LogNorm()
                                      ,cmap = plt.cm.rainbow)
            plt.colorbar()
            plt.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
            linewidths=0.5, cmap = plt.cm.rainbow, levels = [1,100,1000,10000])
            if not ylim == "None": 
                plt.ylim(ylim)
            if not xlim == "None":
                plt.xlim(xlim)
    if save != 'None':
        plt.savefig(save,transparent=False)
        plt.show()
    else: 
        plt.show()

def triangular_plot_slopes(chains,save='None',xlim='None',ylim='None'):
    data=chains
    nsteps,ndim=chains.shape
    fig = plt.figure(figsize=(20,20))
    fig.set(facecolor = "white")
    for i in range(ndim):
        for j in range(i):
            ax=fig.add_subplot(ndim,ndim,ndim*i+j+1)
            #those_slope0=np.extract(np.abs(data[:,0])>0.2,data[:,i]/data[:,j])
            those_slope0=data[:,i]/data[:,j]
            those_slope=np.extract(np.abs(those_slope0)<10,those_slope0)
            ax.hist(those_slope,bins=100)
            ax.set_title(f"x{j+1}/x{i+1}")
            if not ylim == "None": 
                ax.ylim(ylim)
            if not xlim == "None":
                ax.xlim(xlim)
            #ax.set_ylabel(f"x{i}")
    if save != 'None':
        plt.savefig(save,transparent=False)
        plt.show()
    else: 
        plt.show()


# Local Analysis
def local_dim_1_point(x, var_thres=0.99):
    """Calcule la dimension locale en utilisant PCA."""
    pca = PCA()
    pca.fit(x)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    return np.argmax(cumulative_variance >= var_thres) + 1

def compute_local_dim(i, data, neighbors_idx, var_thres, verbose):
    """Computation of local dimension of a given point."""
    local_neighbors = data[neighbors_idx[i]]
    dim = local_dim_1_point(local_neighbors, var_thres=var_thres)
    if verbose >= 2 and i % 100 == 0:
        print(f"Point {i}: Estimated Local Dimension = {dim}")
    return [dim, i]

def local_dim_n_points(data, verbose=0, n_neig=20, var_thres=0.99, n_jobs=-1):
    """"Compute the local dimension for all points."""
    n_points = data.shape[0]

    # Recherche des voisins avec KD-Tree
    if verbose >= 1:
        print("Searching for neighbors using KD-Tree...")
    nbrs = NearestNeighbors(n_neighbors=n_neig, algorithm='kd_tree').fit(data)
    _, neighbors_idx = nbrs.kneighbors(data)

    # Calcul parallèle avec barre de progression
    if verbose >= 1:
        print("Computing the local dimensions...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_local_dim)(i, data, neighbors_idx, var_thres, verbose)
        for i in tqdm(range(n_points), desc="Progression", disable=(verbose <= 2))
    )

    if verbose >= 1:
        print("Computation finished!")

    return results
