import numpy as np 
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from tqdm import tqdm


# Inspection of Conformal Manifold
def triangular_plot(chains,save='None',figsize=(10,10),names=None):
    data=chains
    nsteps,ndim=chains.shape
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=ndim, ncols=ndim, hspace=0, wspace=0)
    axs = gs.subplots()

    for i in range(ndim):
        for j in range(ndim):
            if i<j:
                axs[i,j].axis('off')
            elif i==j:
                counts, bins, _ = axs[i,j].hist(data[:,i], 100, color="k", histtype="step")
                # axs[i,j].yaxis.set_label_position("right")
                axs[i,j].yaxis.tick_right()
                axs[i,j].set_ylim(bottom=-0.06*max(counts))
                if not (i==ndim-1):
                    axs[i,j].sharex(axs[-1,j])
                    plt.setp(axs[i,j].get_xticklabels(), visible=False)
                else:
                    axs[i,j].set_xlim(min(data[:,i]),max(data[:,i]))
                    axs[i,j].set_xlabel(str(names[i]))
            else:
                counts,xbins,ybins,image = axs[i,j].hist2d(data[:,j],data[:,i],bins=100,norm=LogNorm(),cmap = plt.cm.rainbow)
                axs[i,j].contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],linewidths=0.5, cmap = plt.cm.rainbow, levels = [1,100,1000,10000])
                if not (i == ndim-1) :
                    axs[i,j].sharex(axs[-1,j])
                    plt.setp(axs[i,j].get_xticklabels(), visible=False)
                if not (j == 0) :
                    axs[i,j].sharey(axs[i,0])
                    plt.setp(axs[i,j].get_yticklabels(), visible=False)
                    axs[i,j].tick_params(axis='y', which='both', left=False)
                if i == ndim-1 :
                    axs[i,j].set_xlabel(str(names[j]))
                if j == 0 :
                    axs[i,j].set_ylabel(str(names[i]))
                axs[i,j].locator_params(axis='both', nbins=3)
    
    cax = axs[-1,0].inset_axes([-0.6, (ndim-1)/4, 0.1, (ndim-1)/2])
    fig.colorbar(image, cax=cax)
    cax.yaxis.tick_left()

    if save != 'None':
        plt.savefig(save, bbox_inches="tight")
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
