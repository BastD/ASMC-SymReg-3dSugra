import itertools
import sympy as sp
from tqdm import tqdm
import numpy as np 
import os 
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib import colors 
from matplotlib.ticker import PercentFormatter 
from sklearn.decomposition import PCA

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

######## local dimensions ########

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from tqdm import tqdm
import time

def local_dim_1_point(x, var_thres=0.99):
    """Calcule la dimension locale en utilisant PCA."""
    pca = PCA()
    pca.fit(x)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    return np.argmax(cumulative_variance >= var_thres) + 1

def compute_local_dim(i, data, neighbors_idx, var_thres, verbose):
    """Calcule la dimension locale pour un point donné."""
    local_neighbors = data[neighbors_idx[i]]
    dim = local_dim_1_point(local_neighbors, var_thres=var_thres)
    if verbose >= 2 and i % 100 == 0:
        print(f"Point {i}: Dimension locale estimée = {dim}")
    return [dim, i]

def local_dim_n_points(data, verbose=0, n_neig=20, var_thres=0.99, n_jobs=-1):
    """Calcule la dimension locale pour tous les points avec affichage de la progression."""
    n_points = data.shape[0]

    # Recherche des voisins avec KD-Tree
    if verbose >= 1:
        print("Recherche des voisins avec KD-Tree...")
    nbrs = NearestNeighbors(n_neighbors=n_neig, algorithm='kd_tree').fit(data)
    _, neighbors_idx = nbrs.kneighbors(data)

    # Calcul parallèle avec barre de progression
    if verbose >= 1:
        print("Calcul des dimensions locales...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_local_dim)(i, data, neighbors_idx, var_thres, verbose)
        for i in tqdm(range(n_points), desc="Progression", disable=(verbose <= 2))
    )

    #results = Parallel(n_jobs=n_jobs)(
    #    delayed(compute_local_dim)(i, data, neighbors_idx, var_thres, verbose)
    #    for i in range(n_points)
    #)

    if verbose >= 1:
        print("Calcul terminé !")

    return results

#def local_dim_1_point(x,verbose = 0, var_thres = 0.95):
#    data = x
#
#    #scaler = StandardScaler()
#
#    # Do the PCA
#    pca = PCA()
#    pca.fit(data)
#
#    # Explained std for each direction
#    explained_variance_ratio = pca.explained_variance_ratio_
#
#    # Summed std
#    cumulative_variance = np.cumsum(explained_variance_ratio)
#
#    if verbose == 1:
#        print("Variance expliquée par chaque composante :", explained_variance_ratio)
#        print("Variance cumulée :", cumulative_variance)
#
#    # Underlying dimension
#    dimension_sous_jacente = np.argmax(cumulative_variance >= var_thres) + 1
#    if verbose == 1:
#        print(f"Dimension sous-jacente estimée ({var_thres*100}% de variance) : {dimension_sous_jacente}")
#    return dimension_sous_jacente
#
#def local_dim_n_points(data, verbose = 0, min_dist = 0.05,increment_dist = 0.05, n_neig = 20,var_thres=0.99):
#    n_points = data.shape[0]
#    all_local_dim = []
#    for i in range(n_points):
#        if i % 100 == 0 and verbose == 2 :
#            print(i)
#        elif i % 1000 == 0 and verbose == 1 :
#            print(i)
#        this_dist = min_dist
#        n_neigh_bool = True
#        while n_neigh_bool:
#            this_mask = (np.sum((data - data[i,:]) ** 2,axis=1)**.5)<this_dist
#            if np.sum(this_mask) < n_neig:
#                this_dist += increment_dist
#            else:
#                n_neigh_bool = False
#        all_local_dim.append(local_dim_1_point(data[this_mask],var_thres=var_thres))
#    return all_local_dim



#### polynomial research ####
def generate_monomial(data, coeff, exponents):
    """Génère un monôme sous forme numérique et symbolique."""
    n_samples = data.shape[1]
    monomial = np.ones(n_samples) * coeff
    expression_parts = [f"{coeff}"]
    
    for var in exponents:
        monomial *= data[var]
        expression_parts.append(f"x{var}")
    
    return monomial, "*".join(expression_parts)

def generate_polynomial_combinations(data, list_coeff, max_order, n, thres=1e-5, excluded = []):
    """Génère des combinaisons de polynômes sans stocker tous les monômes en mémoire."""
    n_vars, n_samples = data.shape
    
    # Générer toutes les combinaisons possibles de monômes d'ordre ≤ max_order
    monomials = []
    monomials_expr = []
    
    print("Generate monomials")
    for order in range(1, max_order + 1):  # Inclure tous les ordres jusqu'à max_order
        for exponents in itertools.combinations_with_replacement(range(n_vars), order):
            for coeff in list_coeff:
                monomial_values, monomial_expr = generate_monomial(data, coeff, exponents)
                monomials.append((monomial_values, monomial_expr))  # Stocker temporairement pour l'itération
    print("Count combinations")
    #total_combinations = sum(1 for _ in itertools.combinations(monomials, n))  # Nombre total de polynômes à tester
    total_combinations = factorial(len(monomials))/(factorial(n)*factorial(len(monomials)-n))
    #print(total_combinations,n,len(monomials))
    # Générer et tester les combinaisons de polynômes

    for poly_combination in tqdm(itertools.combinations(monomials, n), total=total_combinations, desc="Progression"):
        summed_polynomial = sum(mono[0] for mono in poly_combination)
        summed_expression = " + ".join(mono[1] for mono in poly_combination)

        if np.sum(summed_polynomial**2)/n_samples < thres:
            simplified = simplify_expressions([summed_expression])
            #print(simplified[0])
            if simplified == [0] or str(simplified[0]) in excluded:
                continue
            return summed_expression  # Retourne seulement l'expression du polynôme trouvé

    return None  # Si aucun polynôme valide n'est trouvé

def simplify_expressions(expression_list):
    # Définir les symboles dynamiquement
    variables = {f'x{i}': sp.symbols(f'x{i}') for i in range(10)}  # Ajustez si besoin

    simplified_expressions = []

    for expression_str in expression_list:
        # Convertir la chaîne en expression sympy
        try:
            sympy_expr = sp.sympify(expression_str, locals=variables)
            simplified_expr = sp.simplify(sympy_expr)
            simplified_expressions.append(simplified_expr)
        except Exception as e:
            print(f"Erreur lors de la simplification de {expression_str}: {e}")
            simplified_expressions.append(None)

    return simplified_expressions

def factorial(x):
    """This is a recursive function
    to find the factorial of an integer"""
    res = 1
    for i in range(1,x+1):
        res *= i
    return res
    #if x == 1 or x == 0:
    #    return 1
    #else:
    #    # recursive call to the function
    #    return (x * factorial(x-1))

def travail(n):
    return factorial(200)

def generate_monomial_worker(args):
    """ Fonction exécutée en parallèle pour générer un monôme. """
    data, exponents, coeff = args
    monomial = np.ones(data.shape[1]) * coeff
    expression_parts = []

    for var in exponents:
        monomial *= data[var]
        expression_parts.append(f"x{var}")

    return monomial, f"{coeff}*" + "*".join(expression_parts)

def simplify_worker(expression_str):
    try:
        sympy_expr = sp.sympify(expression_str, locals=variables)
        return sp.simplify(sympy_expr)
    except Exception as e:
        print(f"Erreur lors de la simplification de {expression_str}: {e}")
        return None

def sum_combination_worker(args):
    """Fonction qui effectue la somme des polynômes pour une combinaison donnée."""
    polynomials, polynomial_expressions, combination = args
    summed_polynomial = sum(polynomials[i] for i in combination)
    summed_expression = " + ".join(polynomial_expressions[i] for i in combination)
    return np.sum(summed_polynomial**2)<10**(-5), summed_expression
