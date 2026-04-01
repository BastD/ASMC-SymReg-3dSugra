## Annealing Importance Sampling - Sequential Monte Carlo (ASMC) for polynomial symbolic regression

This is the github repo for the paper ["Machine Learning the Conformal Manifold of Holographic CFT2s"](https://arxiv.org/abs/2511.02981).

The repository contains a module `polynomial_sampler.py` where the method is implemented.
This algorithm performs Annealed Importance Sampling with Sequential Monte Carlo to discover polynomial relations in data, such that $P(x_{\text{data}}) = 0$.

The module is supplemented by two jupyter notebooks:

1. ASMC_tutorial.ipynb: Provides a minimal working example of the algorithm.
2. Gradient_descent_and_local_analysis.ipynb: Applies the method to the data generated via gradient descent (already available from `Points_5d.npy`) on the Supergravity potential described in the [manuscript](https://arxiv.org/abs/2511.02981). It also performs part of the analysis to extract local features. 

## Installation

**Pre-requisites:**

One can clone the repository via 
```
git clone https://github.com/BastD/ASMC-SymReg-3dSugra.git 
```

or directly download the zip file. 

Requirements

- Python 3.11


and packages

```
tensorflow==2.16.1
numpy==1.26.4
scikit-learn==1.8
matplotlib==3.10
tqdm==4.67.3
numba==0.58.1
```

After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```


## Citation
```python
@article{Duboeuf:2025inv,
    author = "Duboeuf, Bastien and Eloy, Camille and Larios, Gabriel",
    title = "{Machine Learning the Conformal Manifold of Holographic CFT$_{2}$s}",
    eprint = "2511.02981",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    month = "11",
    year = "2025"
}
```

## Contact
If you have any questions, please contact [Bastien Duboeuf](https://inspirehep.net/authors/1881378), [Camille Eloy](https://inspirehep.net/authors/1748427) or [Gabriel Larios](https://inspirehep.net/authors/1728355)

