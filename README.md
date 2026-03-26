## Annealing Importance Sampling - Sequential Monte Carlo (ASMC) for polynomial symbolic regression

This is the github repo for the paper ["Machine Learning the Conformal Manifold of Holographic CFT2s"](https://arxiv.org/abs/2511.02981).

The main part of the method implements an Annealed Importance Sampling with Sequential Monte Carlo algorithm to discover polynomial relation among data, such that P(x_data) = 0. 

It is supplemented with a gradient descent applied on a Supergravity potential to populate the underlying algebraic variety, and a local analysis to extract local features. 

## Installation

**Pre-requisites:**

```
Python
```

One can clone the repository via 
```
git clone https://github.com/BastD/ASMC-SymReg-3dSugra.git 
```

or directly download the zip file. 

Requirements

```
tensorflow==2.16.1
numpy<2
sklearn==1.8
matplotlib==3.10
tqdm==4.67.3
```

After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```

## Usage

**Main algorithm**

The main ASMC for polynomial Symbolic regression is available on the 

```python
run_ASMC.ipynb
``` 

notebook. It uses the 
```python
polynomial_sampler.py
```
where the main ASMC algorithm is implemented. 

**Gradient Descent and local analysis**

The gradient descent and local analysis can be found in the Gradient_descent_and_local_analysis.ipynb notebook. 

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
If you have any questions, please contact bastien.duboeuf@aei.mpg.de

