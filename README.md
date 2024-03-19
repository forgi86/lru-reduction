# Model order reduction of deep structured state-space models: A system-theoretic approach

This repository contains the Python code to reproduce the results of the paper [Model order reduction of deep structured state-space models: A system-theoretic approach](http://arxiv.org/abs/2308.13380)
by Marco Forgione, Manas Mejari, and Dario Piga.


## Linear Recurrent Unit
The Linear Recurrent Unit (LRU) is a sequence-to-sequence model defined by a linear dynamical system and implemented in state-space form as:
```math
\begin{align}
x_{k} = Ax_{x-1} + B u_k\\
y_k = \mathcal{R}[C x_k] + D u_k,
\end{align}
```
where $A$ is diagonal and complex-valued; $B, C$ are full complex-valued; $D$ is full real-valued; and $\mathcal{R}[\cdot]$ denotes the real part of its argument.

Smart parameterization/initialization of the system matrices make the LRU block easy to train numerically. Moreover, the use of [parallel scan algorithms](https://en.wikipedia.org/wiki/Prefix_sum) makes execution extremely fast on modern hardware. For more  details, read the original LRU paper.

## Deep LRU Architecture

LRU units are typically organized in a deep LRU architecture like:

<img src="architecture/lru_architecture.png"  width="500">

## Model order reduction and regularization
Model Order Reduction (MOR) is used in this paper to reduce the state dimensionality of Deep LRU architectures. Furthermore, regularization techniques promoting parsimonious representations are introduced.

# Main files

The main files are:

* [train.py](examples/f16/train.py): Training script
* [test.ipynb](examples/f16/test.ipynb): Standard testing (no MOR)
* [test_MOR.ipynb](examples/f16/test_MOR.ipynb): Testing with MOR (user-given order)
* [run_MOR_all.ipynb](examples/f16/run_MOR_all.ipynb): Sweep all model orders for all combinations of regularizers and MOR techniques
 
The training script is based on hydra to handle different configuragion. For instance, the model trained with Hankel nuclear norm minimization

```
python train.py +experiment=larg_reg_hankel
```

The configuration files defining the experiments are in the [conf](examples/f16/conf) folder.

# Software requirements
Experiments were performed on a Python 3.11 conda environment with:

 * numpy
 * scipy
 * matplotlib
 * python-control
 * pytorch (v2.2.1)
 

# Citing

If you find this project useful, we encourage you to:

* Star this repository :star: 



* Cite the [paper](https://arxiv.org/abs/2308.13380) 
```
@article{forgione2023from,
  author={Forgione, Marco and Mejari, Manas, and Piga, Dario},
  journal={IEEE Control Systems Letters}, 
  title={From System Models to Class Models:
   An In-Context Learning Paradigm}, 
  year={2023},
  volume={7},
  number={},
  pages={3513-3518},
  doi={10.1109/LCSYS.2023.3335036}
}
```
