# Introduction

This repository contains the code required to reproduce the experimental results and plots for the paper "[A Dynamic Framework for Grid Adaptation in Kolmogorov–Arnold Networks](https://arxiv.org/abs/TODO)".


# Getting Started

After cloning the repository,

```bash
git clone https://github.com/srigas/kan_grid.git grids
cd grids
```

create a Python virtual environment, activate it and install all dependencies:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
pip3 install -r requirements.txt
```

Note that the [jaxKAN](https://jaxkan.readthedocs.io/en/latest/) package is not installed via PyPI, but is instead provided locally, as this version includes code that (during release) is not yet available inside the package.

Then launch JupyterLab:

```bash
jupyter lab
```

The data for the synthetic functions described in the first experiment have already been generated using the `benchmarks/generate_custom_dataset.py` file.

The code for each family of experiments can be found in the corresponding Jupyter Notebook, with the synthetic function benchmarks in `Benchmark 1 - Custom Functions.ipynb`, the function regression on a subset of the Feynman dataset in `Benchmark 2 - Feynman Subset.ipynb` and the forward PDE problem in `Benchmark 3 - Helmholtz PDE.ipynb`. The corresponding results can be found in the `results` folder, which also includes the plots shown in the paper. The analysis of these results is handled via the `Analysis.ipynb` notebook, while the `Example.ipynb` notebook contains the code required to generate the plot shown in Figure 1 of the paper.


# Citation

If the code and/or results presented in this work helped you for your own work, please cite our work as:

```
@article{kan_grid,
   title={A Dynamic Framework for Grid Adaptation in Kolmogorov–Arnold Networks},
   author={Rigas, Spyros and Papaioannou, Thanasis and Trakadas, Panagiotis and Alexandridis, Georgios},
   year={2026},
   eprint={TODO},
   archivePrefix={arXiv},
   primaryClass={cs.LG},
   url={https://arxiv.org/abs/TODO}, 
}

```
