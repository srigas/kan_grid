"""
Grid Types Module
=================

This module contains the concrete grid type implementations for different KAN layer types.
Each grid type defines the structure of knots/centers used by the corresponding layer's
basis functions.

Available Grid Types
--------------------
SplineGrid
    Grid for B-spline basis functions. Contains G+2k+1 knots per input dimension,
    where G is the number of intervals and k is the spline order. The extra 2k knots
    are for augmentation (k on each side) to ensure proper B-spline support.

RBFGrid
    Grid for Radial Basis Function centers. Contains D centers per input dimension,
    where D is the number of basis functions.

Key Concepts
------------
- **n_in**: Number of input dimensions. Each dimension has its own independent grid.
- **Grid shape**: (n_in, n_knots) where n_knots depends on the grid type.
- **Augmentation** (splines only): Extra knots added at boundaries for B-spline computation.

Example
-------
>>> from src.grids.types import SplineGrid, RBFGrid
>>> 
>>> # Create a spline grid for 3 input dimensions, order 3, with 5 intervals
>>> spline_grid = SplineGrid(n_in=3, k=3, G=5)
>>> print(spline_grid.item.shape)  # (3, 12) = (n_in, G + 2k + 1)
>>>
>>> # Create an RBF grid for 3 input dimensions with 8 centers
>>> rbf_grid = RBFGrid(n_in=3, D=8)
>>> print(rbf_grid.item.shape)  # (3, 8) = (n_in, D)
"""

from .spline import SplineGrid
from .rbf import RBFGrid

__all__ = ["SplineGrid", "RBFGrid"]
