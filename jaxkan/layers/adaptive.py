"""
Grid Adaptive Mixin
===================

This module provides the GridAdaptiveMixin class, which encapsulates all grid
adaptation logic in a reusable way. Any grid-based layer can inherit from this
mixin to gain adaptive grid capabilities with user-configurable components.

The mixin abstracts away the details of:
- Computing bounds from data
- Computing importance weights via IDFs
- Applying adaptation strategies
- Handling grid extension

Usage
-----
Layers that use this mixin must implement the following interface:

Required attributes:
- grid: The grid object (SplineGrid or RBFGrid)
- n_in: Number of input dimensions

Required methods:
- _get_grid_values(): Returns the current grid values (knots or centers)
- _set_grid_values(dim, values): Sets grid values for a dimension
- _get_num_grid_points(): Returns number of interior points per dimension
- _update_grid_size(new_size): Updates the grid size (G or D)
- _recompute_coefficients(x, old_basis_values): Recomputes layer coefficients after grid change

Example
-------
>>> class MyGridLayer(GridAdaptiveMixin, nnx.Module):
...     def _get_grid_values(self):
...         return self.grid.knots
...     # ... implement other required methods
...
>>> layer = MyGridLayer(...)
>>> layer.update_grid(
...     x=x_batch,
...     idf=LossDensity(),
...     strategy=MixedAdaptation(grid_e=0.05),
...     losses=per_sample_loss
... )
"""

import jax.numpy as jnp
from abc import abstractmethod
from typing import Union, Any

from ..grids import (
    BoundsCalculator,
    UniformDensity,
    MixedAdaptation,
    BaseImportanceDensity,
    BaseAdaptationStrategy
)


class GridAdaptiveMixin:
    """
    Mixin class providing adaptive grid capabilities to KAN layers.
    
    This mixin provides a unified `update_grid()` method that handles all aspects
    of grid adaptation. The user can customize the behavior by providing:
    
    - **IDF (Importance Density Function)**: Determines WHERE knots should be placed
      based on various metrics (input distribution, loss, curvature, etc.)
    - **Strategy**: Determines HOW to compute new knot positions (uniform, quantile,
      or a mix of both)
    - **Bounds Calculator**: Determines the grid boundaries from data
    
    The mixin handles the common logic while delegating grid-type-specific operations
    to abstract methods that concrete layers must implement.
    
    Attributes
    ----------
    grid : SplineGrid or RBFGrid
        The grid object (must be provided by the layer class).
    n_in : int
        Number of input dimensions (must be provided by the layer class).
    
    Notes
    -----
    Layers using this mixin must implement the abstract interface methods.
    See the module docstring for the full interface specification.
    """
    
    # =========================================================================
    # Abstract interface - must be implemented by concrete layers
    # =========================================================================
    
    @abstractmethod
    def _get_grid_values(self) -> jnp.ndarray:
        """
        Get the current grid values (knots for splines, centers for RBFs).
        
        Returns
        -------
        jnp.ndarray
            Grid values array. Shape depends on grid type.
        """
        pass
    
    @abstractmethod
    def _set_grid_values(self, dim: int, values: jnp.ndarray) -> None:
        """
        Set grid values for a specific dimension.
        
        Parameters
        ----------
        dim : int
            The dimension index (0 to n_in-1).
        values : jnp.ndarray
            New grid values for this dimension. For splines, these are interior
            knots (augmentation handled internally). For RBFs, these are centers.
        """
        pass
    
    @abstractmethod
    def _get_num_grid_points(self) -> int:
        """
        Get the number of grid points per dimension.
        
        For SplineGrid: Returns G+1 (number of interior knots).
        For RBFGrid: Returns D (number of centers).
        
        Returns
        -------
        int
            Number of grid points per dimension.
        """
        pass
    
    @abstractmethod
    def _update_grid_size(self, new_size: int) -> None:
        """
        Update the grid size (extension).
        
        Parameters
        ----------
        new_size : int
            New grid size. For splines, this is G_new. For RBFs, this is D_new.
        """
        pass
    
    @abstractmethod
    def _compute_basis_output(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the weighted basis function output for coefficient recomputation.
        
        This computes: y = Sum(c_i * B_i(x)) for the current grid, which is used
        to solve for new coefficients after grid adaptation.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input data, shape (batch, n_in).
        
        Returns
        -------
        jnp.ndarray
            Basis output, shape (n_in, batch, n_out).
        """
        pass
    
    @abstractmethod
    def _recompute_coefficients(self, x: jnp.ndarray, target_output: jnp.ndarray) -> None:
        """
        Recompute layer coefficients after grid change.
        
        Solves the least squares problem: find c_new such that
        B_new(x) @ c_new ≈ target_output
        
        Parameters
        ----------
        x : jnp.ndarray
            Input data, shape (batch, n_in).
        target_output : jnp.ndarray
            Target values from old grid, shape (n_in, batch, n_out).
        """
        pass
    
    # =========================================================================
    # Main adaptation method
    # =========================================================================
    
    def update_grid(
        self,
        x: jnp.ndarray,
        idf: BaseImportanceDensity = None,
        strategy: BaseAdaptationStrategy = None,
        bounds_calculator: BoundsCalculator = None,
        grid_size_new: int = None,
        **idf_kwargs
    ) -> None:
        """
        Update the grid based on input data using the specified adaptation components.
        
        This is the main entry point for grid adaptation. It:
        1. Computes target output from current grid (for coefficient preservation)
        2. Computes new bounds from the data
        3. Computes importance weights using the IDF
        4. Computes new grid positions using the strategy
        5. Updates the grid
        6. Recomputes coefficients to preserve function approximation
        
        Parameters
        ----------
        x : jnp.ndarray
            Input data, shape (batch, n_in). Should be representative of the
            data distribution the model will see.
        idf : BaseImportanceDensity, optional
            Importance Density Function that determines where knots should be
            concentrated. If None, defaults to UniformDensity().
            
            Common choices:
            - UniformDensity(): Equal weighting everywhere (default)
            - LossDensity(): Weight by per-sample loss (requires losses= kwarg)
            - CurvatureDensity(): Weight by function curvature
            
        strategy : BaseAdaptationStrategy, optional
            Strategy for computing new grid positions from weights. If None,
            defaults to MixedAdaptation(grid_e=0.1).
            
            Common choices:
            - UniformAdaptation(): Evenly-spaced grid points
            - QuantileAdaptation(): Grid points at weighted quantiles
            - MixedAdaptation(grid_e): Blend of uniform and quantile
            
        bounds_calculator : BoundsCalculator, optional
            Calculator for determining grid bounds from data. If None, defaults
            to BoundsCalculator(margin=0.01).
            
        grid_size_new : int, optional
            New grid size (G for splines, D for RBFs). If None, keeps current
            size and only adapts positions.
            
        **idf_kwargs
            Additional keyword arguments passed to the IDF's compute() method.
            Common kwargs:
            - losses: Per-sample loss values for LossDensity
            - curvatures: Curvature estimates for CurvatureDensity
            - error_gradients: Error gradient magnitudes for ErrorGradientDensity
        
        Examples
        --------
        >>> # Basic adaptation using input distribution
        >>> layer.update_grid(x_batch)
        
        >>> # Loss-weighted adaptation
        >>> layer.update_grid(
        ...     x_batch,
        ...     idf=LossDensity(),
        ...     losses=per_sample_loss
        ... )
        
        >>> # Curvature-based with custom strategy
        >>> layer.update_grid(
        ...     x_batch,
        ...     idf=CurvatureDensity(),
        ...     strategy=MixedAdaptation(grid_e=0.2),
        ...     curvatures=curvature_estimates
        ... )
        
        >>> # Grid extension with adaptation
        >>> layer.update_grid(x_batch, grid_size_new=10)
        
        Notes
        -----
        The coefficient recomputation ensures that the layer's output changes
        minimally after adaptation. This is done via least squares fitting.
        """
        # Set defaults
        if idf is None:
            idf = UniformDensity()
        
        if strategy is None:
            strategy = MixedAdaptation(grid_e=0.1)
        
        if bounds_calculator is None:
            bounds_calculator = BoundsCalculator(margin=0.01)
        
        # Step 1: Compute target output from current grid (for coefficient preservation)
        target_output = self._compute_basis_output(x)
        
        # Step 2: Update grid size if requested
        if grid_size_new is not None:
            current_size = self._get_num_grid_points()
            # For splines, _get_num_grid_points returns G+1, so we need to compare correctly
            self._update_grid_size(grid_size_new)
        
        # Step 3: Compute new bounds from data
        new_bounds = bounds_calculator.compute(x)  # shape (n_in, 2)
        
        # Step 4: Compute importance weights and new grid positions for each dimension
        num_points = self._get_num_grid_points()
        
        for dim in range(self.n_in):
            # Get importance weights for this dimension
            x_dim = x[:, dim]
            weights = idf.compute(x_dim, **idf_kwargs)
            
            # Compute adapted grid positions for this dimension
            dim_bounds = new_bounds[dim]  # shape (2,)
            grid_positions = strategy.compute_grid(
                bounds=dim_bounds,
                num_points=num_points,
                x=x_dim,
                weights=weights
            )
            
            # Set the new grid values
            self._set_grid_values(dim=dim, values=grid_positions)
        
        # Step 5: Recompute coefficients to preserve function output
        self._recompute_coefficients(x, target_output)
