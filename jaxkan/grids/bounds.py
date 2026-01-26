"""
Bounds Calculation
==================

This module provides utilities for automatically calculating and managing grid bounds
based on input data. It replaces the manual `grid_range` parameter with data-driven
bounds computation.

Design Philosophy
-----------------
1. **Data-Driven**: Bounds are determined by actual input data, not arbitrary defaults.
2. **Per-Dimension**: Each input dimension has its own bounds, allowing for different
   scales across dimensions.
3. **Margin-Based**: A configurable margin is added to ensure knots cover slightly
   beyond the data range, preventing edge effects.
4. **Dynamic**: Bounds can expand (or optionally shrink) as new data is seen.

Bound Computation
-----------------
For input data x with shape (batch, n_in):
- Lower bound for dimension i: min(x[:, i]) - margin * range_i
- Upper bound for dimension i: max(x[:, i]) + margin * range_i

Where range_i = max(x[:, i]) - min(x[:, i])

Return Format
-------------
Bounds are returned as a 2D array of shape (n_in, 2) where:
- bounds[dim, 0] is the lower bound for dimension dim
- bounds[dim, 1] is the upper bound for dimension dim

This format allows easy indexing: bounds[0] gives [lower, upper] for dim 0.

For Splines vs RBFs
-------------------
- **Splines**: Bounds define where interior knots are placed. Augmentation knots
  extend k steps beyond bounds on each side.
- **RBFs**: Bounds define where centers are placed. No augmentation needed.

Usage Pattern
-------------
    bounds_calc = BoundsCalculator(margin=0.1)
    
    # Initial bounds from training data
    bounds = bounds_calc.compute(x_train)  # shape (n_in, 2)
    
    # Later, update bounds if needed
    bounds = bounds_calc.update(x_val, bounds)
"""

import jax.numpy as jnp
from typing import Optional


class BoundsCalculator:
    """
    Calculator for data-driven grid bounds.
    
    This class computes bounds for grid knots/centers based on input data.
    It supports configurable margins and can optionally expand (or shrink)
    existing bounds when new data is observed.
    
    Attributes
    ----------
    margin : float
        Fractional margin to add beyond data range. For example, margin=0.1
        adds 10% of the data range on each side.
    allow_shrink : bool
        Whether bounds can shrink when new data has a smaller range.
        If False (default), bounds only expand.
    
    Notes
    -----
    The margin serves several purposes:
    1. Prevents numerical issues at exact boundaries
    2. Provides some extrapolation capability
    3. Ensures augmentation knots (for splines) are well-defined
    
    Examples
    --------
    >>> calc = BoundsCalculator(margin=0.1)
    >>> x = jnp.array([[0.0, -1.0], [1.0, 1.0], [0.5, 0.0]])
    >>> bounds = calc.compute(x)
    >>> print(bounds.shape)  # (2, 2)
    >>> print(bounds[0])  # [-0.1, 1.1]  (dim 0: data range [0,1] + 10% margin)
    >>> print(bounds[1])  # [-1.2, 1.2]  (dim 1: data range [-1,1] + 10% margin)
    """
    
    def __init__(self, margin: float = 0.1, allow_shrink: bool = False):
        """
        Initialize the bounds calculator.
        
        Parameters
        ----------
        margin : float, optional
            Fractional margin to add beyond data range. Default is 0.1 (10%).
            - margin=0.0: Bounds exactly at data min/max
            - margin=0.1: 10% extra on each side
            - margin=0.5: 50% extra on each side (very conservative)
            
            For data in range [a, b] with margin m:
            - Lower bound: a - m * (b - a)
            - Upper bound: b + m * (b - a)
        allow_shrink : bool, optional
            Whether to allow bounds to shrink when updating with new data.
            Default is False (bounds only expand, never shrink).
            
            Set to True if you want bounds to track the current data distribution
            exactly. Set to False (default) for stability across training where
            data distributions may vary.
        
        Raises
        ------
        ValueError
            If margin is negative.
        
        Examples
        --------
        >>> # Conservative margin
        >>> calc = BoundsCalculator(margin=0.2)
        
        >>> # Exact bounds (not recommended - edge effects)
        >>> calc = BoundsCalculator(margin=0.0)
        
        >>> # Allow shrinking (tracks data distribution exactly)
        >>> calc = BoundsCalculator(margin=0.1, allow_shrink=True)
        """
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        self.margin = margin
        self.allow_shrink = allow_shrink
    
    def compute(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute bounds from input data.
        
        This method computes per-dimension bounds based on the min and max
        of the input data, plus the configured margin.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input data of shape (batch, n_in). Bounds are computed per dimension.
        
        Returns
        -------
        jnp.ndarray
            Bounds array of shape (n_in, 2) where bounds[i, 0] is the lower bound
            and bounds[i, 1] is the upper bound for dimension i.
        
        Notes
        -----
        Special cases handled:
        - If all values in a dimension are identical, a small range is used
          to avoid division by zero: range = max(|value|, 1.0) * 0.1
        - Bounds are computed per dimension independently
        
        Examples
        --------
        >>> calc = BoundsCalculator(margin=0.1)
        >>> 
        >>> # Input with shape (batch, n_in)
        >>> x = jnp.array([[0.0, -1.0], [1.0, 1.0]])
        >>> bounds = calc.compute(x)
        >>> print(bounds.shape)  # (2, 2)
        >>> print(bounds[0])  # [-0.1, 1.1]  for dim 0
        >>> print(bounds[1])  # [-1.2, 1.2]  for dim 1
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D with shape (batch, n_in), got shape {x.shape}")
        
        n_in = x.shape[1]
        
        # Compute min and max per dimension
        data_min = jnp.min(x, axis=0)  # shape (n_in,)
        data_max = jnp.max(x, axis=0)  # shape (n_in,)
        
        # Compute range, handling zero-range case
        data_range = data_max - data_min
        # For zero-range dimensions, use a small default range
        safe_range = jnp.where(
            data_range > 0,
            data_range,
            jnp.maximum(jnp.abs(data_min), 1.0) * 0.1
        )
        
        # Apply margin
        margin_size = self.margin * safe_range
        lower = data_min - margin_size
        upper = data_max + margin_size
        
        # Stack into (n_in, 2) array
        bounds = jnp.stack([lower, upper], axis=1)
        
        return bounds
    
    def update(
        self, 
        x: jnp.ndarray, 
        current_bounds: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Update bounds based on new data.
        
        Computes bounds from new data and merges with existing bounds.
        By default (allow_shrink=False), bounds only expand.
        
        Parameters
        ----------
        x : jnp.ndarray
            New input data of shape (batch, n_in).
        current_bounds : jnp.ndarray
            Existing bounds of shape (n_in, 2).
        
        Returns
        -------
        jnp.ndarray
            Updated bounds of shape (n_in, 2).
        
        Examples
        --------
        >>> calc = BoundsCalculator(margin=0.0, allow_shrink=False)
        >>> 
        >>> # Initial bounds
        >>> bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        >>> 
        >>> # New data extends bounds
        >>> x_new = jnp.array([[-5.0, 5.0], [15.0, 5.0]])
        >>> bounds = calc.update(x_new, bounds)
        >>> print(bounds[0])  # [-5.0, 15.0]  (expanded)
        """
        # Compute bounds from new data
        new_bounds = self.compute(x)
        
        if self.allow_shrink:
            # Use new bounds directly
            return new_bounds
        else:
            # Only expand, never shrink
            lower = jnp.minimum(current_bounds[:, 0], new_bounds[:, 0])
            upper = jnp.maximum(current_bounds[:, 1], new_bounds[:, 1])
            return jnp.stack([lower, upper], axis=1)


def compute_initial_bounds(
    x: jnp.ndarray, 
    margin: float = 0.1
) -> jnp.ndarray:
    """
    Convenience function to compute initial bounds from data.
    
    This is a stateless version of BoundsCalculator.compute() for simple use cases
    where you just need to compute bounds once without tracking state.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input data of shape (batch, n_in).
    margin : float, optional
        Fractional margin to add. Default is 0.1 (10%).
    
    Returns
    -------
    jnp.ndarray
        Bounds array of shape (n_in, 2).
    
    Examples
    --------
    >>> x = jnp.array([[0.0, -1.0], [1.0, 1.0]])
    >>> bounds = compute_initial_bounds(x, margin=0.1)
    >>> print(bounds[0])  # [-0.1, 1.1]  for dim 0
    >>> print(bounds[1])  # [-1.2, 1.2]  for dim 1
    """
    calc = BoundsCalculator(margin=margin)
    return calc.compute(x)
