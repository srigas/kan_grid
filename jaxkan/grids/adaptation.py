"""
Adaptation Strategies
=====================

This module provides strategies for computing new knot positions during grid
adaptation. Strategies take input samples and importance weights, and produce
new knot positions.

Strategy Design Principles
--------------------------
1. **Weight-Driven**: Strategies use importance weights from IDFs to guide placement.
2. **Bounds-Aware**: All strategies respect the provided bounds for each dimension.
3. **Dimension-Independent**: Each input dimension is adapted independently.
4. **Grid-Type Agnostic**: Strategies compute knot positions; the grid type determines
   what to do with them (e.g., splines need augmentation, RBFs don't).

Available Strategies
--------------------
UniformAdaptation
    Places knots uniformly within bounds, ignoring weights.
QuantileAdaptation
    Places knots at weighted quantiles of the input distribution.
MixedAdaptation
    Linear combination of uniform and quantile placement.

Mathematical Background
-----------------------
**Quantile Adaptation**:
Given samples x with weights w, we want knot positions such that each interval
contains approximately equal cumulative weight.

For n_knots positions, we find positions p_0, p_1, ..., p_{n_knots-1} such that:
- p_0 = lower bound
- p_{n_knots-1} = upper bound  
- CDF(p_i) ≈ i / (n_knots - 1)

Where CDF is the weighted empirical cumulative distribution function.

Usage Pattern
-------------
    strategy = MixedAdaptation(grid_e=0.1)
    
    # Get importance weights from IDF
    weights = density(x, metric=per_sample_loss)
    
    # Compute new grid positions per dimension
    new_grid = strategy.compute_grid(bounds, num_points=G+1, x=x, weights=weights)
    
    # Set knots on grid (grid handles augmentation if needed)
    grid.set_knots(dim=0, interior_knots=new_grid)
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp


class BaseAdaptationStrategy(ABC):
    """
    Abstract base class for all adaptation strategies.
    
    Adaptation strategies compute new knot positions based on input samples
    and importance weights. Different strategies produce different knot
    distributions suitable for various use cases.
    
    Subclasses must implement the `compute_grid()` method.
    
    Notes
    -----
    The returned knots are interior knots only. For spline grids, augmentation
    knots are computed separately by the grid class.
    
    Examples
    --------
    To create a custom strategy:
    
    >>> class MyStrategy(BaseAdaptationStrategy):
    ...     def compute_grid(
    ...         self, 
    ...         bounds: jnp.ndarray,
    ...         num_points: int,
    ...         x: jnp.ndarray = None,
    ...         weights: jnp.ndarray = None
    ...     ) -> jnp.ndarray:
    ...         # Custom logic for single dimension
    ...         return jnp.linspace(bounds[0], bounds[1], num_points)
    """
    
    @abstractmethod
    def compute_grid(
        self, 
        bounds: jnp.ndarray,
        num_points: int,
        x: jnp.ndarray = None,
        weights: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Compute new grid positions for a single dimension.
        
        Parameters
        ----------
        bounds : jnp.ndarray
            Bounds as array of shape (2,) containing [lower, upper].
        num_points : int
            Number of grid points to place. For splines with G intervals, 
            this is G+1. For RBFs with D centers, this is D.
        x : jnp.ndarray, optional
            Input samples of shape (batch,) for this dimension.
            Some strategies may ignore this.
        weights : jnp.ndarray, optional
            Importance weights of shape (batch,), typically from an IDF.
            Some strategies may ignore this.
        
        Returns
        -------
        jnp.ndarray
            Grid positions of shape (num_points,), sorted in ascending order.
        """
        pass


class UniformAdaptation(BaseAdaptationStrategy):
    """
    Uniform adaptation strategy - places knots uniformly within bounds.
    
    This strategy ignores the input samples and weights entirely, placing
    knots at evenly-spaced positions between the lower and upper bounds.
    
    This is equivalent to grid_e=1.0 in the old adaptation scheme.
    
    Use Cases
    ---------
    - When function complexity is expected to be uniform across the domain
    - As a baseline for comparison
    - When weights are unreliable or noisy
    
    Examples
    --------
    >>> strategy = UniformAdaptation()
    >>> bounds = jnp.array([0.0, 1.0])
    >>> grid = strategy.compute_grid(bounds, num_points=5)
    >>> print(grid)  # [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    
    def compute_grid(
        self, 
        bounds: jnp.ndarray,
        num_points: int,
        x: jnp.ndarray = None,
        weights: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Compute uniformly-spaced grid positions for a single dimension.
        
        Parameters
        ----------
        bounds : jnp.ndarray
            Bounds as array of shape (2,) containing [lower, upper].
        num_points : int
            Number of grid points to place.
        x : jnp.ndarray, optional
            Input samples. Ignored by this strategy.
        weights : jnp.ndarray, optional
            Importance weights. Ignored by this strategy.
        
        Returns
        -------
        jnp.ndarray
            Uniformly-spaced grid positions of shape (num_points,).
        
        Notes
        -----
        For num_points positions between lower and upper:
        grid[i] = lower + i * (upper - lower) / (num_points - 1)
        """
        lower = bounds[0]
        upper = bounds[1]
        
        if num_points == 1:
            # Single point at midpoint
            return jnp.array([(lower + upper) / 2])
        
        # Create uniformly spaced grid
        return jnp.linspace(lower, upper, num_points)


class QuantileAdaptation(BaseAdaptationStrategy):
    """
    Quantile adaptation strategy - places knots at weighted quantiles.
    
    This strategy places knots such that each interval contains approximately
    equal cumulative weight. This concentrates knots in regions with high
    importance weights.
    
    This is equivalent to grid_e=0.0 in the old adaptation scheme, but now
    generalized to support arbitrary importance weights (not just uniform).
    
    Mathematical Details
    --------------------
    Given samples x_i with weights w_i, we compute the weighted empirical CDF:
        F(t) = Σ_{x_i <= t} w_i / Σ w_i
    
    Then place knots at the inverse CDF values:
        knot_j = F^{-1}(j / (n_knots - 1))
    
    This ensures equal weighted mass in each interval.
    
    Use Cases
    ---------
    - When importance weights meaningfully indicate where resolution is needed
    - For loss-based, curvature-based, or salience-based adaptation
    - When function complexity varies across the domain
    
    Examples
    --------
    >>> strategy = QuantileAdaptation()
    >>> x = jnp.linspace(0.0, 1.0, 100)
    >>> weights = jnp.ones(100)
    >>> bounds = jnp.array([0.0, 1.0])
    >>> grid = strategy.compute_grid(bounds, num_points=5, x=x, weights=weights)
    """
    
    def compute_grid(
        self, 
        bounds: jnp.ndarray,
        num_points: int,
        x: jnp.ndarray = None,
        weights: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Compute grid positions at weighted quantiles for a single dimension.
        
        Parameters
        ----------
        bounds : jnp.ndarray
            Bounds as array of shape (2,) containing [lower, upper].
        num_points : int
            Number of grid points to place.
        x : jnp.ndarray, optional
            Input samples of shape (batch,). Required for this strategy.
        weights : jnp.ndarray, optional
            Importance weights of shape (batch,). Higher weights mean
            more grid points should be placed near those samples.
            Defaults to uniform weights if not provided.
        
        Returns
        -------
        jnp.ndarray
            Grid positions of shape (num_points,).
        
        Notes
        -----
        Implementation:
        1. Sort samples by value
        2. Compute weighted cumulative distribution
        3. Find sample indices corresponding to target quantiles
        4. The first and last points are set to the bounds to ensure full coverage.
        """
        lower = bounds[0]
        upper = bounds[1]
        
        if num_points == 1:
            return jnp.array([(lower + upper) / 2])
        
        # If no data provided, fall back to uniform
        if x is None:
            return jnp.linspace(lower, upper, num_points)
        
        batch = x.shape[0]
        
        # Default to uniform weights if not provided
        if weights is None:
            weights = jnp.ones(batch)
        
        # Sort by x value
        sort_idx = jnp.argsort(x)
        x_sorted = x[sort_idx]
        w_sorted = weights[sort_idx]
        
        # Compute cumulative weights (CDF-like)
        cumsum = jnp.cumsum(w_sorted)
        cumsum = cumsum / (cumsum[-1] + 1e-10)  # Normalize to [0, 1], avoid div by zero
        
        # Target quantiles: 0, 1/(n-1), 2/(n-1), ..., 1
        target_quantiles = jnp.arange(num_points, dtype=jnp.float32) / (num_points - 1)
        
        # Find indices where cumsum crosses each target quantile
        indices = jnp.searchsorted(cumsum, target_quantiles, side='left')
        indices = jnp.clip(indices, 0, batch - 1)
        
        # Get grid positions from sorted x values
        grid = x_sorted[indices]
        
        # Enforce bounds at endpoints
        grid = grid.at[0].set(lower)
        grid = grid.at[-1].set(upper)
        
        # Clip all values to bounds
        grid = jnp.clip(grid, lower, upper)
        
        return grid


class MixedAdaptation(BaseAdaptationStrategy):
    """
    Mixed adaptation strategy - linear blend of uniform and quantile placement.
    
    This strategy combines uniform and quantile adaptation with a mixing
    parameter `grid_e`:
        grid = grid_e * uniform_grid + (1 - grid_e) * quantile_grid
    
    This is directly equivalent to the old grid adaptation behavior, but now
    with support for arbitrary importance weights in the quantile component.
    
    Attributes
    ----------
    grid_e : float
        Mixing parameter in [0, 1].
        - grid_e=0.0: Pure quantile (adaptive to weights)
        - grid_e=1.0: Pure uniform (ignores weights)
        - grid_e=0.5: Equal mix of both
    
    Use Cases
    ---------
    - When you want some adaptation but also some regularity
    - To balance between data-driven and uniform placement
    - For robustness when weights are noisy or unreliable
    
    Examples
    --------
    >>> strategy = MixedAdaptation(grid_e=0.1)  # 10% uniform, 90% quantile
    >>> grid = strategy.compute_grid(bounds, num_points=10, x=x, weights=weights)
    
    >>> # Equivalent to old behavior with UniformDensity
    >>> strategy = MixedAdaptation(grid_e=0.05)  # Old default
    """
    
    def __init__(self, grid_e: float = 0.1):
        """
        Initialize the mixed adaptation strategy.
        
        Parameters
        ----------
        grid_e : float, optional
            Mixing parameter. Default is 0.1.
            - grid_e=0.0: Pure quantile adaptation
            - grid_e=1.0: Pure uniform adaptation
            Values outside [0, 1] are clipped.
        
        Examples
        --------
        >>> # Mostly adaptive
        >>> strategy = MixedAdaptation(grid_e=0.05)
        
        >>> # Balanced
        >>> strategy = MixedAdaptation(grid_e=0.5)
        
        >>> # Mostly uniform
        >>> strategy = MixedAdaptation(grid_e=0.9)
        """
        self.grid_e = float(jnp.clip(grid_e, 0.0, 1.0))
        self._uniform = UniformAdaptation()
        self._quantile = QuantileAdaptation()
    
    def compute_grid(
        self, 
        bounds: jnp.ndarray,
        num_points: int,
        x: jnp.ndarray = None,
        weights: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Compute grid positions as a blend of uniform and quantile.
        
        Parameters
        ----------
        bounds : jnp.ndarray
            Bounds as array of shape (2,) containing [lower, upper].
        num_points : int
            Number of grid points to place.
        x : jnp.ndarray, optional
            Input samples of shape (batch,).
        weights : jnp.ndarray, optional
            Importance weights of shape (batch,).
        
        Returns
        -------
        jnp.ndarray
            Blended grid positions of shape (num_points,).
        
        Notes
        -----
        The blending is done element-wise:
            grid = grid_e * uniform + (1 - grid_e) * quantile
        
        Edge cases:
        - grid_e=0.0: Returns pure quantile grid
        - grid_e=1.0: Returns pure uniform grid
        """
        # Short-circuit for pure strategies
        if self.grid_e == 1.0:
            return self._uniform.compute_grid(bounds, num_points, x, weights)
        if self.grid_e == 0.0:
            return self._quantile.compute_grid(bounds, num_points, x, weights)
        
        # Compute both types of grids
        uniform_grid = self._uniform.compute_grid(bounds, num_points, x, weights)
        quantile_grid = self._quantile.compute_grid(bounds, num_points, x, weights)
        
        # Linear blend
        grid = self.grid_e * uniform_grid + (1.0 - self.grid_e) * quantile_grid
        
        return grid
