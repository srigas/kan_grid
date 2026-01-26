"""
Spline Grid Implementation
==========================

This module provides the SplineGrid class for managing B-spline knot vectors
in KAN (Kolmogorov-Arnold Network) layers.

B-Spline Background
-------------------
B-splines of order k are piecewise polynomial functions defined over a knot vector.
For a knot vector with G intervals (G+1 interior knots), we need G+k basis functions.
To properly define these basis functions at the boundaries, we augment the knot vector
with k additional knots on each side, resulting in G+2k+1 total knots.

Grid Structure
--------------
For a SplineGrid with parameters (n_in, k, G):
- Shape: (n_in, G + 2k + 1)
- Each row corresponds to one input dimension
- Each row contains: [k augmented left knots] + [G+1 interior knots] + [k augmented right knots]

Example knot vector for k=3, G=4:
    [-3h, -2h, -h, 0, h, 2h, 3h, 4h, 5h, 6h, 7h]  (11 knots = 4 + 2*3 + 1)
    |____aug____|  |___interior___|  |____aug____|
"""

import jax.numpy as jnp


class SplineGrid:
    """
    Grid class for B-spline basis functions in KAN layers.
    
    This class manages the knot vectors used to define B-spline basis functions.
    Each input dimension has its own independent knot vector, allowing for
    dimension-specific adaptation to data distributions.
    
    Attributes
    ----------
    n_in : int
        Number of input dimensions. Each dimension has its own knot vector.
    k : int
        Order of the B-spline basis functions. Determines smoothness (C^{k-1} continuity)
        and the number of augmentation knots needed.
    G : int
        Number of grid intervals. The number of interior knots is G+1, and the number
        of basis functions is G+k.
    knots : jnp.ndarray
        The actual knot array of shape (n_in, G + 2k + 1). Each row is a knot vector
        for one input dimension.
    
    Notes
    -----
    - The grid is initialized uniformly but can be adapted to data distributions
      using the set_knots() method.
    - Augmentation knots are automatically computed based on the interior knot spacing.
    - All dimensions start with identical knot vectors; adaptation makes them different.
    
    Examples
    --------
    >>> # Create a grid for 2 input dimensions, cubic splines (k=3), 5 intervals
    >>> bounds = jnp.array([[0.0, 1.0], [-1.0, 1.0]])
    >>> grid = SplineGrid(n_in=2, k=3, G=5, bounds=bounds)
    >>> print(grid.knots.shape)
    (2, 12)
    """
    
    def __init__(
        self, 
        n_in: int, 
        G: int, 
        k: int = 3, 
        bounds: jnp.ndarray = None
    ):
        """
        Initialize a SplineGrid instance.
        
        Parameters
        ----------
        n_in : int
            Number of input dimensions. Must be positive.
        G : int
            Number of grid intervals. The number of interior knots is G+1.
        k : int, optional
            Order of the B-spline basis functions. Default is 3 (cubic splines).
        bounds : jnp.ndarray, optional
            Initial bounds for each dimension as array of shape (n_in, 2).
            bounds[i, 0] is the lower bound for dimension i.
            bounds[i, 1] is the upper bound for dimension i.
            If None, defaults to [-1, 1] for all dimensions.
        
        Raises
        ------
        ValueError
            If n_in < 1, k < 1, G < 1, or bounds have incorrect shapes.
        """
        if n_in < 1:
            raise ValueError(f"n_in must be positive, got {n_in}")
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        if G < 1:
            raise ValueError(f"G must be positive, got {G}")
            
        self.n_in = n_in
        self.k = k
        self.G = G
        
        # Store bounds
        if bounds is None:
            self._bounds = jnp.stack([
                jnp.full((n_in,), -1.0),
                jnp.full((n_in,), 1.0)
            ], axis=1)  # shape (n_in, 2)
        else:
            self._bounds = bounds
        
        # Initialize the grid
        self.knots = self._initialize()
    
    def _initialize(self) -> jnp.ndarray:
        """
        Create and initialize the knot vectors.
        
        Returns
        -------
        jnp.ndarray
            Knot array of shape (n_in, G + 2k + 1).
        """
        lower = self._bounds[:, 0]  # shape (n_in,)
        upper = self._bounds[:, 1]  # shape (n_in,)
            
        # Step size per dimension: shape (n_in,)
        h = (upper - lower) / self.G
        
        # Create knot indices: [-k, -k+1, ..., 0, 1, ..., G, ..., G+k]
        # Shape: (G + 2k + 1,)
        indices = jnp.arange(-self.k, self.G + self.k + 1, dtype=jnp.float32)
        
        # Compute knots: lower + indices * h for each dimension
        # indices: (G + 2k + 1,), h: (n_in,), lower: (n_in,)
        # Result: (n_in, G + 2k + 1)
        grid = lower[:, None] + indices[None, :] * h[:, None]
        
        return grid
    
    def get_bounds(self) -> jnp.ndarray:
        """
        Get the current bounds for each dimension.
        
        Returns
        -------
        jnp.ndarray
            Bounds array of shape (n_in, 2) where bounds[i] = [lower, upper]
            for dimension i.
        """
        return self._bounds
    
    def set_knots(self, dim: int, interior_knots: jnp.ndarray) -> None:
        """
        Set the knot vector for a specific dimension from interior knots.
        
        This method takes interior knots and computes the full knot vector including
        augmentation knots.
        
        Parameters
        ----------
        dim : int
            The dimension to update (0 to n_in-1).
        interior_knots : jnp.ndarray
            Interior knot positions of shape (G + 1,). These are the knots
            within the domain bounds, not including augmentation.
        
        Notes
        -----
        Augmentation knots are computed using uniform spacing based on the
        span of the interior knots: h = (interior_knots[-1] - interior_knots[0]) / G.
        This matches the original grid update behavior and ensures consistency
        between the interior knots and augmentation knots.
        
        The stored bounds for this dimension are also updated to match the
        new interior knot endpoints.
        """
        # Get bounds from the actual interior knots (not stored bounds)
        lower = interior_knots[0]
        upper = interior_knots[-1]
        
        # Update stored bounds to reflect the new knot range
        self._bounds = self._bounds.at[dim, 0].set(lower)
        self._bounds = self._bounds.at[dim, 1].set(upper)
        
        # Compute step size for augmentation from interior knot span
        # This matches the original: h = (grid[-1] - grid[0]) / G_new
        h = (upper - lower) / self.G
        
        # Compute augmentation knots
        # Left: lower - k*h, lower - (k-1)*h, ..., lower - h
        left_aug = lower - jnp.arange(self.k, 0, -1) * h  # shape (k,)
        
        # Right: upper + h, upper + 2*h, ..., upper + k*h  
        right_aug = upper + jnp.arange(1, self.k + 1) * h  # shape (k,)
        
        # Concatenate: left_aug + interior + right_aug
        new_knots = jnp.concatenate([left_aug, interior_knots, right_aug])
        
        # Update the knots for this dimension
        self.knots = self.knots.at[dim].set(new_knots)
        
    def update_G(self, new_G: int) -> None:
        """
        Update the number of grid intervals (extension).
        
        This method reinitializes the grid with the new G value, using
        uniform knot spacing based on the current bounds.
        
        Parameters
        ----------
        new_G : int
            New number of grid intervals. Must be positive.
        
        Raises
        ------
        ValueError
            If new_G < 1.
        """
        if new_G < 1:
            raise ValueError(f"new_G must be positive, got {new_G}")
        self.G = new_G
        self.knots = self._initialize()
