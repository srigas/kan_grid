"""
RBF Grid Implementation
=======================

This module provides the RBFGrid class for managing Radial Basis Function (RBF) 
center positions in KAN (Kolmogorov-Arnold Network) layers.

RBF Background
--------------
Radial Basis Functions are localized basis functions centered at specific points.
Each RBF φ(||x - c||) computes a response based on the distance from the input x
to its center c. Common RBF types include Gaussian, multiquadric, and inverse
multiquadric.

Grid Structure
--------------
For an RBFGrid with parameters (n_in, D):
- Shape: (n_in, D)
- Each row corresponds to one input dimension
- Each row contains D center positions for that dimension

Unlike SplineGrid, RBFGrid does not require augmentation. The D centers directly
define D basis functions per input dimension.
"""

import jax.numpy as jnp


class RBFGrid:
    """
    Grid class for Radial Basis Function centers in KAN layers.
    
    This class manages the center positions used to define RBF basis functions.
    Each input dimension has its own independent set of centers, allowing for
    dimension-specific adaptation to data distributions.
    
    Attributes
    ----------
    n_in : int
        Number of input dimensions. Each dimension has its own center array.
    D : int
        Number of RBF centers per dimension. This directly equals the number
        of basis functions per dimension (unlike splines which have G+k).
    centers : jnp.ndarray
        The actual center array of shape (n_in, D). Each row contains center
        positions for one input dimension.
    
    Notes
    -----
    - The grid is initialized with uniformly-spaced centers but can be adapted
      to data distributions using the set_centers() method.
    - No augmentation is needed (unlike SplineGrid).
    - All dimensions start with the same center pattern scaled to their bounds.
    
    Examples
    --------
    >>> # Create a grid for 2 input dimensions with 10 centers each
    >>> bounds = jnp.array([[0.0, 1.0], [-1.0, 1.0]])
    >>> grid = RBFGrid(n_in=2, D=10, bounds=bounds)
    >>> print(grid.centers.shape)
    (2, 10)
    """
    
    def __init__(
        self, 
        n_in: int, 
        D: int, 
        bounds: jnp.ndarray = None
    ):
        """
        Initialize an RBFGrid instance.
        
        Parameters
        ----------
        n_in : int
            Number of input dimensions. Must be positive.
        D : int
            Number of RBF centers per dimension. Must be at least 2.
        bounds : jnp.ndarray, optional
            Initial bounds for each dimension as array of shape (n_in, 2).
            bounds[i, 0] is the lower bound for dimension i.
            bounds[i, 1] is the upper bound for dimension i.
            If None, defaults to [-1, 1] for all dimensions.
        
        Raises
        ------
        ValueError
            If n_in < 1, D < 2, or bounds have incorrect shapes.
        """
        if n_in < 1:
            raise ValueError(f"n_in must be positive, got {n_in}")
        if D < 2:
            raise ValueError(f"D must be at least 2, got {D}")
            
        self.n_in = n_in
        self.D = D
        
        # Store bounds
        if bounds is None:
            self._bounds = jnp.stack([
                jnp.full((n_in,), -1.0),
                jnp.full((n_in,), 1.0)
            ], axis=1)  # shape (n_in, 2)
        else:
            self._bounds = bounds
        
        # Initialize the grid
        self.centers = self._initialize()
    
    def _initialize(self) -> jnp.ndarray:
        """
        Create and initialize the center positions.
        
        Returns
        -------
        jnp.ndarray
            Center array of shape (n_in, D).
        """
        lower = self._bounds[:, 0]  # shape (n_in,)
        upper = self._bounds[:, 1]  # shape (n_in,)
        
        # Create normalized positions [0, 1/(D-1), ..., 1]
        t = jnp.linspace(0, 1, self.D)  # shape (D,)
        
        # Scale to [lower, upper] for each dimension
        # t: (D,), lower/upper: (n_in,)
        # Result: (n_in, D)
        centers = lower[:, None] + t[None, :] * (upper - lower)[:, None]
        
        return centers
    
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
    
    def set_centers(self, dim: int, centers: jnp.ndarray) -> None:
        """
        Set the centers for a specific dimension.
        
        Parameters
        ----------
        dim : int
            The dimension to update (0 to n_in-1).
        centers : jnp.ndarray
            Center positions of shape (D,).
        
        Notes
        -----
        The stored bounds for this dimension are updated to match the
        span of the new centers.
        """
        # Update stored bounds to reflect the new center range
        self._bounds = self._bounds.at[dim, 0].set(centers[0])
        self._bounds = self._bounds.at[dim, 1].set(centers[-1])
        
        self.centers = self.centers.at[dim].set(centers)
        
    def update_D(self, new_D: int) -> None:
        """
        Update the number of centers (extension).
        
        This method reinitializes the grid with the new D value, using
        uniform center spacing based on the current bounds.
        
        Parameters
        ----------
        new_D : int
            New number of centers. Must be at least 2.
        
        Raises
        ------
        ValueError
            If new_D < 2.
        """
        if new_D < 2:
            raise ValueError(f"new_D must be at least 2, got {new_D}")
        self.D = new_D
        self.centers = self._initialize()
