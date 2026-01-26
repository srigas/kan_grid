"""
Importance Density Functions (IDFs)
===================================

This module provides Importance Density Functions that determine WHERE knots should
be placed during grid adaptation. IDFs take input samples and optionally a metric
array, and output importance weights that guide knot placement.

IDF Design Principles
---------------------
1. **Separation of Concerns**: IDFs do not compute metrics themselves. They receive
   pre-computed metrics (loss, gradients, etc.) as input and only handle the
   weight computation logic.
2. **Uniform Interface**: All IDFs have a `compute(x, **kwargs)` method.
3. **Per-Dimension Output**: Weights are returned per sample per dimension,
   shape (batch, n_in), allowing dimension-specific adaptation.
4. **Composable** (future): Multiple IDFs could be combined with weighted averaging.

Available IDFs
--------------
UniformDensity
    Assigns equal importance to all samples. Results in uniformly-spaced knots.
LossDensity
    Importance based on per-sample loss. Places more knots where loss is high.
CurvatureDensity
    Importance based on function curvature. Places more knots where function is "curvy".
SalienceDensity
    Importance based on input sensitivity. Places more knots where small input
    changes cause large output changes.
GradientLossDensity
    Importance based on loss sensitivity to inputs. Places more knots where
    small input changes cause large loss changes.

Usage Pattern
-------------
    # During adaptation
    density = LossDensity()
    
    # Compute per-sample loss externally
    per_sample_loss = compute_per_sample_loss(model, x, y)
    
    # Get importance weights
    weights = density.compute(x, losses=per_sample_loss)  # shape (batch, n_in)
    
    # Use weights for quantile-based knot placement
    knots = adaptation_strategy.compute_grid(bounds, n_knots, x, weights)
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp


class BaseImportanceDensity(ABC):
    """
    Abstract base class for all Importance Density Functions.
    
    Importance Density Functions (IDFs) compute importance weights for each
    sample that guide knot placement during grid adaptation. Higher weights
    indicate regions that should have finer resolution (more knots).
    
    Subclasses must implement the `compute()` method with the appropriate
    keyword arguments for their specific metrics.
    
    Notes
    -----
    The output weights should be non-negative. They do not need to sum to 1;
    normalization is handled by the adaptation strategy.
    """
    
    @abstractmethod
    def compute(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Compute importance weights for the given samples.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input samples of shape (batch, n_in).
        **kwargs
            Additional keyword arguments for metric data (e.g., losses=, curvatures=).
        
        Returns
        -------
        jnp.ndarray
            Importance weights of shape (batch, n_in).
        """
        pass


class UniformDensity(BaseImportanceDensity):
    """
    Uniform importance density - assigns equal weight to all samples.
    
    This IDF results in uniformly-spaced knots across the domain, regardless
    of where the data is concentrated.
    
    Examples
    --------
    >>> density = UniformDensity()
    >>> x = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> weights = density.compute(x)
    >>> print(weights)
    [[1. 1.]
     [1. 1.]
     [1. 1.]]
    """
    
    def compute(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Return uniform weights (all ones).
        
        Parameters
        ----------
        x : jnp.ndarray
            Input samples of shape (batch,) or (batch, n_in). Only used to 
            determine output shape.
        **kwargs
            Ignored for uniform density.
        
        Returns
        -------
        jnp.ndarray
            Array of ones with shape (batch,).
        """
        return jnp.ones(x.shape[0])


class LossDensity(BaseImportanceDensity):
    """
    Loss-based importance density - higher weights where loss is high.
    
    This IDF assigns importance proportional to the per-sample loss.
    The intuition is that regions with high loss need finer resolution
    to better fit the function.
    
    Parameters
    ----------
    epsilon : float
        Small constant added to avoid zero weights.
    
    Examples
    --------
    >>> density = LossDensity()
    >>> x = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> per_sample_loss = jnp.array([0.1, 0.5, 0.2])  # shape (batch,)
    >>> weights = density.compute(x, losses=per_sample_loss)
    >>> print(weights)  # Loss broadcast to all dimensions
    [[0.1 0.1]
     [0.5 0.5]
     [0.2 0.2]]
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize loss density.
        
        Parameters
        ----------
        epsilon : float, optional
            Small constant added to weights to avoid zeros. Default is 1e-8.
        """
        self.epsilon = epsilon
    
    def compute(self, x: jnp.ndarray, losses: jnp.ndarray = None, **kwargs) -> jnp.ndarray:
        """
        Compute weights proportional to per-sample loss.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input samples of shape (batch,) or (batch, n_in).
        losses : jnp.ndarray, optional
            Per-sample loss of shape (batch,). If not provided, falls back
            to uniform weights.
        **kwargs
            Additional ignored arguments.
        
        Returns
        -------
        jnp.ndarray
            Importance weights of shape (batch,). Each sample gets a weight
            proportional to its loss.
        """
        if losses is None:
            return jnp.ones(x.shape[0])
        
        # Return 1D weights: (batch,)
        return losses + self.epsilon


class CurvatureDensity(BaseImportanceDensity):
    """
    Curvature-based importance density - higher weights where curvature is high.
    
    This IDF assigns importance proportional to function curvature.
    The intuition is that highly curved regions need finer resolution.
    
    Parameters
    ----------
    epsilon : float
        Small constant added to avoid zero weights.
    
    Examples
    --------
    >>> density = CurvatureDensity()
    >>> x = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> curvatures = jnp.array([0.1, 0.5, 0.2])  # shape (batch,)
    >>> weights = density.compute(x, curvatures=curvatures)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize curvature density.
        
        Parameters
        ----------
        epsilon : float, optional
            Small constant added to weights to avoid zeros. Default is 1e-8.
        """
        self.epsilon = epsilon
    
    def compute(self, x: jnp.ndarray, curvatures: jnp.ndarray = None, **kwargs) -> jnp.ndarray:
        """
        Compute weights proportional to curvature.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input samples of shape (batch,) or (batch, n_in).
        curvatures : jnp.ndarray, optional
            Per-sample curvature of shape (batch,). If not provided, falls back
            to uniform weights.
        **kwargs
            Additional ignored arguments.
        
        Returns
        -------
        jnp.ndarray
            Importance weights of shape (batch,).
        """
        if curvatures is None:
            return jnp.ones(x.shape[0])
        
        return curvatures + self.epsilon


class ErrorGradientDensity(BaseImportanceDensity):
    """
    Error gradient density - higher weights where approximation error changes rapidly.
    
    This IDF assigns importance based on the gradient of the error (residual)
    with respect to inputs. Regions where the error changes rapidly indicate
    transitions between well-fit and poorly-fit regions, which benefit from
    finer knot resolution.
    
    Mathematical Justification
    --------------------------
    The error gradient |∇(y - ŷ)| identifies "edges" in the error surface.
    These are boundaries between regions where the model fits well and where
    it fits poorly. Placing knots at these boundaries helps the model
    transition smoothly and reduces overall error.
    
    Parameters
    ----------
    epsilon : float
        Small constant added to avoid zero weights.
    
    Examples
    --------
    >>> density = ErrorGradientDensity()
    >>> x = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> error_grads = jnp.array([0.1, 0.5, 0.2])  # shape (batch,)
    >>> weights = density.compute(x, error_gradients=error_grads)
    
    Notes
    -----
    Error gradients can be computed via finite differences or autodiff:
    
    .. code-block:: python
    
        def compute_error_gradient(model, x, y, epsilon=1e-3):
            # Finite difference approximation
            error = jnp.abs(y - model(x))  # shape (batch,)
            
            # For 1D input, perturb and compute gradient
            error_plus = jnp.abs(y - model(x + epsilon))
            error_minus = jnp.abs(y - model(x - epsilon))
            grad_error = jnp.abs(error_plus - error_minus) / (2 * epsilon)
            
            return grad_error  # shape (batch,)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize error gradient density.
        
        Parameters
        ----------
        epsilon : float, optional
            Small constant added to weights to avoid zeros. Default is 1e-8.
        """
        self.epsilon = epsilon
    
    def compute(self, x: jnp.ndarray, error_gradients: jnp.ndarray = None, **kwargs) -> jnp.ndarray:
        """
        Compute weights based on error gradient magnitude.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input samples of shape (batch,) or (batch, n_in).
        error_gradients : jnp.ndarray, optional
            Gradient of error w.r.t. inputs, shape (batch,) or (batch, n_in).
            If not provided, falls back to uniform weights.
        **kwargs
            Additional ignored arguments.
        
        Returns
        -------
        jnp.ndarray
            Importance weights of shape (batch,).
        
        Notes
        -----
        If error_gradients is 2D, we sum absolute values across dimensions.
        """
        if error_gradients is None:
            return jnp.ones(x.shape[0])
        
        # Sum gradient magnitudes across dimensions if 2D
        if error_gradients.ndim == 2:
            return jnp.sum(jnp.abs(error_gradients), axis=1) + self.epsilon
        else:
            return jnp.abs(error_gradients) + self.epsilon
