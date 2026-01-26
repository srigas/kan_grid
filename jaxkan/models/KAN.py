from jax import numpy as jnp

from flax import nnx

from ..layers import get_layer
from ..grids import (
    BaseImportanceDensity,
    BaseAdaptationStrategy,
    BoundsCalculator
)
from .utils import normalize_to_list

from typing import Union, List


class KAN(nnx.Module):
    """
    KAN class, corresponding to a network of KAN Layers.

    Attributes:
        layers (nnx.List):
            List of KAN layer instances.
    
    Grid Adaptation
    ---------------
    Grid adaptation is performed via the `update_grids()` method. The decision
    of **when** to adapt is left to the user's training loop, allowing maximum
    flexibility. The `src.grids` module provides utilities for this:
    
    - **State tracking**: `AdaptationState`, `update_state()`, `reset_after_adaptation()`
    - **Triggers**: `PeriodicTrigger`, `LossPlateauTrigger`, `ScheduledTrigger`
    - **IDFs**: `UniformDensity`, `LossDensity`, `CurvatureDensity`, etc.
    - **Strategies**: `UniformAdaptation`, `QuantileAdaptation`, `MixedAdaptation`
    
    Example training loop with automatic triggering::
    
        from src.grids import (
            AdaptationState, update_state, reset_after_adaptation,
            LossPlateauTrigger, LossDensity
        )
        
        model = KAN(layer_dims=[2, 5, 1], layer_type='spline', 
                    required_parameters={'k': 3, 'G': 5})
        
        state = AdaptationState()
        trigger = LossPlateauTrigger(patience=50, cooldown=10)
        
        for epoch in range(1000):
            loss = train_step(...)
            state = update_state(state, loss)
            
            if trigger(state):
                model.update_grids(x_batch, idf=LossDensity(), losses=per_sample_loss)
                state = reset_after_adaptation(state)
    """
    
    def __init__(self, layer_dims: List[int], layer_type: str = "spline",
                 required_parameters: Union[None, dict] = None, seed: int = 42
                ):
        """
        Initializes a KAN model.

        Args:
            layer_dims (List[int]):
                Defines the network in terms of nodes. E.g. [4,5,1] is a network with 2 layers: one with n_in=4 and n_out=5 and one with n_in=5 and n_out = 1.
            layer_type (str):
                Type of layer to use (e.g., 'spline').
            required_parameters (dict):
                Dictionary containing parameters required for the chosen layer type.
            seed (int):
                Random key selection for initializations wherever necessary.
                
        Example:
            >>> req_params = {'k': 3, 'G': 5}
            >>> model = KAN(layer_dims = [2,5,1], layer_type='spline', required_parameters=req_params, seed=42)
        """
        # Get the corresponding layer class based on layer_type
        LayerClass = get_layer(layer_type.lower())
            
        if required_parameters is None:
            raise ValueError("required_parameters must be provided as a dictionary for the selected layer_type.")
        
        self.layers = nnx.List([
                LayerClass(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    **required_parameters,
                    seed=seed
                )
                for i in range(len(layer_dims) - 1)
            ])
    
    def update_grids(
        self, 
        x: jnp.ndarray,
        idf: Union[BaseImportanceDensity, List[BaseImportanceDensity], None] = None,
        strategy: Union[BaseAdaptationStrategy, List[BaseAdaptationStrategy], None] = None,
        bounds_calculator: Union[BoundsCalculator, List[BoundsCalculator], None] = None,
        grid_size_new: Union[int, List[int], None] = None,
        degree_new: Union[int, List[int], None] = None,
        **idf_kwargs
    ) -> None:
        """
        Performs the grid update for each layer of the KAN architecture.
        
        Each parameter can be specified as either:
        - A single value: Applied to all layers
        - A list: One value per layer (length must match number of layers)
        
        This method supports two types of layers:
        - **Grid-based layers** (SplineLayer, RBFLayer): Use the full adaptation
          framework with IDF, strategy, bounds, and grid_size_new.
        - **Degree-based layers** (ChebyshevLayer, LegendreLayer, FourierLayer, 
          SineLayer): Only support polynomial degree extension via degree_new.

        Args:
            x (jnp.array):
                Inputs for the first layer, shape (batch, n_in).
            idf (BaseImportanceDensity or List, optional):
                Importance Density Function that determines where knots should be
                concentrated. If None, uses UniformDensity() for all layers.
                Can be a single IDF (applied to all layers) or a list of IDFs.
                **Only applies to grid-based layers (SplineLayer, RBFLayer).**
                
                Common choices:
                - UniformDensity(): Equal weighting everywhere (default)
                - LossDensity(): Weight by per-sample loss (requires losses= kwarg)
                - CurvatureDensity(): Weight by function curvature
                
            strategy (BaseAdaptationStrategy or List, optional):
                Strategy for computing new grid positions. If None, uses
                MixedAdaptation with each layer's grid_e parameter.
                Can be a single strategy (applied to all layers) or a list.
                **Only applies to grid-based layers (SplineLayer, RBFLayer).**
                
                Common choices:
                - UniformAdaptation(): Evenly-spaced grid points
                - QuantileAdaptation(): Grid points at weighted quantiles
                - MixedAdaptation(grid_e): Blend of uniform and quantile
                
            bounds_calculator (BoundsCalculator or List, optional):
                Calculator for determining grid bounds from data. If None, uses
                BoundsCalculator(margin=0.01) for all layers.
                **Only applies to grid-based layers (SplineLayer, RBFLayer).**
                
            grid_size_new (int or List, optional):
                New grid size (G for splines, D for RBFs). If None, keeps current
                grid size and only adapts knot/center positions.
                Can be a single int (applied to all layers) or a list of ints.
                **Only applies to grid-based layers (SplineLayer, RBFLayer).**
                
            degree_new (int or List, optional):
                New polynomial degree. Can be a single int or a list (one per layer).
                If None, keeps current degrees.
                **Only applies to degree-based layers (ChebyshevLayer, LegendreLayer).**
                
            **idf_kwargs:
                Additional keyword arguments passed to the IDF's compute() method.
                These are passed to ALL layers that use an IDF requiring extra data.
                
                Common kwargs:
                - losses: Per-sample loss values for LossDensity
                - curvatures: Curvature estimates for CurvatureDensity
                - error_gradients: Error gradient magnitudes for ErrorGradientDensity
            
        Examples:
            >>> # Spline/RBF layers - full grid adaptation
            >>> req_params = {'k': 3, 'G': 5}
            >>> model = KAN(layer_dims=[2, 5, 1], layer_type='spline', 
            ...             required_parameters=req_params, seed=42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> # Basic adaptation (uses UniformDensity for all layers)
            >>> model.update_grids(x=x_batch)
            >>>
            >>> # Loss-weighted adaptation for all layers
            >>> model.update_grids(x=x_batch, idf=LossDensity(), losses=per_sample_loss)
            >>>
            >>> # Different IDF per layer
            >>> model.update_grids(
            ...     x=x_batch,
            ...     idf=[LossDensity(), UniformDensity()],
            ...     losses=per_sample_loss
            ... )
            >>>
            >>> # Grid extension
            >>> model.update_grids(x=x_batch, grid_size_new=[8, 12])
            >>>
            >>> # Chebyshev/Legendre layers - degree extension only
            >>> req_params = {'D': 5}
            >>> model = KAN(layer_dims=[2, 5, 1], layer_type='chebyshev', 
            ...             required_parameters=req_params, seed=42)
            >>> model.update_grids(x=x_batch, degree_new=10)
        """
        num_layers = len(self.layers)
        
        # Normalize all parameters to per-layer lists
        idf_list = normalize_to_list(idf, num_layers, "idf")
        strategy_list = normalize_to_list(strategy, num_layers, "strategy")
        bounds_calc_list = normalize_to_list(bounds_calculator, num_layers, "bounds_calculator")
        grid_size_list = normalize_to_list(grid_size_new, num_layers, "grid_size_new")
        degree_list = normalize_to_list(degree_new, num_layers, "degree_new")

        # Loop over each layer
        for i, layer in enumerate(self.layers):
            
            # Grid-based layers (SplineLayer, RBFLayer) - full adaptation framework
            if hasattr(layer, 'update_grid'):
                layer.update_grid(
                    x,
                    idf=idf_list[i],
                    strategy=strategy_list[i],
                    bounds_calculator=bounds_calc_list[i],
                    grid_size_new=grid_size_list[i],
                    **idf_kwargs
                )
            
            # Degree-based layers (ChebyshevLayer, LegendreLayer) - only degree extension
            elif hasattr(layer, 'update_degree'):
                # Only call if degree_new is specified for this layer
                if degree_list[i] is not None:
                    layer.update_degree(x, degree_new=degree_list[i])

            # Perform a forward pass to get the input for the next layer
            x = layer(x)

    
    def __call__(self, x):
        """
        Equivalent to the network's forward pass.

        Args:
            x (jnp.array):
                Inputs for the first layer, shape (batch, self.layers[0]).

        Returns:
            x (jnp.array):
                Network output, shape (batch, self.layers[-1]).
            
        Example:
            >>> req_params = {'k': 3, 'G': 5}
            >>> model = KAN(layer_dims = [2,5,1], layer_type='base', required_parameters=req_params, seed=42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = model(x_batch)
        """

        # Pass through each layer of the KAN
        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x
