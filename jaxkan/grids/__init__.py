"""
Grid Adaptation Framework
=========================

This package provides a unified, extensible framework for grid adaptation in
Kolmogorov-Arnold Networks (KANs). It handles the "when", "where", and "how"
of adapting grid knots during training.

Package Structure
-----------------
types/
    Concrete grid implementations (SplineGrid, RBFGrid).
state.py
    State management for tracking training progress and trigger conditions.
triggers.py
    Trigger classes that determine WHEN adaptation should occur.
idf.py
    Importance Density Functions that determine WHERE knots should be placed.
bounds.py
    Utilities for computing and managing grid bounds from data.
adaptation.py
    Strategies for HOW to compute new knot positions.

Quick Start
-----------
1. **Create a trigger** to determine when adaptation occurs:

    >>> from src.grids.triggers import LossPlateauTrigger
    >>> trigger = LossPlateauTrigger(patience=50, rel_threshold=0.05)

2. **Create an IDF** to determine knot placement priorities:

    >>> from src.grids.idf import LossDensity
    >>> density = LossDensity()

3. **Create an adaptation strategy**:

    >>> from src.grids.adaptation import MixedAdaptation
    >>> strategy = MixedAdaptation(grid_e=0.1)

4. **Use in training loop**:

    >>> from src.grids.state import AdaptationState, update_state, reset_after_adaptation
    >>> 
    >>> state = AdaptationState()
    >>> 
    >>> for epoch in range(num_epochs):
    ...     loss = train_step(...)
    ...     state = update_state(state, loss)
    ...     
    ...     if trigger(state):
    ...         # Compute weights and adapt
    ...         weights = density(x, metric=per_sample_loss)
    ...         # ... perform adaptation ...
    ...         state = reset_after_adaptation(state)

Framework Components
--------------------

**Triggers** (When to adapt)
    - PeriodicTrigger: Every N epochs
    - ScheduledTrigger: At specific epochs
    - LossPlateauTrigger: When loss stops improving

**IDFs** (Where to place knots)
    - UniformDensity: Equal importance everywhere
    - LossDensity: Based on per-sample loss
    - CurvatureDensity: Based on function curvature
    - ErrorGradientDensity: Based on error gradient magnitude

**Strategies** (How to compute knots)
    - UniformAdaptation: Evenly-spaced knots
    - QuantileAdaptation: Knots at weighted quantiles
    - MixedAdaptation: Blend of uniform and quantile

**Grid Types**
    - SplineGrid: For B-spline layers (includes augmentation)
    - RBFGrid: For RBF layers (no augmentation)

Design Philosophy
-----------------
1. **Separation of Concerns**: Each component has a single responsibility.
2. **Composability**: Mix and match triggers, IDFs, and strategies.
3. **Extensibility**: Easy to add new triggers, IDFs, or strategies.
4. **Lightweight**: Minimal overhead during training.
5. **JAX-Friendly**: Compatible with JIT compilation where possible.
"""

# State management
from .state import (
    AdaptationState,
    update_state,
    reset_after_adaptation,
    epochs_since_adaptation
)

# Triggers
from .triggers import (
    BaseTrigger,
    PeriodicTrigger,
    ScheduledTrigger,
    LossPlateauTrigger,
    CompositeTrigger
)

# Importance Density Functions
from .idf import (
    BaseImportanceDensity,
    UniformDensity,
    LossDensity,
    CurvatureDensity,
    ErrorGradientDensity
)

# Bounds calculation
from .bounds import (
    BoundsCalculator,
    compute_initial_bounds
)

# Adaptation strategies
from .adaptation import (
    BaseAdaptationStrategy,
    UniformAdaptation,
    QuantileAdaptation,
    MixedAdaptation
)

# Grid types
from .types import SplineGrid, RBFGrid

__all__ = [
    # State
    "AdaptationState",
    "update_state", 
    "reset_after_adaptation",
    "epochs_since_adaptation",
    # Triggers
    "BaseTrigger",
    "PeriodicTrigger",
    "ScheduledTrigger",
    "LossPlateauTrigger",
    "CompositeTrigger",
    # IDFs
    "BaseImportanceDensity",
    "UniformDensity",
    "LossDensity",
    "CurvatureDensity",
    "ErrorGradientDensity",
    # Bounds
    "BoundsCalculator",
    "compute_initial_bounds",
    # Strategies
    "BaseAdaptationStrategy",
    "UniformAdaptation",
    "QuantileAdaptation",
    "MixedAdaptation",
    # Grid types
    "SplineGrid",
    "RBFGrid",
]
