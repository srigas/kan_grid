"""
Custom Benchmark Functions for Grid Adaptation Testing
=======================================================

This module provides a curated set of test functions specifically designed to
evaluate grid adaptation strategies in KAN networks. 

Notes
-----
All functions follow the convention:
- Input: x of shape (batch, n_in)
- Output: y of shape (batch, 1)
- Bounds are provided as list of (min, max) tuples per dimension
"""

import jax.numpy as jnp


# =============================================================================
# Category 1: Localized Features
# =============================================================================

def f1(x):
    """
    Sharp Gaussian Peak (2D)
    
    A sharp localized peak on a smooth sinusoidal background.
    Peak location: (0.3, 0.7)
    Peak amplitude: 100
    Background amplitude: ~5
    
    Characteristics:
    - Very high curvature at peak center
    - Smooth everywhere else
    - Output range: approximately [-5, 105]
    """
    x1, x2 = x[:, [0]], x[:, [1]]
    peak = 100.0 * jnp.exp(-100 * ((x1 - 0.3)**2 + (x2 - 0.7)**2))
    background = 5.0 * jnp.sin(2 * jnp.pi * x1) * jnp.cos(2 * jnp.pi * x2)
    return peak + background


def f2(x):
    """
    Step Function (1D)
    
    A smoothed step function (tanh) with adjustable steepness.
    Transition at x = 0 with steepness parameter k = 50.
    
    Characteristics:
    - Near-discontinuity at x = 0
    - Flat plateaus away from transition
    - Output range: [-1, 1]
    """
    k = 50.0  # Steepness
    return jnp.tanh(k * x[:, [0]])


def f3(x):
    """
    Multiple Peaks (2D)
    
    Three Gaussian peaks at different locations with different amplitudes.
    Peak locations: (0.2, 0.2), (0.8, 0.3), (0.5, 0.8)
    Peak amplitudes: 50, 30, 80
    
    Characteristics:
    - Multiple localized features
    - Varying peak widths and heights
    - Output range: approximately [0, 80]
    """
    x1, x2 = x[:, [0]], x[:, [1]]
    peak1 = 50.0 * jnp.exp(-80 * ((x1 - 0.2)**2 + (x2 - 0.2)**2))
    peak2 = 30.0 * jnp.exp(-60 * ((x1 - 0.8)**2 + (x2 - 0.3)**2))
    peak3 = 80.0 * jnp.exp(-100 * ((x1 - 0.5)**2 + (x2 - 0.8)**2))
    return peak1 + peak2 + peak3


def f4(x):
    """
    Runge-like Function (1D)
    
    The classic Runge function that causes polynomial interpolation to fail.
    Modified to have a sharper peak for enhanced difficulty.
    
    Formula: 1 / (1 + 25*x^2)
    
    Characteristics:
    - Sharp peak at x = 0
    - Rapid decay to near-zero
    - Output range: [~0.04, 1] on [-1, 1]
    """
    return 1.0 / (1.0 + 25.0 * x[:, [0]]**2)


# =============================================================================
# Category 2: Multi-scale Function
# =============================================================================

def f5(x):
    """
    Chirp Function (1D)
    
    Sinusoid with frequency that increases along the domain.
    Formula: sin(π * x^2 * 10)
    
    Characteristics:
    - Low frequency near x = 0
    - High frequency near x = 1
    - Spatially varying resolution requirements
    - Output range: [-1, 1]
    """
    return jnp.sin(jnp.pi * x[:, [0]]**2 * 10)



# =============================================================================
# Category 3: Boundary Layers
# =============================================================================

def f6(x):
    """
    Boundary Layer (1D)
    
    Function with rapid transition near the boundary (x = 1).
    Common in fluid dynamics and singular perturbation problems.
    Formula: x - exp(-50*(1-x))
    
    Characteristics:
    - Nearly linear for most of domain
    - Sharp exponential decay near x = 1
    - Output range: approximately [0, 1]
    """
    x1 = x[:, [0]]
    return x1 - jnp.exp(-50 * (1 - x1))


def f7(x):
    """
    Corner Singularity (2D)
    
    Function with singularity-like behavior at the corner (0, 0).
    Formula: sqrt(x1^2 + x2^2 + 0.01)
    
    Characteristics:
    - High curvature near origin
    - Smooth cone-like shape elsewhere
    - Small epsilon (0.01) to avoid exact singularity
    - Output range: [0.1, ~1.4] on [0,1]^2
    """
    x1, x2 = x[:, [0]], x[:, [1]]
    return jnp.sqrt(x1**2 + x2**2 + 0.01)


def f8(x):
    """
    Double Boundary Layer (1D)
    
    Function with rapid transitions at BOTH boundaries.
    Formula: tanh(30*x) + tanh(30*(x-1))
    
    Characteristics:
    - Sharp transitions at x = 0 and x = 1
    - Flat in the middle
    - Output range: approximately [-2, 0]
    """
    x1 = x[:, [0]]
    return jnp.tanh(30 * x1) + jnp.tanh(30 * (x1 - 1))


# =============================================================================
# Category 4: Higher Dimensional
# =============================================================================

def f9(x):
    """
    6D Sparse Interaction (6D)
    
    Function that depends strongly on only some dimension pairs.
    Formula: sin(π*x1*x2) + cos(π*x3*x4) + x5*x6
    
    Characteristics:
    - Sparse interaction structure
    - Tests whether adaptation finds important dimensions
    - Output range: approximately [-3, 3]
    """
    return (jnp.sin(jnp.pi * x[:, [0]] * x[:, [1]]) + 
            jnp.cos(jnp.pi * x[:, [2]] * x[:, [3]]) + 
            x[:, [4]] * x[:, [5]])


def f10(x):
    """
    4D Ridge (4D)
    
    Function with a ridge along a diagonal in 4D space.
    Ridge along direction (1,1,1,1)/2.
    
    Characteristics:
    - 1D feature embedded in 4D
    - Sharp in orthogonal directions
    - Output range: approximately [0, 10]
    """
    # Distance from the diagonal (1,1,1,1)
    mean = (x[:, [0]] + x[:, [1]] + x[:, [2]] + x[:, [3]]) / 4
    dist_sq = ((x[:, [0]] - mean)**2 + (x[:, [1]] - mean)**2 + 
               (x[:, [2]] - mean)**2 + (x[:, [3]] - mean)**2)
    return 10.0 * jnp.exp(-20 * dist_sq)


# =============================================================================
# Function Registry
# =============================================================================

CUSTOM_FUNCTIONS = {
    'f1': {
        'function': f1,
        'name': 'Sharp Peak 2D',
        'category': 'localized',
        'n_in': 2,
        'bounds': [(-1.0, 1.0), (-1.0, 1.0)],
        'description': 'Sharp Gaussian peak on smooth background'
    },
    'f2': {
        'function': f2,
        'name': 'Step Function 1D',
        'category': 'localized',
        'n_in': 1,
        'bounds': [(-1.0, 1.0)],
        'description': 'Smoothed step function with steep transition'
    },
    'f3': {
        'function': f3,
        'name': 'Multiple Peaks 2D',
        'category': 'localized',
        'n_in': 2,
        'bounds': [(0.0, 1.0), (0.0, 1.0)],
        'description': 'Three Gaussian peaks at different locations'
    },
    'f4': {
        'function': f4,
        'name': 'Runge Function 1D',
        'category': 'localized',
        'n_in': 1,
        'bounds': [(-1.0, 1.0)],
        'description': 'Classic Runge function with sharp central peak'
    },
    'f5': {
        'function': f5,
        'name': 'Chirp Function 1D',
        'category': 'multiscale',
        'n_in': 1,
        'bounds': [(0.0, 1.0)],
        'description': 'Sinusoid with spatially varying frequency'
    },
    'f6': {
        'function': f6,
        'name': 'Boundary Layer 1D',
        'category': 'boundary',
        'n_in': 1,
        'bounds': [(0.0, 1.0)],
        'description': 'Sharp exponential transition near boundary'
    },
    'f7': {
        'function': f7,
        'name': 'Corner Singularity 2D',
        'category': 'boundary',
        'n_in': 2,
        'bounds': [(0.0, 1.0), (0.0, 1.0)],
        'description': 'High curvature near corner origin'
    },
    'f8': {
        'function': f8,
        'name': 'Double Boundary Layer 1D',
        'category': 'boundary',
        'n_in': 1,
        'bounds': [(0.0, 1.0)],
        'description': 'Sharp transitions at both boundaries'
    },
    'f9': {
        'function': f9,
        'name': '6D Sparse Interaction',
        'category': 'highdim',
        'n_in': 6,
        'bounds': [(0.0, 1.0)] * 6,
        'description': 'Sparse pairwise interactions in 6D'
    },
    'f10': {
        'function': f10,
        'name': '4D Ridge',
        'category': 'highdim',
        'n_in': 4,
        'bounds': [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        'description': '1D ridge structure embedded in 4D'
    },
}


def get_function_info(func_id: str) -> dict:
    
    if func_id not in CUSTOM_FUNCTIONS:
        raise ValueError(f"Unknown function ID: {func_id}. "
                        f"Available: {list(CUSTOM_FUNCTIONS.keys())}")
    return CUSTOM_FUNCTIONS[func_id]


def list_functions_by_category(category: str = None) -> list:

    if category is None:
        return list(CUSTOM_FUNCTIONS.keys())
    
    return [fid for fid, info in CUSTOM_FUNCTIONS.items() 
            if info['category'] == category]


def get_all_categories() -> list:
    
    return ['localized', 'multiscale', 'boundary', 'highdim']
