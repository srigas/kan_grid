from .Spline import SplineLayer
from .Chebyshev import ChebyshevLayer
from .Legendre import LegendreLayer
from .Fourier import FourierLayer
from .RBF import RBFLayer
from .Sine import SineLayer
from .adaptive import GridAdaptiveMixin


def get_layer(layer_type: str):
    """
    Helper method that creates a mapping between layer type codes and the actual classes.

    Args:
        layer_type (str):
            Code of layer to be used.
            
    Returns:
        layer (jaxkan.layers.Layer):
            A jaxkan.layers layer class instance to be used as the building block of a KAN.
            
    Example:
        >>> LayerClass = get_layer("spline")
    """
    layer_map = {
        "spline": SplineLayer,
        "chebyshev": ChebyshevLayer,
        "legendre": LegendreLayer,
        "fourier": FourierLayer,
        "rbf": RBFLayer,
        "sine": SineLayer
    }
    
    if layer_type not in layer_map:
        raise ValueError(f"Unknown layer type: {layer_type}. Available types: {list(layer_map.keys())}")
        
    LayerClass = layer_map[layer_type]
        
    return LayerClass