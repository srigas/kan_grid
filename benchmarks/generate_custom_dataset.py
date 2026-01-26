"""
Generate Custom Function Training Datasets
===========================================

This script generates pre-sampled training datasets for all custom benchmark functions.

Each .npz file contains:
- X: Training inputs (4000, n_in)
- y: Training outputs (4000, 1)

metadata.json contains bounds and function info for all functions.
"""

import jax.numpy as jnp
import jax
import numpy as np
import json
import os
from pathlib import Path

# Import function registry
from custom_funcs import CUSTOM_FUNCTIONS, get_function_info


# =============================================================================
# Uniform Sampling
# =============================================================================

def sample_uniform(func, bounds, n_samples, seed):
    
    key = jax.random.PRNGKey(seed)
    n_in = len(bounds)
    
    keys = jax.random.split(key, n_in)
    X_list = []
    for i, (min_val, max_val) in enumerate(bounds):
        x_i = jax.random.uniform(keys[i], (n_samples,), minval=min_val, maxval=max_val)
        X_list.append(x_i)
    
    X = jnp.stack(X_list, axis=1)
    y = func(X)
    
    return np.array(X), np.array(y)


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_datasets(output_dir='benchmarks/custom_funcs_data', n_samples=4000, seed=42):
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating custom function datasets...")
    print(f"Output directory: {output_path}")
    print(f"Samples per dataset: {n_samples}")
    print("=" * 80)
    
    metadata = {}
    
    for func_id in sorted(CUSTOM_FUNCTIONS.keys()):
        info = get_function_info(func_id)
        func = info['function']
        bounds = info['bounds']
        
        print(f"\n{func_id}: {info['name']} ({info['n_in']}D)")
        
        # Generate dataset
        print(f"  Generating samples...", end=" ")
        X_uniform, y_uniform = sample_uniform(func, bounds, n_samples, seed + int(func_id[1:]))
        
        uniform_file = output_path / f"{func_id}.npz"
        np.savez(uniform_file, X=X_uniform, y=y_uniform)
        print(f"✓ Saved to {uniform_file.name}")
        
        # Store metadata
        metadata[func_id] = {
            'name': info['name'],
            'category': info['category'],
            'n_in': info['n_in'],
            'bounds': bounds,
            'description': info['description']
        }
    
    # Save metadata
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Dataset generation complete!")
    print(f"Metadata saved to {metadata_file.name}")
    print(f"\nTotal files generated: {len(CUSTOM_FUNCTIONS)} datasets + 1 metadata file")


if __name__ == '__main__':
    generate_datasets()
