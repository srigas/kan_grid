import jax
import jax.numpy as jnp


@jax.jit
def solve_single_lstsq(A_single, B_single):
    """
    Simulates linalg.lstsq by reformulating the problem AX = B via the normal equations: (A^T A) X = A^T B. This is used instead of linalg.lstsq because it's much faster.

    Args:
        A_single (jnp.array):
            Matrix A of AX = B, shape (M, N).
        B_single (jnp.array):
            Matrix B of AX = B, shape (M, K).
        
    Returns:
        single_solution (jnp.array):
            Matrix X of AX = B, shape (N, K).
            
    Example:
        >>> A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        >>> B = jnp.array([[1.0], [2.0]])
        >>>
        >>> solution = solve_single_lstsq(A, B)
    """
    
    AtA = jnp.dot(A_single.T, A_single)
    AtB = jnp.dot(A_single.T, B_single)
    single_solution = jax.scipy.linalg.solve(AtA, AtB, assume_a='pos')
    
    return single_solution


@jax.jit
def solve_full_lstsq(A_full, B_full):
    """
    Parallelizes the single case using vmap.

    Args:
        A_full (jnp.array):
            Matrix A of AX = B, shape (n_dims, M, N).
        B_full (jnp.array):
            Matrix B of AX = B, shape (n_dims, M, K).
        
    Returns:
        full_solution (jnp.array):
            Matrix X of AX = B, shape (n_dims, N, K).
            
    Example:
        >>> A = jnp.array([[[2.0, 1.0], [1.0, 3.0]], [[1.0, 2.0], [2.0, 1.0]]])
        >>> B = jnp.array([[[1.0], [2.0]], [[2.0], [3.0]]])
        >>>
        >>> solution = solve_full_lstsq(A, B)
    """
    solve_full = jax.vmap(solve_single_lstsq, in_axes=(0, 0))
    full_solution = solve_full(A_full, B_full)
    return full_solution


def solve_full_lstsq_batched(A_full, B_full, batch_size=1024):
    """
    Memory-efficient batched least squares solver that accumulates normal equations
    in chunks to avoid GPU shared memory limits.
    
    Instead of computing A^T @ A for the full batch at once, this accumulates
    the normal equation components (A^T A and A^T B) chunk by chunk, then solves.
    
    Args:
        A_full (jnp.array):
            Matrix A of AX = B, shape (n_dims, M, N) where M is sample size.
        B_full (jnp.array):
            Matrix B of AX = B, shape (n_dims, M, K).
        batch_size (int):
            Number of samples to process at a time. Default 1024.
        
    Returns:
        full_solution (jnp.array):
            Matrix X of AX = B, shape (n_dims, N, K).
            
    Example:
        >>> A = jnp.ones((4, 10000, 10))  # 4 dims, 10k samples, 10 basis funcs
        >>> B = jnp.ones((4, 10000, 5))   # 4 dims, 10k samples, 5 outputs
        >>> solution = solve_full_lstsq_batched(A, B, batch_size=2000)
    """
    n_dims, M, N = A_full.shape
    K = B_full.shape[-1]
    
    # Initialize accumulators for normal equations: A^T A and A^T B
    # Shape: (n_dims, N, N) and (n_dims, N, K)
    AtA_accum = jnp.zeros((n_dims, N, N), dtype=A_full.dtype)
    AtB_accum = jnp.zeros((n_dims, N, K), dtype=A_full.dtype)
    
    # Process in chunks
    num_chunks = (M + batch_size - 1) // batch_size
    
    for i in range(num_chunks):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, M)
        
        # Extract chunk: (n_dims, chunk, N) and (n_dims, chunk, K)
        A_chunk = A_full[:, start_idx:end_idx, :]
        B_chunk = B_full[:, start_idx:end_idx, :]
        
        # Accumulate A^T @ A and A^T @ B for this chunk
        # Using einsum for batched transpose-multiply
        AtA_accum = AtA_accum + jnp.einsum('ijk,ijl->ikl', A_chunk, A_chunk)
        AtB_accum = AtB_accum + jnp.einsum('ijk,ijl->ikl', A_chunk, B_chunk)
    
    # Solve the normal equations for each dimension
    def solve_normal(AtA, AtB):
        return jax.scipy.linalg.solve(AtA, AtB, assume_a='pos')
    
    full_solution = jax.vmap(solve_normal)(AtA_accum, AtB_accum)
    
    return full_solution
