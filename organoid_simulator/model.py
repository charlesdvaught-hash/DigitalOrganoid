import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import logging

logger = logging.getLogger(__name__)

def power_iteration_spectral_radius(W, maxiter=200, tol=1e-5):
    """
    Estimate the spectral radius of sparse matrix W using power iteration.
    Because W is non-normal and might have a complex conjugate leading pair,
    we estimate the spectral radius using the ratio of successive vector norms
    (||x_{k+1}|| / ||x_k||) averaged over the last few iterations.
    """
    N = W.shape[0]
    # Initialize a random vector
    np.random.seed(42)  # For reproducibility in spectral radius fallback
    x = np.random.randn(N).astype(np.float32)
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        return 0.0
    x = x / norm_x

    norms = []
    for i in range(maxiter):
        x_next = W.dot(x)
        norm_next = np.linalg.norm(x_next)
        if norm_next == 0:
            return 0.0
        ratio = norm_next / np.linalg.norm(x)
        norms.append(ratio)
        x = x_next / norm_next
        
        # Convergence check on successive ratios
        if len(norms) > 5:
            recent_ratios = norms[-5:]
            if np.max(recent_ratios) - np.min(recent_ratios) < tol:
                break

    # Average the last few ratios to smooth out oscillations from complex leading eigenvalues
    return float(np.mean(norms[-5:])) if len(norms) > 0 else 0.0

def compute_spectral_radius(W):
    """
    Compute the spectral radius (magnitude of largest eigenvalue) of sparse matrix W.
    Primarily uses scipy.sparse.linalg.eigs, falling back to power iteration if it fails or is slow.
    """
    # Cast to float64 for eigs to be numerically stable, then return float
    W_64 = W.astype(np.float64)
    try:
        # eigs is primary
        vals = spla.eigs(W_64, k=1, which='LM', return_eigenvectors=False, maxiter=500, tol=1e-4)
        return float(np.abs(vals[0]))
    except Exception as e:
        logger.warning(f"eigs failed or did not converge: {e}. Falling back to power iteration.")
        return power_iteration_spectral_radius(W_64, maxiter=200, tol=1e-5)

def build_recurrent_weights(N, K, topology='random', lambda_ring=0.1, seed=None):
    """
    Build sparse weight matrix W_rec satisfying Dale's Law (80% excitatory, 20% inhibitory by columns).
    Exactly K outgoing connections per column (neuron).
    
    Parameters:
    - N: number of neurons (will round to nearest integer, and enforce Dale's split)
    - K: number of connections per neuron (out-degree)
    - topology: 'random' or 'ring'
    - lambda_ring: decay scale for ring topology (fraction of N)
    - seed: random seed
    
    Returns:
    - W_rec: scipy.sparse.csc_matrix (or csr_matrix) of shape (N, N) with sign-constrained columns
    - neuron_types: 1D array of shape (N,) with +1 for Exc and -1 for Inh
    """
    if seed is not None:
        np.random.seed(seed)
        
    K = min(K, N - 1)
    
    # 1. Enforce Dale's Law: 80% E, 20% I
    n_exc = int(round(0.8 * N))
    n_inh = N - n_exc
    
    # Randomly assign E/I identities across columns
    identities = np.ones(N, dtype=np.float32)
    inh_indices = np.random.choice(N, size=n_inh, replace=False)
    identities[inh_indices] = -1.0
    
    # 2. Build adjacency structure
    # For CSC: column pointers (indptr) and row indices (indices)
    # Since each column has exactly K non-zero outgoing connections, CSC is perfect for construction.
    indptr = np.arange(0, N * K + 1, K, dtype=np.int32)
    indices = np.zeros(N * K, dtype=np.int32)
    data = np.zeros(N * K, dtype=np.float32)
    
    if topology == 'ring':
        # Calculate distance-decay probabilities on a 1D ring
        # For neuron j, distance to neuron i is min(|i - j|, N - |i - j|)
        # Probability P(i, j) \propto exp(-d(i, j) / \lambda)
        lambda_val = lambda_ring * N
        
        # We can precompute the decay weights for offsets from 1 to N-1
        offsets = np.arange(1, N)
        dists = np.minimum(offsets, N - offsets)
        decay_weights = np.exp(-dists / lambda_val)
        prob_dist = decay_weights / np.sum(decay_weights)
    else:
        prob_dist = None
        
    for j in range(N):
        # We sample exactly K targets for column j, excluding j itself
        if topology == 'ring':
            # Relative targets (excluding self offset = 0)
            chosen_offsets = np.random.choice(offsets, size=K, replace=False, p=prob_dist)
            # Map back to absolute targets
            chosen_targets = (j + chosen_offsets) % N
        else:
            # Uniform random-sparse sampling excluding self
            candidates = np.delete(np.arange(N), j)
            chosen_targets = np.random.choice(candidates, size=K, replace=False)
            
        # Draw non-zero magnitudes from a half-normal distribution |N(0, 1)|
        magnitudes = np.abs(np.random.randn(K).astype(np.float32))
        
        # Apply sign corresponding to the source column (neuron j's identity)
        col_sign = identities[j]
        values = magnitudes * col_sign
        
        start_idx = j * K
        end_idx = start_idx + K
        
        # Store in arrays
        indices[start_idx:end_idx] = chosen_targets
        data[start_idx:end_idx] = values
        
    # Build CSC matrix
    W_rec = sp.csc_matrix((data, indices, indptr), shape=(N, N), dtype=np.float32)
    # Convert to CSR for faster matrix-vector multiplication in dynamics simulation
    W_rec = W_rec.tocsr()
    
    return W_rec, identities

def scale_spectral_radius(W_rec, target_rho):
    """
    Scale W_rec to have target spectral radius.
    """
    if target_rho is None:
        return W_rec
    
    rho = compute_spectral_radius(W_rec)
    if rho > 0:
        W_rec = W_rec.multiply(target_rho / rho)
    else:
        logger.warning("Computed spectral radius is 0; W_rec was not scaled.")
    return W_rec

def run_simulation(W_rec, W_in, u_seq, a=0.3, x_init=None):
    """
    Simulate the network dynamics:
    x(t+1) = (1 - a) * x(t) + a * tanh(W_rec * x(t) + W_in * u(t))
    
    Parameters:
    - W_rec: sparse recurrent weight matrix (CSR, shape N x N)
    - W_in: input weight matrix (shape N x 1)
    - u_seq: 1D array of input signals, shape (T,)
    - a: leak rate (default 0.3)
    - x_init: initial state vector (shape N,), default zeros
    
    Returns:
    - states: numpy array of shape (T, N) of float32 containing all state histories
    """
    N = W_rec.shape[0]
    T = len(u_seq)
    
    if x_init is None:
        x = np.zeros(N, dtype=np.float32)
    else:
        x = np.array(x_init, dtype=np.float32)
        
    states = np.zeros((T, N), dtype=np.float32)
    
    # Cast matrices to float32
    W_rec = W_rec.astype(np.float32)
    W_in = W_in.reshape(N, 1).astype(np.float32)
    u_seq = u_seq.astype(np.float32)
    
    one_minus_a = 1.0 - a
    
    for t in range(T):
        # We compute the linear term: W_rec * x + W_in * u(t)
        # scipy sparse matrix-vector dot is fast.
        linear_drive = W_rec.dot(x) + W_in[:, 0] * u_seq[t]
        x = one_minus_a * x + a * np.tanh(linear_drive)
        states[t, :] = x
        
    return states

def run_homeostasis(W_rec, target_mean=0.1, a=0.3, num_iters=10, probe_steps=500, seed=123):
    """
    Offline homeostasis mechanism using binary search.
    Tunes the input gain (scale of W_in) to achieve a target mean activity level:
    mean(|x(t)|) across all units and timesteps ≈ target_mean.
    
    W_in is initialized as a dense N x 1 matrix, drawn from uniform(-1, 1).
    We scale W_in to find the target activity. W_rec is kept exactly constant.
    
    Returns:
    - W_in: adjusted input weight matrix of shape (N, 1)
    - history: list of (scale, measured_mean_activity)
    """
    N = W_rec.shape[0]
    np.random.seed(seed)
    
    # Initialize baseline W_in
    W_in_base = np.random.uniform(-1.0, 1.0, size=(N, 1)).astype(np.float32)
    
    # Generate a separate random probe sequence (i.i.d. Uniform(-1, 1))
    u_probe = np.random.uniform(-1.0, 1.0, size=probe_steps).astype(np.float32)
    
    # Binary search limits for the input scale
    low = 1e-5
    high = 100.0
    best_scale = 1.0
    history = []
    
    for iteration in range(num_iters):
        mid = (low + high) / 2.0
        W_in_test = W_in_base * mid
        
        states = run_simulation(W_rec, W_in_test, u_probe, a=a)
        
        # Calculate mean absolute activity (excluding first 50 steps as brief warm-up)
        # Note: Jaegar or other conventions suggest using the entire probe, but skipping 50 is safer.
        mean_act = np.mean(np.abs(states[50:]))
        history.append((mid, mean_act))
        
        if mean_act < target_mean:
            # We need more activity -> increase scale
            low = mid
            best_scale = mid
        else:
            # We need less activity -> decrease scale
            high = mid
            best_scale = mid
            
        if abs(mean_act - target_mean) < 0.01 * target_mean:
            break
            
    logger.info(f"Homeostasis finished. Chosen scale: {best_scale:.4f}, achieved mean absolute activity: {mean_act:.4f}")
    return W_in_base * best_scale, history
