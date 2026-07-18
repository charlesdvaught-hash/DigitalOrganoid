import numpy as np
import scipy.stats as stats
import logging
import powerlaw

logger = logging.getLogger(__name__)

def solve_ridge_multi(X, Y, alpha=1e-4):
    """
    Solve ridge regression for a matrix of multiple target columns Y using states X,
    optimized to avoid redundant matrix inversions/factorizations.
    
    Y has shape (T, k_max).
    X has shape (T, N).
    
    If N (number of features / neurons) < T (number of training samples),
    solve via Primal form: (X^T * X + alpha * I)^{-1} * X^T * Y
    
    If N >= T,
    solve via Dual form: X^T * (X * X^T + alpha * I)^{-1} * Y
    
    Returns:
    - W: weight matrix of shape (N, k_max)
    """
    T, N = X.shape
    k_max = Y.shape[1]
    
    # Cast to float64 for numerical stability during solve, then return weight as float32
    X_64 = X.astype(np.float64)
    Y_64 = Y.astype(np.float64)
    
    if N < T:
        # Primal form
        # (N x N)
        A = X_64.T.dot(X_64) + alpha * np.eye(N)
        b = X_64.T.dot(Y_64)
        W = np.linalg.solve(A, b)
    else:
        # Dual form
        # (T x T)
        A = X_64.dot(X_64.T) + alpha * np.eye(T)
        C = np.linalg.solve(A, Y_64)
        W = X_64.T.dot(C)
        
    return W.astype(np.float32)

def compute_memory_capacity(states, u_seq, k_max=50, alpha=1e-4, train_frac=0.8):
    """
    Compute reservoir memory capacity (MC) using vectorized multiple-target ridge regression.
    To avoid redundant inversions, we trim the first k_max states so that the feature matrix X
    remains completely identical across all delays k = 1..k_max.
    
    Parameters:
    - states: array of shape (T_total, N), simulated states
    - u_seq: array of shape (T_total,), driving input sequence
    - k_max: max delay (default 50)
    - alpha: ridge regularization parameter (default 1e-4)
    - train_frac: fraction of sequence to use for training (default 0.8)
    
    Returns:
    - total_mc: sum of squared correlations corr^2(target_k, prediction_k)
    - mc_curves: dict mapping delay k (1..k_max) to its r^2 capacity value
    """
    T_total, N = states.shape
    
    # Trim the first k_max steps so that for any delay k, we can align state[t] with u[t - k]
    # on the EXACT SAME state matrix slice X_aligned = states[k_max:]
    X_aligned = states[k_max:, :]
    L = len(X_aligned)
    
    # Construct target matrix Y of shape (L, k_max)
    # Column k-1 contains the target for delay k: u[k_max - k : -k]
    Y_aligned = np.zeros((L, k_max), dtype=np.float32)
    for k in range(1, k_max + 1):
        Y_aligned[:, k - 1] = u_seq[k_max - k : T_total - k]
        
    # Split temporally into train / test
    L_train = int(round(train_frac * L))
    
    X_train = X_aligned[:L_train, :]
    Y_train = Y_aligned[:L_train, :]
    
    X_test = X_aligned[L_train:, :]
    Y_test = Y_aligned[L_train:, :]
    
    # Solve for all readouts in a single vectorized call
    W = solve_ridge_multi(X_train, Y_train, alpha=alpha)
    
    # Predict on test set
    Y_pred = X_test.dot(W)
    
    total_mc = 0.0
    mc_curves = {}
    
    # Calculate R2 for each delay
    for k in range(1, k_max + 1):
        pred_col = Y_pred[:, k - 1]
        true_col = Y_test[:, k - 1]
        
        std_pred = np.std(pred_col)
        std_true = np.std(true_col)
        
        if std_pred > 1e-9 and std_true > 1e-9:
            corr = np.corrcoef(true_col, pred_col)[0, 1]
            r2 = (corr ** 2) if not np.isnan(corr) else 0.0
        else:
            r2 = 0.0
            
        mc_curves[k] = float(r2)
        total_mc += r2
        
    return float(total_mc), mc_curves

def compute_criticality_metrics(states, binarization_threshold_sigma=2.0, adaptive=True, min_avalanches=300):
    """
    Compute population avalanche metrics.
    
    1. Binarize per unit: unit i is "active" at t if |x_i(t)| > theta_i,
       where theta_i = mean(x_i) + binarization_threshold_sigma * std(x_i).
    2. Population activity A(t) = count of active units.
    3. Define avalanches as consecutive timesteps where A(t) exceeds its median.
       (If median is 0, we define threshold as 1).
    4. Compute avalanche size as sum of A(t) during the excursion.
    5. Fit the power-law exponent of the avalanche sizes using the `powerlaw` package.
       Report the fitted exponent alpha, Kolmogorov-Smirnov distance (D), xmin,
       and the loglikelihood-ratio vs. an exponential distribution.
    6. Compute the branching ratio:
       sigma = sum A(t+1) / sum A(t) summed over active avalanche periods (pooled-ratio estimator).
       
    Parameters:
    - states: array of shape (T, N), simulated states
    - binarization_threshold_sigma: multiplier for std to binarize states (default 2.0)
    - adaptive: if True, lower the threshold sigma (e.g., to 1.5) if detected avalanches < min_avalanches
    - min_avalanches: threshold to trigger adaptive binarization (default 300)
    
    Returns:
    - dict containing:
        - avalanche_exponent: float, MLE fitted exponent (or np.nan if fit failed)
        - branching_ratio: float, branching ratio sigma
        - num_avalanches: int, number of detected avalanches
        - threshold_sigma_used: float, the sigma threshold used
        - ks_distance: float, KS statistic of the powerlaw fit
        - xmin: float, the estimated lower bound for power-law behavior
        - log_likelihood_ratio: float, log likelihood ratio of power_law vs exponential (positive values favor powerlaw)
        - p_value: float, p-value of the log likelihood ratio
    """
    T, N = states.shape
    
    # Precompute mean and std per neuron
    means = np.mean(states, axis=0)
    stds = np.std(states, axis=0)
    
    sigma = binarization_threshold_sigma
    
    def detect_avalanches(s_thresh):
        # Vectorized thresholding
        theta = means + s_thresh * stds
        active = (np.abs(states) > theta).astype(np.int32)
        A = np.sum(active, axis=1) # Shape (T,)
        
        # Set global threshold as median of population activity
        global_threshold = np.median(A)
        if global_threshold < 1:
            global_threshold = 1.0
            
        # Detect active periods where A(t) > global_threshold
        is_active = A > global_threshold
        
        # Identify contiguous blocks of True values
        # We find transitions
        padded = np.zeros(T + 2, dtype=bool)
        padded[1:-1] = is_active
        
        # Starts are where padded transitions from False to True
        starts = np.where(~padded[:-1] & padded[1:])[0]
        # Ends are where padded transitions from True to False (note ends are exclusive index in padded)
        ends = np.where(padded[:-1] & ~padded[1:])[0]
        
        # Adjust ends back to the index in original is_active sequence (which is exclusive)
        # ends contains the index j in padded corresponding to the last True + 1 of an active block.
        # Since padded is offset by 1, ends is the exact exclusive end index in the 0-based is_active array.
        # Removing the subtraction of 1 fixes the under-counting correctness bug.
        ends_orig = ends
        
        avalanches = []
        sizes = []
        durations = []
        
        # To compute branching ratio:
        # pooled-ratio estimator: sigma_branch = sum_{t in active periods but not the end of an avalanche} A(t+1) / sum A(t)
        sum_A_t = 0.0
        sum_A_t_plus_1 = 0.0
        
        for start, end in zip(starts, ends_orig):
            # start is inclusive, end is exclusive
            av_slice = A[start:end]
            size = np.sum(av_slice)
            sizes.append(size)
            durations.append(end - start)
            avalanches.append((start, end))
            
            if len(av_slice) > 1:
                # We can compute branching ratio transitions within this avalanche
                sum_A_t += np.sum(av_slice[:-1])
                sum_A_t_plus_1 += np.sum(av_slice[1:])
                
        branching_ratio = (sum_A_t_plus_1 / sum_A_t) if sum_A_t > 0 else 0.0
        return sizes, durations, len(sizes), branching_ratio, s_thresh
        
    sizes, durations, num_av, branching_ratio, sigma_used = detect_avalanches(sigma)
    
    # If adaptive is enabled and we have too few avalanches, try a lower threshold
    if adaptive and num_av < min_avalanches and sigma > 1.5:
        logger.info(f"Detected only {num_av} avalanches with sigma={sigma}. Trying adaptive fallback to sigma=1.5.")
        sizes, durations, num_av, branching_ratio, sigma_used = detect_avalanches(1.5)
        if num_av < min_avalanches:
            logger.info(f"Adaptive fallback still yielded low events ({num_av}). Trying sigma=1.0.")
            sizes, durations, num_av, branching_ratio, sigma_used = detect_avalanches(1.0)
            
    # Fit powerlaw MLE
    avalanche_exponent = np.nan
    ks_distance = np.nan
    xmin = np.nan
    log_likelihood_ratio = np.nan
    p_value = np.nan
    
    if num_av >= 20:
        try:
            # Use discrete fitting since sizes are integer-valued counts
            fit = powerlaw.Fit(sizes, discrete=True, verbose=False)
            avalanche_exponent = float(fit.alpha)
            ks_distance = float(fit.D)
            xmin = float(fit.xmin)
            
            # Compare power law vs exponential
            R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
            log_likelihood_ratio = float(R)
            p_value = float(p)
        except Exception as e:
            logger.warning(f"Powerlaw fitting failed: {e}")
            
    return {
        'avalanche_exponent': avalanche_exponent,
        'branching_ratio': float(branching_ratio),
        'num_avalanches': int(num_av),
        'threshold_sigma_used': float(sigma_used),
        'ks_distance': ks_distance,
        'xmin': xmin,
        'log_likelihood_ratio': log_likelihood_ratio,
        'p_value': p_value
    }
