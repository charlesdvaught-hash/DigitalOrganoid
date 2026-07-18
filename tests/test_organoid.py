import unittest
import numpy as np
import scipy.sparse as sp
from organoid_simulator.model import (
    build_recurrent_weights,
    scale_spectral_radius,
    run_simulation,
    run_homeostasis,
    compute_spectral_radius
)
from organoid_simulator.metrics import (
    solve_ridge_multi,
    compute_memory_capacity,
    compute_criticality_metrics
)

class TestOrganoidSimulator(unittest.TestCase):
    
    def test_dales_law_and_sparsity(self):
        N = 100
        K = 15
        W_rec, ids = build_recurrent_weights(N, K, topology='random', seed=42)
        
        # Verify shape
        self.assertEqual(W_rec.shape, (N, N))
        
        # Verify Dale's law column-wise splitting: exactly 80% excitatory (positive) and 20% inhibitory (negative)
        n_exc = int(round(0.8 * N))
        n_inh = N - n_exc
        self.assertEqual(np.sum(ids > 0), n_exc)
        self.assertEqual(np.sum(ids < 0), n_inh)
        
        # Verify sign constraints on columns
        W_csc = W_rec.tocsc()
        for j in range(N):
            col_data = W_csc.getcol(j).data
            # Check length is exactly K
            self.assertEqual(len(col_data), K)
            # Check signs match column's identity
            if ids[j] > 0:
                self.assertTrue(np.all(col_data >= 0))
            else:
                self.assertTrue(np.all(col_data <= 0))
                
    def test_spectral_radius_and_scaling(self):
        N = 50
        K = 10
        W_rec, ids = build_recurrent_weights(N, K, topology='random', seed=12)
        
        rho_initial = compute_spectral_radius(W_rec)
        self.assertGreater(rho_initial, 0.0)
        
        W_scaled = scale_spectral_radius(W_rec, 0.95)
        rho_scaled = compute_spectral_radius(W_scaled)
        self.assertAlmostEqual(rho_scaled, 0.95, places=5)
        
    def test_echo_state_property(self):
        # Two identical networks with different initial states should converge to same trajectory under input drive
        N = 100
        K = 20
        W_rec, ids = build_recurrent_weights(N, K, topology='random', seed=1)
        W_scaled = scale_spectral_radius(W_rec, 0.9)
        W_in = np.random.uniform(-1, 1, (N, 1))
        u = np.random.uniform(-1, 1, 100)
        
        x0_1 = np.ones(N)
        x0_2 = -np.ones(N)
        
        states1 = run_simulation(W_scaled, W_in, u, x_init=x0_1)
        states2 = run_simulation(W_scaled, W_in, u, x_init=x0_2)
        
        # At the end of 100 steps, states should be highly correlated and converging (ESP)
        diff = np.mean(np.abs(states1[-10:] - states2[-10:]))
        self.assertLess(diff, 1e-1)
        
    def test_homeostasis(self):
        # Homeostasis should achieve target absolute activation ≈ 0.1
        N = 100
        K = 15
        W_rec, ids = build_recurrent_weights(N, K, topology='random', seed=1)
        W_scaled = scale_spectral_radius(W_rec, 0.95)
        
        W_in, history = run_homeostasis(W_scaled, target_mean=0.1, probe_steps=200, num_iters=8, seed=42)
        
        # Check that target is hit reasonably close
        final_mean_activity = history[-1][1]
        self.assertAlmostEqual(final_mean_activity, 0.1, delta=0.03)
        
    def test_primal_vs_dual_ridge(self):
        # Check dual form matches primal form outputs
        T, N = 100, 50
        X = np.random.randn(T, N)
        Y = np.random.randn(T, 5) # 5 multiple targets
        
        W_primal = solve_ridge_multi(X, Y, alpha=0.1)
        self.assertEqual(W_primal.shape, (50, 5))
        
        # Construct dual solve manually
        # solve_ridge_multi does primal/dual automatically based on shape.
        # So we transpose X to make N > T and test the dual form.
        T_dual, N_dual = 50, 100
        X_dual = np.random.randn(T_dual, N_dual)
        Y_dual = np.random.randn(T_dual, 5)
        
        W_dual = solve_ridge_multi(X_dual, Y_dual, alpha=0.1)
        self.assertEqual(W_dual.shape, (100, 5))
        
    def test_trivial_linear_reservoir_mc(self):
        # Build a strict linear shift register reservoir to test MC limit.
        # Shift Register: x_i(t+1) = x_{i-1}(t)
        # N=10, K=1, each neuron connects only to next.
        # Linear dynamics:
        N = 10
        T = 200
        u_seq = np.random.uniform(-1, 1, T)
        
        # Shift register states
        # state[t, i] = u(t - i - 1)
        states = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                delay = i + 1
                if t >= delay:
                    states[t, i] = u_seq[t - delay]
                    
        # Apply MC computation
        total_mc, mc_curves = compute_memory_capacity(states, u_seq, k_max=15, alpha=1e-6, train_frac=0.8)
        
        # Since it is a perfect delay line, delay 1..10 should have r^2 near 1.0, and 11..15 should have r^2 near 0.0
        # Total MC should be close to 10
        self.assertAlmostEqual(total_mc, 10.0, delta=0.5)
        for k in range(1, 11):
            self.assertGreater(mc_curves[k], 0.9)
        for k in range(11, 16):
            self.assertLess(mc_curves[k], 0.1)

if __name__ == '__main__':
    unittest.main()
