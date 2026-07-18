import matplotlib.pyplot as plt
import numpy as np
import os
import json
import logging

logger = logging.getLogger(__name__)

def generate_all_plots(results, output_dir='plots'):
    """
    Generate plots from simulation results and save to output_dir.
    
    Plots produced:
    1. MC vs N (with K as series)
    2. Criticality vs N: Exponent & Branching Ratio vs N (with K as series)
    3. Representative per-delay MC decay curve r^2(k) for a large-N configuration
    4. MC vs Spectral Radius (for small/mid N configurations where spectral radius was swept)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out failed or skipped runs
    valid_runs = [r for r in results if r['status'] == 'success']
    if not valid_runs:
        logger.warning("No valid runs to plot!")
        return
        
    # Group results by (N, K, rho, topology) to aggregate across seeds (mean ± std)
    grouped = {}
    for r in valid_runs:
        key = (r['N'], r['K'], r['rho'], r['topology'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
        
    # Standardize data structure for plotting
    aggregated = []
    for key, runs in grouped.items():
        N, K, rho, topology = key
        mcs = [run['total_mc'] for run in runs]
        exponents = [run['avalanche_exponent'] for run in runs if not np.isnan(run['avalanche_exponent'])]
        branchings = [run['branching_ratio'] for run in runs]
        runtimes = [run['elapsed_time_seconds'] for run in runs]
        mems = [run['state_memory_mb'] for run in runs]
        
        aggregated.append({
            'N': N,
            'K': K,
            'rho': rho,
            'topology': topology,
            'mc_mean': np.mean(mcs),
            'mc_std': np.std(mcs) if len(mcs) > 1 else 0.0,
            'exp_mean': np.mean(exponents) if exponents else np.nan,
            'exp_std': np.std(exponents) if len(exponents) > 1 else 0.0,
            'br_mean': np.mean(branchings),
            'br_std': np.std(branchings) if len(branchings) > 1 else 0.0,
            'runtime_mean': np.mean(runtimes),
            'mem_mean': np.mean(mems)
        })
        
    # ------------------ Plot 1: MC vs N (with K as series, fixed rho = 0.95 or close) ------------------
    plt.figure(figsize=(8, 5))
    # Filter for standard rho (usually 0.95 or closest) and standard random topology
    target_rho = 0.95
    rhos_avail = {a['rho'] for a in aggregated}
    closest_rho = min(rhos_avail, key=lambda x: abs(x - target_rho)) if rhos_avail else 0.95
    
    plot_data_1 = [a for a in aggregated if abs(a['rho'] - closest_rho) < 1e-4 and a['topology'] == 'random']
    ks = sorted(list({a['K'] for a in plot_data_1}))
    
    for k in ks:
        k_data = sorted([a for a in plot_data_1 if a['K'] == k], key=lambda x: x['N'])
        ns = [x['N'] for x in k_data]
        mc_means = [x['mc_mean'] for x in k_data]
        mc_stds = [x['mc_std'] for x in k_data]
        
        plt.errorbar(ns, mc_means, yerr=mc_stds, marker='o', capsize=5, label=f'K = {k}')
        
    plt.xscale('log')
    plt.xlabel('Network Size (N, log scale)', fontsize=11)
    plt.ylabel('Memory Capacity (MC)', fontsize=11)
    plt.title(f'Memory Capacity vs. Network Size N (rho = {closest_rho})', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'mc_vs_n.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    
    # ------------------ Plot 2: Criticality vs N (Exponent & Branching Ratio) ------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Exponent vs N
    for k in ks:
        k_data = sorted([a for a in plot_data_1 if a['K'] == k], key=lambda x: x['N'])
        ns = [x['N'] for x in k_data if not np.isnan(x['exp_mean'])]
        exp_means = [x['exp_mean'] for x in k_data if not np.isnan(x['exp_mean'])]
        exp_stds = [x['exp_std'] for x in k_data if not np.isnan(x['exp_mean'])]
        
        if ns:
            ax1.errorbar(ns, exp_means, yerr=exp_stds, marker='s', capsize=5, label=f'K = {k}')
            
    ax1.set_xscale('log')
    ax1.set_xlabel('Network Size (N, log scale)', fontsize=11)
    ax1.set_ylabel('Avalanche Size Exponent (tau)', fontsize=11)
    ax1.set_title('Avalanche Exponent vs. Size N', fontsize=12, fontweight='bold')
    ax1.axhline(1.5, color='red', linestyle='--', alpha=0.7, label='Theoretical (1.5)')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=10)
    
    # Branching Ratio vs N
    for k in ks:
        k_data = sorted([a for a in plot_data_1 if a['K'] == k], key=lambda x: x['N'])
        ns = [x['N'] for x in k_data]
        br_means = [x['br_mean'] for x in k_data]
        br_stds = [x['br_std'] for x in k_data]
        
        ax2.errorbar(ns, br_means, yerr=br_stds, marker='^', capsize=5, label=f'K = {k}')
        
    ax2.set_xscale('log')
    ax2.set_xlabel('Network Size (N, log scale)', fontsize=11)
    ax2.set_ylabel('Branching Ratio (sigma)', fontsize=11)
    ax2.set_title('Branching Ratio vs. Size N', fontsize=12, fontweight='bold')
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Criticality (1.0)')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'criticality_vs_n.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    
    # ------------------ Plot 3: Representative per-delay MC decay curve r^2(k) ------------------
    plt.figure(figsize=(8, 5))
    # Pick the largest size N completed successfully
    completed_ns = sorted(list({run['N'] for run in valid_runs}))
    if completed_ns:
        largest_n = completed_ns[-1]
        large_n_run = next((r for r in valid_runs if r['N'] == largest_n and r['rho'] == closest_rho), None)
        if large_n_run and large_n_run['mc_curves']:
            curves = large_n_run['mc_curves']
            # Keys might be strings or ints depending on serialization
            delays = sorted([int(k) for k in curves.keys()])
            vals = [curves[str(k)] if str(k) in curves else curves[k] for k in delays]
            
            plt.plot(delays, vals, marker='o', linewidth=2, color='darkblue', label=f'Actual (N={largest_n}, K={large_n_run["K"]})')
            plt.xlabel('Delay (k)', fontsize=11)
            plt.ylabel('Squared Correlation (r^2)', fontsize=11)
            plt.title(f'Memory Capacity Decay Curve (N={largest_n}, K={large_n_run["K"]}, rho={closest_rho})', fontsize=12, fontweight='bold')
            plt.grid(True, ls="--", alpha=0.5)
            plt.ylim(-0.05, 1.05)
            
            # Theoretical delay line (shift register) control plot comparison
            plt.plot(delays, [1.0 if d <= 10 else 0.0 for d in delays], color='gray', linestyle=':', alpha=0.8, label='Linear Shift Register Control (Theoretical)')
            plt.legend(fontsize=10)
            
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, 'mc_decay_curve.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    
    # ------------------ Plot 4: MC vs Spectral Radius (for swept rhos) ------------------
    plt.figure(figsize=(8, 5))
    # Gather rhos and small sizes
    plot_data_4 = [a for a in aggregated if a['topology'] == 'random' and a['N'] <= 1000]
    unique_ns = sorted(list({a['N'] for a in plot_data_4}))
    
    for n in unique_ns:
        n_data = sorted([a for a in plot_data_4 if a['N'] == n], key=lambda x: x['rho'])
        rhos = [x['rho'] for x in n_data]
        mc_means = [x['mc_mean'] for x in n_data]
        mc_stds = [x['mc_std'] for x in n_data]
        
        plt.errorbar(rhos, mc_means, yerr=mc_stds, marker='p', capsize=5, label=f'N = {n}')
        
    plt.xlabel('Spectral Radius (rho)', fontsize=11)
    plt.ylabel('Memory Capacity (MC)', fontsize=11)
    plt.title('Memory Capacity vs. Recurrent Spectral Radius (rho)', fontsize=12, fontweight='bold')
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot4_path = os.path.join(output_dir, 'mc_vs_rho.png')
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    
    logger.info("Successfully generated and saved all plots in the 'plots/' directory!")
