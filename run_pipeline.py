import os
import json
import logging
import argparse
from organoid_simulator.sweep import run_parameter_sweep
from organoid_simulator.plot import generate_all_plots
from organoid_simulator.pdf_report import compile_pdf_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Organoid Neural Tissue Simulation Sweep Runner")
    parser.add_argument("--checkpoint", type=str, default="sweep_results.json", help="Checkpoint file path")
    parser.add_argument("--budget", type=float, default=25.0, help="Runtime budget per config in minutes")
    parser.add_argument("--quick", action="store_true", help="Run a quick/reduced sweep for testing")
    args = parser.parse_args()

    # Define the core sweep parameter grid
    if args.quick:
        # Reduced test grid for testing/validation
        ns = [100, 300, 1000]
        ks = [50]
        rhos = [0.95]
        seeds = [1, 2]
    else:
        # Standard full log-step sweep
        ns = [100, 300, 1000, 3000, 10000, 30000, 100000]
        ks = [50, 100]
        rhos = [0.95]
        seeds = [1, 2, 3] # 3 seeds per config
        
    configs = []
    
    # 1. Main Size & Connectivity Sweep (with standard fixed rho=0.95)
    for n in ns:
        for k in ks:
            # Synapse count K should be less than neuron count N
            if k >= n:
                continue
            for rho in rhos:
                for seed in seeds:
                    configs.append({
                        'N': n,
                        'K': k,
                        'rho': rho,
                        'seed': seed,
                        'topology': 'random'
                    })
                    
    # 2. Spectral Radius Sensitivity Sweep (light sweep for small/mid N <= 1000)
    # We sweep rho ∈ {0.8, 1.05} to compare with our primary rho=0.95
    if not args.quick:
        sensitivity_ns = [100, 300, 1000]
        sensitivity_rhos = [0.8, 1.05]
        for n in sensitivity_ns:
            for k in ks:
                if k >= n:
                    continue
                for rho in sensitivity_rhos:
                    for seed in seeds[:2]: # 2 seeds are sufficient for the sensitivity check
                        configs.append({
                            'N': n,
                            'K': k,
                            'rho': rho,
                            'seed': seed,
                            'topology': 'random'
                        })

    # 3. Locality Topology Comparison variant (1D Ring decay)
    # Run a subset to see how local distance-decay topology changes the MC knee
    if not args.quick:
        topology_ns = [100, 300, 1000, 3000]
        for n in topology_ns:
            for k in ks:
                if k >= n:
                    continue
                for seed in seeds[:2]:
                    configs.append({
                        'N': n,
                        'K': k,
                        'rho': 0.95,
                        'seed': seed,
                        'topology': 'ring',
                        'lambda_ring': 0.1
                    })

    logger.info(f"Total configurations scheduled: {len(configs)}")
    
    # Run the sweep
    results = run_parameter_sweep(
        configs,
        checkpoint_file=args.checkpoint,
        time_budget_minutes=args.budget
    )
    
    # Filter for successfully finished configs
    valid_results = [r for r in results if r['status'] == 'success']
    logger.info(f"Sweep complete. Successfully ran {len(valid_results)} of {len(configs)} configs.")
    
    # Generate Plots
    generate_all_plots(results)
    
    # Generate the comprehensive ReportLab PDF Report
    compile_pdf_report(results, output_filename='organoid_findings_report.pdf')
    
    logger.info("Pipeline executed successfully!")

if __name__ == "__main__":
    main()
