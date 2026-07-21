"""
Batch runner executing the seven biological inheritance experiments (A-G) and
logging results/metrics to `results_evolution.json`.
"""
import json
import numpy as np
from organoid_simulator.creature_embodied import build_creature_network, run_creature_simulation
from organoid_simulator.snapshot_utils import capture_snapshot
from organoid_simulator.evolution_experiments import (
    experiment_a_structural_combination,
    experiment_b_developmental_inheritance,
    experiment_c_co_culture_fusion,
    experiment_d_neural_transplantation,
    experiment_e_trait_crossover,
    experiment_f_experience_transfer,
    experiment_g_multistage_pipeline
)
from organoid_simulator.creature_metrics import (
    calculate_network_metrics,
    calculate_behavioral_metrics,
    firing_stats,
    metabolic_stats
)

def run_experiment_evaluation(net, T_steps=2000, label='organism', seed=42):
    """Runs a closed-loop simulation on a network, returns behavior/activity and its snapshot."""
    res = run_creature_simulation(net, T_steps, dale=True, stdp_on=True, seed=seed)

    # Calculate metrics
    neural_m = calculate_network_metrics(net)
    behavior_m = calculate_behavioral_metrics(res['behavior'], T_steps)
    f_stats = firing_stats(res['spikes'])
    m_stats = metabolic_stats(res)

    # Combine metrics
    combined_metrics = {**neural_m, **behavior_m, **f_stats, **m_stats}

    # Capture snapshot
    snapshot = capture_snapshot(net, label=label)

    return combined_metrics, snapshot, res

def main():
    print("Initializing Parents for Evolution Experiments...")
    # Build two distinct, well-foraging parent brains (using scaffold parameters N=200, K=4)
    p1_net = build_creature_network(200, 4, fi=0.2, seed=101)
    p2_net = build_creature_network(200, 4, fi=0.2, seed=202)

    T_steps = 3000

    print("Evaluating Parent 1...")
    p1_metrics, p1_snapshot, _ = run_experiment_evaluation(p1_net, T_steps, label='Parent-1', seed=12)
    print(f"Parent 1: Food Eaten = {p1_metrics['food_eaten']}, Modularity = {p1_metrics['spatial_modularity']:.4f}")

    print("Evaluating Parent 2...")
    p2_metrics, p2_snapshot, _ = run_experiment_evaluation(p2_net, T_steps, label='Parent-2', seed=34)
    print(f"Parent 2: Food Eaten = {p2_metrics['food_eaten']}, Modularity = {p2_metrics['spatial_modularity']:.4f}")

    parents_data = [
        {"id": "p1", "label": "Parent 1 (Seed 101)", "metrics": p1_metrics, "snapshot": p1_snapshot},
        {"id": "p2", "label": "Parent 2 (Seed 202)", "metrics": p2_metrics, "snapshot": p2_snapshot}
    ]

    experiments_data = []
    lineage_nodes = [
        {"id": "p1", "label": "Parent 1", "parents": []},
        {"id": "p2", "label": "Parent 2", "parents": []}
    ]

    # Define Experiments to Run
    experiments_config = [
        {
            "name": "Experiment A: Structural Brain Combination",
            "code": "A",
            "func": lambda: experiment_a_structural_combination(p1_net, p2_net, seed=42),
            "parent_ids": ["p1", "p2"]
        },
        {
            "name": "Experiment B: Developmental Inheritance",
            "code": "B",
            "func": lambda: experiment_b_developmental_inheritance(p1_net, p2_net, seed=42),
            "parent_ids": ["p1", "p2"]
        },
        {
            "name": "Experiment C: Neural Co-Culture / Fusion",
            "code": "C",
            "func": lambda: experiment_c_co_culture_fusion(p1_net, p2_net, seed=42),
            "parent_ids": ["p1", "p2"]
        },
        {
            "name": "Experiment D: Neural Transplantation",
            "code": "D",
            "func": lambda: experiment_d_neural_transplantation(p1_net, p2_net, transplant_frac=0.15, seed=42),
            "parent_ids": ["p1", "p2"]
        },
        {
            "name": "Experiment E: Developmental Trait Crossover",
            "code": "E",
            "func": lambda: experiment_e_trait_crossover(p1_net, p2_net, seed=42),
            "parent_ids": ["p1", "p2"]
        },
        {
            "name": "Experiment F: Experience Transfer",
            "code": "F",
            "func": lambda: experiment_f_experience_transfer(p1_net, seed=42),
            "parent_ids": ["p1"]
        },
        {
            "name": "Experiment G: Combined Multi-Stage System",
            "code": "G",
            "func": lambda: experiment_g_multistage_pipeline(p1_net, p2_net, seed=42),
            "parent_ids": ["p1", "p2"]
        }
    ]

    # Run the batch
    for exp in experiments_config:
        print(f"\nRunning {exp['name']}...")
        child_net = exp['func']()

        # Evaluate offspring
        label = f"Offspring-{exp['code']}"
        child_metrics, child_snapshot, _ = run_experiment_evaluation(child_net, T_steps, label=label, seed=56)
        print(f"{label}: Food Eaten = {child_metrics['food_eaten']}, Modularity = {child_metrics['spatial_modularity']:.4f}, Survival Steps = {child_metrics['survival_steps']}")

        experiments_data.append({
            "experiment": exp['name'],
            "code": exp['code'],
            "parent_ids": exp['parent_ids'],
            "metrics": child_metrics,
            "snapshot": child_snapshot
        })

        lineage_nodes.append({
            "id": label.lower(),
            "label": label,
            "parents": exp['parent_ids'],
            "method": exp['code']
        })

    output = {
        "metadata": {
            "T_steps": T_steps,
            "scaffold_N": 200,
            "scaffold_K": 4
        },
        "parents": parents_data,
        "experiments": experiments_data,
        "lineage": {
            "nodes": lineage_nodes
        }
    }

    output_filepath = "results_evolution.json"
    with open(output_filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nEvolution Experiments run completed successfully! Logs saved to {output_filepath}")

if __name__ == "__main__":
    main()
