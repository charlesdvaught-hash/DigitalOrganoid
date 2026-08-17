"""
Evaluate Flocke-inspired mechanisms on CPU simulation engine.
Measures food eaten, mean speed, firing rates, branching ratio (criticality sigma),
and steering correlation r across seeds.
"""
import numpy as np
from organoid_simulator.creature_embodied import (
    build_creature_network, run_creature_simulation
)
from organoid_simulator.creature_metrics import firing_stats, avalanche_stats


def evaluate_condition(name, seeds=[42, 7, 123, 99], T=3000,
                       cpg=False, cerebellar=False,
                       multichannel=False, bilateral=False):
    food_list = []
    speed_list = []
    sigma_list = []
    rate_list = []

    for seed in seeds:
        net = build_creature_network(
            N=600, K=10, fi=0.2, seed=seed, bilateral_symmetric=bilateral
        )
        res = run_creature_simulation(
            net, T=T, dale=True, stdp_on=True, seed=seed,
            cpg_enabled=cpg,
            cerebellar_enabled=cerebellar,
            multichannel_neuromod=multichannel
        )
        b = res['behavior']
        f_stat = firing_stats(res['spikes'])
        a_stat = avalanche_stats(res['spikes'])

        food_list.append(b['food_eaten'])
        speed_list.append(b['mean_speed'])
        rate_list.append(f_stat['mean_pop_rate'])
        sigma_list.append(a_stat.get('mean_avalanche_size', 0.0))

    mean_food = np.mean(food_list)
    std_food = np.std(food_list)
    mean_speed = np.mean(speed_list)
    mean_sigma = np.mean(sigma_list)
    mean_rate = np.mean(rate_list)

    print(f"[{name}]")
    print(f"  Food Eaten: {mean_food:.2f} +/- {std_food:.2f} (per seed: {food_list})")
    print(f"  Mean Speed: {mean_speed:.5f}")
    print(f"  Population Firing Rate: {mean_rate:.4f}")
    print(f"  Mean Avalanche Size: {mean_sigma:.3f}")
    print("-" * 50)
    return {
        'name': name,
        'mean_food': mean_food,
        'std_food': std_food,
        'mean_speed': mean_speed,
        'mean_sigma': mean_sigma,
        'mean_rate': mean_rate,
    }


def main():
    print("=== Flocke CPU Comparative Benchmark ===")
    seeds = [42, 7, 123, 99]
    T = 2000

    evaluate_condition("Baseline", seeds, T=T)
    evaluate_condition("Bilateral Symmetry", seeds, T=T, bilateral=True)
    evaluate_condition("Innate CPG Prior", seeds, T=T, cpg=True)
    evaluate_condition("Cerebellar Forward Model", seeds, T=T, cerebellar=True)
    evaluate_condition("Multi-Channel Neuromod", seeds, T=T, multichannel=True)
    evaluate_condition("ALL COMBINED", seeds, T=T, cpg=True, cerebellar=True,
                       multichannel=True, bilateral=True)


if __name__ == "__main__":
    main()
