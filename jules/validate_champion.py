"""
Validate a champion on HELD-OUT arena seeds it was never selected on.

Fitness during evolution uses a fixed training seed set (1000-1014). If a
champion scores similarly on fresh held-out seeds (5000-5014), the evolved gain
is a real forager/survivor improvement, not overfitting to those 15 arenas.

Usage:  python3 validate_champion.py [champions_json] [rank]
"""
import sys
import json
import numpy as np

from organoid_simulator import creature_breedstock as bs


def main():
    champs_path = sys.argv[1] if len(sys.argv) > 1 else 'breedstock_champions.json'
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    with open(champs_path) as f:
        store = json.load(f)
    champ = store['champions'][rank]
    _, ctx = bs.build_scaffold()

    genome = np.asarray(champ['genome'], dtype=float)
    base = ctx['base_genome']

    train = list(bs.DEFAULT_EVAL_SEEDS)
    held = list(bs.DEFAULT_HELDOUT_SEEDS)

    print(f"Champion #{rank+1} (gen {champ['generation']})  vs unevolved base genome")
    print(f"{'':14}{'train (seen)':>16}{'held-out (fresh)':>18}")
    for name, g in [('base', base), ('champion', genome)]:
        tr = bs.evaluate_genome(ctx, g, seeds=train)
        hv = bs.evaluate_genome(ctx, g, seeds=held)
        print(f"  {name:11}  fit {tr['fitness']:6.3f} food {tr['mean_food']:.2f}"
              f"   fit {hv['fitness']:6.3f} food {hv['mean_food']:.2f}"
              f" surv {hv['mean_survival']:.0f}")


if __name__ == '__main__':
    main()
