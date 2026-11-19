"""
experiments.py — QIS-NET Project 2 sensitivity experiment (STUDENT FILE).

Implement run_sensitivity: one-at-a-time (OAT) sensitivity of the variant
scenario's delivered RATE to memory T2 over params['experiment']['t2_grid_ms'],
using >= params['experiment']['seeds_per_point'] seeds per grid point.

Output contract (autograded): write experiment_results.json containing
{
  "parameter": "t2_ms",
  "grid": [...],
  "seeds_per_point": int,
  "points": [
     {"t2_ms": x, "rate_mean": m, "rate_ci95": h, "n_seeds": k}, ...
  ]
}
CI: 95% t-interval half-width over the per-seed rates.

AI-assistant use declaration: run_sensitivity was written with help from
Claude (Anthropic AI assistant) and reviewed by Group 4.
"""

import copy
import json
import math

import numpy as np

from policy import make_policy
from simulator import run_variant_scenario


def mean_ci95(values):
    """Return (mean, 95% CI half-width) using the t-distribution."""
    n = len(values)
    m = float(np.mean(values))
    if n < 2:
        return m, 0.0
    s = float(np.std(values, ddof=1))
    # t_{0.975, df} for small df (table lookup, no scipy needed)
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
               6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    t = t_table.get(n - 1, 1.96)
    return m, t * s / math.sqrt(n)


def run_sensitivity(params, seeds, sim_time_ms=None,
                    out_path="experiment_results.json") -> dict:
    """One-at-a-time (OAT) sweep of delivered rate against memory T2.

    For each T2 value in the grid we change ONLY physics.t2_ms (everything
    else stays as in the variant), run the variant scenario once per seed,
    and summarise the per-seed rates as mean +/- 95% t-interval half-width.
    """
    exp = params["experiment"]
    grid = list(exp["t2_grid_ms"])
    seeds_per_point = int(exp["seeds_per_point"])
    if sim_time_ms is None:
        sim_time_ms = params["grading"]["sim_time_ms"]

    use_seeds = list(seeds)[:seeds_per_point]
    if len(use_seeds) < seeds_per_point:
        raise ValueError(f"need at least {seeds_per_point} seeds, "
                         f"got {len(use_seeds)}")

    points = []
    for t2 in grid:
        # copy so one grid point can never change another (or the caller's params)
        params_copy = copy.deepcopy(params)
        params_copy["physics"]["t2_ms"] = t2

        rates = []
        for seed in use_seeds:
            policy = make_policy(params_copy)   # fresh policy per run
            result = run_variant_scenario(params_copy, sim_time_ms, seed, policy)
            rates.append(float(result["rate"]))

        mean, half_width = mean_ci95(rates)
        points.append({
            "t2_ms": t2,
            "rate_mean": mean,
            "rate_ci95": half_width,
            "n_seeds": len(rates),
        })

    results = {
        "parameter": "t2_ms",
        "grid": grid,
        "seeds_per_point": seeds_per_point,
        "points": points,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import sys
    params = json.load(open(sys.argv[1] if len(sys.argv) > 1
                            else "variant_params.json"))
    exp = params["experiment"]
    seeds = exp["seeds"]
    out = run_sensitivity(params, seeds)
    print(json.dumps(out, indent=2))
