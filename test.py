import json
from policy import make_policy
from simulator import run_one_hop, run_two_hop, run_variant_scenario

with open("variant_params.json") as f:
    params = json.load(f)

sim_time_ms = params["grading"]["sim_time_ms"]
main_seed = params["grading"]["seed"]

# Reference scenario tests
print("<--- ONE-HOP -->")
print(run_one_hop(params, sim_time_ms, main_seed))

print("\n<--- TWO-HOP -->")

test_seeds = [main_seed, 1005, 1006, 1007, 1008]

for seed in test_seeds:
    result = run_two_hop(params, sim_time_ms, seed)

    print(
        f"Seed {seed}: "
        f"Rate = {result['rate']:.3f} pairs/s, "
        f"Fidelity = {result['mean_fidelity']:.4f}"
    )

print("\n<--- VARIANT -->")

seeds = [
    params["grading"]["seed"],
    *params["grading"]["held_out_seeds"]
]

for seed in seeds:
    policy = make_policy(params)

    result = run_variant_scenario(
        params,
        sim_time_ms=params["grading"]["sim_time_ms"],
        seed=seed,
        policy=policy
    )

    print(f"Seed {seed}: {result}")