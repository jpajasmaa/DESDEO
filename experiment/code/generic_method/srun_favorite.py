"""Cleaned Up Test Runner for the Favorite Method"""

import numpy as np
from desdeo.gdm.voting_rules import majority_rule
from desdeo.tools import PyomoIpoptSolver, ProximalSolver
from desdeo.tools.scalarization import add_asf_diff, add_asf_nondiff
from desdeo.problem.testproblems.dtlz2_problem import dtlz2
from desdeo.problem.testproblems import dmitry_forest_problem_disc

from desdeo.gdm.favorite_method import (
    IPR_Options, GPRMOptions, ZoomOptions, FavOptions, cluster_points,
    favorite_method, generate_next_iteration_mps
)
from visualizations import visualize_3d_clusters


def run_dtlz2():
    # --- 1. SETUP PROBLEM ---
    dtlz2_problem = dtlz2(8, 3)
    reference_points = {
        "DM1": {"f_1": 0.0, "f_2": 0.9, "f_3": 0.5},
        "DM2": {"f_1": 0.5, "f_2": 0.0, "f_3": 0.9},
        "DM3": {"f_1": 0.9, "f_2": 0.5, "f_3": 0.0},
    }
    # Generate random reference points and find MPS
    n_of_dms = 3
    reference_points = {}
    for i in range(n_of_dms):
        reference_points[f"DM{i+1}"] = {"f_1": np.random.random(), "f_2": np.random.random(), "f_3": np.random.random()}

    most_preferred_solutions = {}
    for dm in reference_points.keys():
        p, target = add_asf_diff(dtlz2_problem, symbol="asf", reference_point=reference_points[dm])
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        most_preferred_solutions[f"{dm}"] = res.optimal_objectives

    # --- 2. CONFIGURE INITIAL OPTIONS ---
    ipr_options = IPR_Options(
        most_preferred_solutions=most_preferred_solutions,
        num_initial_reference_points=10000,
        version="convex_hull",
        # version="box",
    )
    fav_options = FavOptions(
        GPRMoptions=GPRMOptions(method_options=ipr_options),
        candidate_generation_options="mm",
        zoom_options=ZoomOptions(num_steps_remaining=4),
        original_most_preferred_solutions=most_preferred_solutions,
        votes=None,
        total_n_of_candidates=5
    )

    # --- 3. THE LOOP ---
    results_history = []
    current_options = fav_options
    fractions = [0.8, 0.6, 0.4, 0.2]  # Shrink the hull each iteration

    # Run 4 iterations
    for iter_idx in range(4):
        print(f"\n{'='*40}\n--- RUNNING ITERATION {iter_idx + 1} ---\n{'='*40}")

        # 1. Run Core Method
        fav_results = favorite_method(
            problem=dtlz2_problem,
            options=current_options,
            results_list=results_history
        )
        results_history.append(fav_results)

        # 2. Simulate DM Voting (In a notebook, you could pause here to ask for input)
        votes = {"DM1": 2, "DM2": 2, "DM3": 2}

        pts_mat, cents_mat, labels = cluster_points(fav_results)
        winning_idx = majority_rule(votes)

        # 3. Generate Next Iteration MPS and get visualization data!
        next_mps = generate_next_iteration_mps(
            fav_results=fav_results,
            cluster_labels=labels,
            winning_idx=winning_idx,
            fraction_to_keep=fractions[iter_idx]
        )

        # 4. Visualize the Current Iteration
        visualize_3d_clusters(
            options=current_options.GPRMoptions,
            points_arr=pts_mat,
            centers_arr=cents_mat,
            labels=labels,
            n_predetermined=len(fav_results.fair_solutions),
            iter_n=iter_idx + 1
        )

        # 5. Prepare Options for Next Iteration
        current_options = current_options.model_copy(deep=True)
        current_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
        current_options.GPRMoptions.method_options.version = "convex_hull"
        current_options.zoom_options.num_steps_remaining = max(1, 4 - iter_idx - 1)
        current_options.votes = votes


def run_dmitry_forest_problem():
    # dmitry forest problem
    # --- 1. SETUP PROBLEM ---
    problem = dmitry_forest_problem_disc()
    obj_names = [obj.name for obj in problem.objectives]
    obj_symbols = [obj.symbol for obj in problem.objectives]

    # Fetch the problem's bounds for realistic random generation
    ideal = problem.get_ideal_point()
    nadir = problem.get_nadir_point()
    print(ideal, nadir)

    reference_points = {
        "DM1": {"f_1": 0.0, "f_2": 0.9, "f_3": 0.5},
        "DM2": {"f_1": 0.5, "f_2": 0.0, "f_3": 0.9},
        "DM3": {"f_1": 0.9, "f_2": 0.5, "f_3": 0.0},
        "DM4": {"f_1": 0.9, "f_2": 0.5, "f_3": 0.0},
    }
    # Generate random reference points and find MPS
    n_of_dms = 3
    most_preferred_solutions = {}
    for i in range(n_of_dms):
        dm_name = f"DM{i+1}"
        random_target = {
            name: np.random.uniform(ideal[name], nadir[name])
            for name in obj_symbols
        }
        p, target = add_asf_nondiff(problem, symbol="asf", reference_point=random_target)
        solver = ProximalSolver(p)
        res = solver.solve(target)
        most_preferred_solutions[dm_name] = res.optimal_objectives

    # --- 2. CONFIGURE INITIAL OPTIONS ---
    ipr_options = IPR_Options(
        most_preferred_solutions=most_preferred_solutions,
        num_initial_reference_points=10000,
        # version="box",
        version="convex_hull",
    )
    fav_options = FavOptions(
        GPRMoptions=GPRMOptions(method_options=ipr_options),
        candidate_generation_options="mm",
        zoom_options=ZoomOptions(num_steps_remaining=4),
        original_most_preferred_solutions=most_preferred_solutions,
        votes=None,
        total_n_of_candidates=5
    )

    print(most_preferred_solutions)
    # --- 3. THE LOOP ---
    results_history = []
    current_options = fav_options
    fractions = [0.8, 0.6, 0.4, 0.2]  # Shrink the hull each iteration

    # Run 4 iterations
    for iter_idx in range(4):
        print(f"\n{'='*40}\n--- RUNNING ITERATION {iter_idx + 1} ---\n{'='*40}")

        # 1. Run Core Method
        fav_results = favorite_method(
            problem=problem,
            options=current_options,
            results_list=results_history
        )
        results_history.append(fav_results)

        # 2. Simulate DM Voting (In a notebook, you could pause here to ask for input)
        votes = {"DM1": 2, "DM2": 2, "DM3": 2}

        pts_mat, cents_mat, labels = cluster_points(fav_results)
        winning_idx = majority_rule(votes)

        # 3. Generate Next Iteration MPS and get visualization data!
        next_mps = generate_next_iteration_mps(
            fav_results=fav_results,
            cluster_labels=labels,
            winning_idx=winning_idx,
            fraction_to_keep=fractions[iter_idx]
        )

        # 4. Visualize the Current Iteration
        visualize_3d_clusters(
            options=current_options.GPRMoptions,
            points_arr=pts_mat,
            centers_arr=cents_mat,
            labels=labels,
            n_predetermined=len(fav_results.fair_solutions),
            iter_n=iter_idx + 1
        )

        # 5. Prepare Options for Next Iteration
        current_options = current_options.model_copy(deep=True)
        current_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
        current_options.GPRMoptions.method_options.version = "convex_hull"
        current_options.zoom_options.num_steps_remaining = max(1, 4 - iter_idx - 1)
        current_options.votes = votes


if __name__ == "__main__":

    # run_dtlz2()
    run_dmitry_forest_problem()

    print("Favorite finished")
