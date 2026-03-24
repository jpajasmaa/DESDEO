"""Test Runner for the Favorite Method"""

import numpy as np
from desdeo.tools import PyomoIpoptSolver
from desdeo.tools.scalarization import add_asf_diff
from desdeo.problem.testproblems.dtlz2_problem import dtlz2

# Import Logic
from desdeo.gdm.favorite_method import (
    IPR_Options, GPRMOptions, ZoomOptions, FavOptions,
    favorite_method, find_candidates, hausdorff_candidates, cluster_points,
    expand_and_generate_candidates
)

# Import Visualization
from visualizations import visualize_selection_2d, visualize_3d_clusters, visualize_expansion

if __name__ == "__main__":

    # --- 1. SETUP PROBLEM & DMs ---
    dtlz2_problem = dtlz2(8, 3)
    n_of_dms = 3

    # Generate random reference points and find MPS
    reference_points = {}
    for i in range(n_of_dms):
        reference_points[f"DM{i+1}"] = {"f_1": np.random.random(), "f_2": np.random.random(), "f_3": np.random.random()}

    reference_points = {
        "DM1": {"f_1": 0.0, "f_2": 0.9, "f_3": 0.5},
        "DM2": {"f_1": 0.5, "f_2": 0.0, "f_3": 0.9},
        "DM3": {"f_1": 0.9, "f_2": 0.5, "f_3": 0.0},
    }

    print("Initial Reference Points:", reference_points)

    most_preferred_solutions = {}
    for dm in reference_points.keys():
        p, target = add_asf_diff(dtlz2_problem, symbol="asf", reference_point=reference_points[dm])
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        most_preferred_solutions[f"{dm}"] = res.optimal_objectives

    # --- 2. CONFIGURE OPTIONS ---
    ipr_options = IPR_Options(
        most_preferred_solutions=most_preferred_solutions,
        num_initial_reference_points=10000,
        version="convex_hull",
    )
    grpmoptions = GPRMOptions(method_options=ipr_options)
    zoomoptions = ZoomOptions(num_steps_remaining=4)

    fav_options = FavOptions(
        GPRMoptions=grpmoptions,
        candidate_generation_options="mm",
        zoom_options=zoomoptions,
        original_most_preferred_solutions=most_preferred_solutions,
        votes=None,
        total_n_of_candidates=5
    )

    total_n_of_candidates = 5

    # --- 3. RUN ITERATION 1 ---
    print("\n--- Running Iteration 1 ---")
    fav_results = favorite_method(
        problem=dtlz2_problem,
        options=fav_options,
        results_list=[]
    )
    print("Iter 1 Complete.")

    # --- 4. CLUSTERING & SELECTION ---
    # handled now in one function
    points_matrix, centers_matrix, cluster_labels = find_candidates(fav_results)

    # TODO: these are repeated solely for ease of visualization in development.
    # =========================================================================
    all_points = fav_results.GPRMResults.raw_results.evaluated_points
    fairs = fav_results.fair_solutions
    # n_of_candidates = total_n_of_candidates - len(fairs)
    # candidates = hausdorff_candidates(all_points, fairs, n_of_candidates)
    # Visualization 1
    # visualize_selection_2d(all_points, fairs, candidates, "2D Space: Average (Centers)")
    # points_matrix, centers_matrix, cluster_labels = cluster_points(all_points, candidates)
    # Visualization 2
    visualize_3d_clusters(fav_options.GPRMoptions, points_matrix, centers_matrix, cluster_labels, len(fairs), 1)
    # TODO: remove later
    # =========================================================================

    # --- 5. HULL EXPANSION ---

    # TODO: here we need to interact to get the votes from the DMs. for now, we determine it with the index.
    votes = {"DM1": 2, "DM2": 2, "DM3": 2, "DM4": 2}
    winning_idx = 2

    # TODO: these would also go somewhere else, some sort of helper function
    winning_points = points_matrix[cluster_labels == winning_idx]
    winning_center = centers_matrix[winning_idx]

    print(f"\nCluster {winning_idx} selected with {len(winning_points)} points.")
    print("--- Generating New Candidates via Convex Hull Expansion ---")

    fraction_to_keep = 0.8
    num_new_points = 1000

    new_candidates_k = expand_and_generate_candidates(
        winning_cluster_k=winning_points,
        all_points_k=points_matrix,
        fraction_keep=fraction_to_keep,
        num_new_points=num_new_points
    )
    print(f"Successfully generated {len(new_candidates_k)} new candidate points.")

    # Visualization 3
    # visualize_expansion(points_matrix, winning_points, new_candidates_k, winning_center, winning_idx, fraction_to_keep)

    # --- 6. RUN ITERATION 2 (With Extended Hull) ---
    print("\n--- Running Iteration 2 ---")

    # Transform numpy array to "Most Preferred Solutions" dict for IPR
    next_iter_mps = {}
    obj_names = [f"f_{i+1}" for i in range(new_candidates_k.shape[1])]
    for i, point in enumerate(new_candidates_k):
        point_dict = {name: val for name, val in zip(obj_names, point)}
        next_iter_mps[f"gen_{i}"] = point_dict

    # Clone and Update Options
    fav_options_2 = fav_options.model_copy(deep=True)
    fav_options_2.GPRMoptions.method_options.most_preferred_solutions = next_iter_mps
    fav_options_2.GPRMoptions.method_options.version = "convex_hull"  # Use the hull of the new points!
    fraction_to_keep = 0.8
    fav_options_2.zoom_options.num_steps_remaining = 3

    # Voting (DM1 wins previous round)
    fav_options_2.votes = votes

    fav_results_2 = favorite_method(
        problem=dtlz2_problem,
        options=fav_options_2,
        results_list=[fav_results]
    )

    print("Iter 2 Complete.")
    print("Fair Solutions found in Iter 2:", len(fav_results_2.fair_solutions))

    # TODO: here, repeat earlier steps of clustering, voting, expanding
    # --- 4. CLUSTERING & SELECTION ---
    # handled now in one function
    points_matrix, centers_matrix, cluster_labels = find_candidates(fav_results_2)

    # TODO: these are repeated solely for ease of visualization in development.
    # =========================================================================
    all_points = fav_results_2.GPRMResults.raw_results.evaluated_points
    fairs = fav_results_2.fair_solutions
    n_of_candidates = total_n_of_candidates - len(fairs)
    candidates = hausdorff_candidates(all_points, fairs, n_of_candidates)
    # Visualization 1
    # visualize_selection_2d(all_points, fairs, candidates, "2D Space: Average (Centers)")
    points_matrix, centers_matrix, cluster_labels = cluster_points(all_points, candidates)
    # Visualization 2
    visualize_3d_clusters(fav_options_2.GPRMoptions, points_matrix, centers_matrix, cluster_labels, len(fairs), 2)
    # TODO: here we need to interact to get the votes from the DMs. for now, we determine it with the index.
    votes = {"DM1": 0, "DM2": 0, "DM3": 0, "DM4": 0}
    winning_idx = 0

    # TODO: these would also go somewhere else, some sort of helper function
    winning_points = points_matrix[cluster_labels == winning_idx]
    winning_center = centers_matrix[winning_idx]

    print(f"\nCluster {winning_idx} selected with {len(winning_points)} points.")
    print("--- Generating New Candidates via Convex Hull Expansion ---")

    num_new_points = 1000

    new_candidates_k = expand_and_generate_candidates(
        winning_cluster_k=winning_points,
        all_points_k=points_matrix,
        fraction_keep=fraction_to_keep,
        num_new_points=num_new_points
    )
    print(f"Successfully generated {len(new_candidates_k)} new candidate points.")

    # Visualization 3
    # visualize_expansion(points_matrix, winning_points, new_candidates_k, winning_center, winning_idx, fraction_to_keep)

    # --- 6. RUN ITERATION 2 (With Extended Hull) ---
    print("\n--- Running Iteration X ---")

    # Transform numpy array to "Most Preferred Solutions" dict for IPR
    next_iter_mps = {}
    obj_names = [f"f_{i+1}" for i in range(new_candidates_k.shape[1])]
    for i, point in enumerate(new_candidates_k):
        point_dict = {name: val for name, val in zip(obj_names, point)}
        next_iter_mps[f"gen_{i}"] = point_dict

    print(next_iter_mps)

    # Clone and Update Options
    fav_options_3 = fav_options_2.model_copy(deep=True)
    fav_options_3.GPRMoptions.method_options.most_preferred_solutions = next_iter_mps
    fav_options_3.GPRMoptions.method_options.version = "convex_hull"  # Use the hull of the new points!
    fraction_to_keep = 0.4
    fav_options_3.zoom_options.num_steps_remaining = 2

    # Voting (DM1 wins previous round)
    fav_options_3.votes = votes

    fav_results_3 = favorite_method(
        problem=dtlz2_problem,
        options=fav_options_3,
        results_list=[fav_results, fav_results_2]
        # results_list=[fav_results_2]
    )

    print("Iter 3 Complete.")
    print("Fair Solutions found in Iter 3:", len(fav_results_3.fair_solutions))

    # --- 4. CLUSTERING & SELECTION ---
    # handled now in one function
    points_matrix, centers_matrix, cluster_labels = find_candidates(fav_results_3)

    # TODO: these are repeated solely for ease of visualization in development.
    # =========================================================================
    all_points = fav_results_3.GPRMResults.raw_results.evaluated_points
    fairs = fav_results_3.fair_solutions
    n_of_candidates = total_n_of_candidates - len(fairs)
    candidates = hausdorff_candidates(all_points, fairs, n_of_candidates)
    # Visualization 1
    # visualize_selection_2d(all_points, fairs, candidates, "2D Space: Average (Centers)")
    points_matrix, centers_matrix, cluster_labels = cluster_points(all_points, candidates)
    # Visualization 2
    visualize_3d_clusters(fav_options_3.GPRMoptions, points_matrix, centers_matrix, cluster_labels, len(fairs), 3)

    # TODO: here we need to interact to get the votes from the DMs. for now, we determine it with the index.
    votes = {"DM1": 0, "DM2": 0, "DM3": 0, "DM4": 0}
    winning_idx = 0

    # TODO: these would also go somewhere else, some sort of helper function
    winning_points = points_matrix[cluster_labels == winning_idx]
    winning_center = centers_matrix[winning_idx]

    print(f"\nCluster {winning_idx} selected with {len(winning_points)} points.")
    print("--- Generating New Candidates via Convex Hull Expansion ---")

    num_new_points = 1000

    new_candidates_k = expand_and_generate_candidates(
        winning_cluster_k=winning_points,
        all_points_k=points_matrix,
        fraction_keep=fraction_to_keep,
        num_new_points=num_new_points
    )
    print(f"Successfully generated {len(new_candidates_k)} new candidate points.")

    # Visualization 3
    # visualize_expansion(points_matrix, winning_points, new_candidates_k, winning_center, winning_idx, fraction_to_keep)

   # --- 6. RUN ITERATION 2 (With Extended Hull) ---
    print("\n--- Running Iteration X ---")

    # Transform numpy array to "Most Preferred Solutions" dict for IPR
    next_iter_mps = {}
    obj_names = [f"f_{i+1}" for i in range(new_candidates_k.shape[1])]
    for i, point in enumerate(new_candidates_k):
        point_dict = {name: val for name, val in zip(obj_names, point)}
        next_iter_mps[f"gen_{i}"] = point_dict

    print(next_iter_mps)

    # Clone and Update Options
    fav_options_4 = fav_options_3.model_copy(deep=True)
    fav_options_4.GPRMoptions.method_options.most_preferred_solutions = next_iter_mps
    fav_options_4.GPRMoptions.method_options.version = "convex_hull"  # Use the hull of the new points!
    fraction_to_keep = 0.1
    fav_options_4.zoom_options.num_steps_remaining = 1

    # Voting (DM1 wins previous round)
    fav_options_4.votes = votes

    fav_results_4 = favorite_method(
        problem=dtlz2_problem,
        options=fav_options_4,
        results_list=[fav_results, fav_results_2, fav_results_3]
        # results_list=[fav_results_3]
    )

    print("Iter 4 Complete.")
    print("Fair Solutions found in Iter 4:", len(fav_results_4.fair_solutions))

    # --- 4. CLUSTERING & SELECTION ---
    # handled now in one function
    points_matrix, centers_matrix, cluster_labels = find_candidates(fav_results_4)

    # TODO: these are repeated solely for ease of visualization in development.
    # =========================================================================
    all_points = fav_results_4.GPRMResults.raw_results.evaluated_points
    fairs = fav_results_4.fair_solutions
    n_of_candidates = total_n_of_candidates - len(fairs)
    candidates = hausdorff_candidates(all_points, fairs, n_of_candidates)
    # Visualization 1
    # visualize_selection_2d(all_points, fairs, candidates, "2D Space: Average (Centers)")
    points_matrix, centers_matrix, cluster_labels = cluster_points(all_points, candidates)
    # Visualization 2
    visualize_3d_clusters(fav_options_4.GPRMoptions, points_matrix, centers_matrix, cluster_labels, len(fairs), 4)

    # TODO: here we need to interact to get the votes from the DMs. for now, we determine it with the index.
    votes = {"DM1": 0, "DM2": 0, "DM3": 0, "DM4": 0}
    winning_idx = 0

    # TODO: these would also go somewhere else, some sort of helper function
    winning_points = points_matrix[cluster_labels == winning_idx]
    winning_center = centers_matrix[winning_idx]

    print(f"\nCluster {winning_idx} selected with {len(winning_points)} points.")
    print("--- Generating New Candidates via Convex Hull Expansion ---")

    num_new_points = 1000

    new_candidates_k = expand_and_generate_candidates(
        winning_cluster_k=winning_points,
        all_points_k=points_matrix,
        fraction_keep=fraction_to_keep,
        num_new_points=num_new_points
    )
    print(f"Successfully generated {len(new_candidates_k)} new candidate points.")

    # Visualization 3
    # visualize_expansion(points_matrix, winning_points, new_candidates_k, winning_center, winning_idx, fraction_to_keep)
