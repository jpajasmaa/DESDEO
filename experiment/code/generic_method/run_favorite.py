"""Test Runner for the Favorite Method"""

import numpy as np
from scipy.spatial.distance import cdist

from desdeo.gdm.voting_rules import majority_rule
from desdeo.tools import guess_best_solver
from desdeo.tools.scalarization import add_asf_diff
from desdeo.problem.testproblems import dtlz2

# Import Logic
from desdeo.gdm.favorite_method import (
    IPR_Options, GPRMOptions, ZoomOptions, FavOptions,
    favorite_method, cluster_points, generate_next_iteration_mps,
    tie_breaker_avgproj, recluster_for_tie_breaker, calculate_fraction_to_keep
)

# Import Visualization (Make sure this file is in your directory)
from visualizations import visualize_3d_clusters

if __name__ == "__main__":

    # --- 1. SETUP PROBLEM & DMs ---
    dtlz2_problem = dtlz2(8, 3)
    n_of_dms = 4

    # Generate random reference points and find MPS
    reference_points = {}
    for i in range(n_of_dms):
        reference_points[f"DM{i+1}"] = {"f_1": np.random.random(), "f_2": np.random.random(), "f_3": np.random.random()}

    print("Initial Reference Points:", reference_points)

    most_preferred_solutions = {}
    for dm in reference_points.keys():
        p, target = add_asf_diff(dtlz2_problem, symbol="asf", reference_point=reference_points[dm])
        solver = guess_best_solver(p)(p)
        res = solver.solve(target)
        most_preferred_solutions[f"{dm}"] = res.optimal_objectives

    # --- 2. CONFIGURE INITIAL OPTIONS ---
    ipr_options = IPR_Options(
        most_preferred_solutions=most_preferred_solutions,
        num_initial_reference_points=10000,
        version="convex_hull",
    )

    grpmoptions = GPRMOptions(method_options=ipr_options)
    zoomoptions = ZoomOptions(num_steps_remaining=4)

    current_options = FavOptions(
        GPRMoptions=grpmoptions,
        candidate_generation_options="mm",
        zoom_options=zoomoptions,
        original_most_preferred_solutions=most_preferred_solutions,
        votes=None,
        total_n_of_candidates=5
    )

    # --- 3. AUTOMATED ITERATION LOOP ---
    MAX_ITERS = 5
    results_history = []

    for iter_idx in range(MAX_ITERS):
        print(f"\n{'='*50}")
        print(f"--- Running Iteration {iter_idx + 1} / {MAX_ITERS} ---")
        print(f"{'='*50}")

        # 3.1 Evaluate Points & Generate Candidates
        fav_results = favorite_method(
            problem=dtlz2_problem,
            options=current_options,
            results_list=results_history
        )
        print("Candidates generated successfully.")

        # 3.2 Extract Data & Visualize Current State
        pts_mat, cents_mat, labels = cluster_points(fav_results)

        print("\nCandidate Centers:")
        print(cents_mat)

        # Call visualizer (will likely pause execution until window is closed depending on backend)
        visualize_3d_clusters(current_options.GPRMoptions, pts_mat, cents_mat, labels, len(fav_results.fair_solutions), iter_idx + 1)

        if iter_idx == MAX_ITERS - 1:
            print("\nFinal iteration complete. Reached maximum zooming depth.")
            break

        # 3.3 Simulate DMs Voting
        dm_names = list(most_preferred_solutions.keys())

        if iter_idx == 0:
            # Iteration 1: Clear win for Candidate 0
            votes = {dm_names[0]: 0, dm_names[1]: 0, dm_names[2]: 0, dm_names[3]: 1}
        elif iter_idx == 1:
            # Iteration 2: Force a Tie between Candidate 0 and Candidate 1
            votes = {dm_names[0]: 0, dm_names[1]: 0, dm_names[2]: 1, dm_names[3]: 1}
        else:
            # Subsequent Iterations: Clear win for Candidate 0
            votes = {dm_names[0]: 0, dm_names[1]: 0, dm_names[2]: 0, dm_names[3]: 0}

        print(f"\nSimulated Votes: {votes}")

        # Calculate vote distribution to find winner or tie
        vote_counts = {}
        for v in votes.values():
            vote_counts[v] = vote_counts.get(v, 0) + 1

        max_votes = max(vote_counts.values()) if vote_counts else 0
        tied_indices = [cand for cand, count in vote_counts.items() if count == max_votes]
        winning_idx = tied_indices[0] if len(tied_indices) == 1 else None

        compromise_solution = None
        active_labels = labels
        candidates_pool = fav_results.fair_solutions

        # 3.4 Tie-Breaker Routing (Adjacency Check & Average Projection)
        if winning_idx is None:
            print(f"⚠️ Tie detected among candidates {tied_indices}!")

            is_adjacent = False
            if len(tied_indices) == 2:
                idx_a, idx_b = tied_indices[0], tied_indices[1]
                pts_a = pts_mat[active_labels == idx_a]
                pts_b = pts_mat[active_labels == idx_b]

                if len(pts_a) > 0 and len(pts_b) > 0:
                    dists = cdist(pts_a, pts_b, metric='euclidean')
                    min_dist = np.min(dists)

                    internal_dists = cdist(pts_a, pts_a, metric='euclidean')
                    avg_internal_dist = np.mean(internal_dists) if len(internal_dists) > 0 else float('inf')

                    # Threshold: 1.5x the average distance between points in Cluster A
                    is_adjacent = min_dist < (avg_internal_dist * 1.5)

            if is_adjacent:
                print("   -> Clusters are geometric neighbors. Synthesizing Average Projection compromise.")
                tied_votes = {dm: v for dm, v in votes.items() if v in [tied_indices[0], tied_indices[1]]}
                compromise_solution = tie_breaker_avgproj(dtlz2_problem, tied_votes, candidates_pool)
            else:
                print("   -> Clusters are NOT adjacent. Falling back to random selection.")
                winning_idx = np.random.choice(tied_indices)

        # 3.5 Re-cluster around the new compromise (if tie-breaker generated one)
        if compromise_solution is not None:
            candidates_pool, active_labels, winning_idx = recluster_for_tie_breaker(
                all_points=fav_results.GPRMResults.raw_results.evaluated_points,
                existing_candidates=candidates_pool,
                compromise_solution=compromise_solution
            )
            print("   -> Reclustered Voronoi partitions around new compromise solution.")

        # 3.6 Advance Iteration (Calculate Hull Expansion & Shrink Space)
        rs = MAX_ITERS - iter_idx
        dynamic_fraction = calculate_fraction_to_keep(
            current_iter=iter_idx,
            max_iters=MAX_ITERS,
            num_objectives=len(dtlz2_problem.objectives)
        )

        print(f"\nWinning Cluster: {winning_idx}. Generating new candidates...")
        print(f"Fraction to keep: {dynamic_fraction:.3f}")

        next_mps = generate_next_iteration_mps(
            fav_results=fav_results,
            cluster_labels=active_labels,
            winning_idx=winning_idx,
            fraction_to_keep=dynamic_fraction,
            num_new_points=1000
        )

        # 3.7 Update options for the next loop
        current_options = current_options.model_copy(deep=True)
        current_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
        current_options.GPRMoptions.method_options.version = "convex_hull"
        current_options.zoom_options.num_steps_remaining = max(1, rs - 1)
        current_options.votes = votes

        # Save results to history
        results_history.append(fav_results)

    print("\nScript Execution Finished.")
