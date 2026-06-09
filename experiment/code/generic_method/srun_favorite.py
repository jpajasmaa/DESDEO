import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from desdeo.gdm.voting_rules import majority_rule
from desdeo.tools import PyomoIpoptSolver, ProximalSolver
from desdeo.tools.scalarization import add_asf_diff, add_asf_nondiff
from desdeo.problem.testproblems.dtlz2_problem import dtlz2
from desdeo.problem.testproblems import dmitry_forest_problem_disc

# Import all the new mathematically robust functions!
from desdeo.gdm.favorite_method import (
    IPR_Options, GPRMOptions, ZoomOptions, FavOptions, cluster_points,
    favorite_method, generate_next_iteration_mps,
    calculate_fraction_to_keep, tie_breaker_avgproj, recluster_for_tie_breaker, recluster_for_tie_breaker2, generate_next_mps_from_compromise
)
from visualizations import visualize_3d_clusters

# ==========================================
# NEW: 3D Visualization of the entire process
# ==========================================
def visualize_iterations_3d(
    results_history: list,
    labels_history: list[np.ndarray],
    phase_titles: list[str],  # <-- NEW PARAMETER
    title="Favorite Method: Objective Space Shrinking"
):
    """
    Creates a side-by-side 3D Plotly visualization of all Favorite Method iterations.
    """
    n_iters = len(results_history)

    # Create subplots using the dynamic phase titles!
    fig = make_subplots(
        rows=1, cols=n_iters,
        specs=[[{'type': 'scatter3d'} for _ in range(n_iters)]],
        subplot_titles=phase_titles
    )

    for i, (res, labels) in enumerate(zip(results_history, labels_history)):
        points = res.GPRMResults.raw_results.evaluated_points
        obj_keys = list(points[0].objectives.keys())
        k1, k2, k3 = obj_keys[0], obj_keys[1], obj_keys[2]

        p_x = [p.objectives[k1] for p in points]
        p_y = [p.objectives[k2] for p in points]
        p_z = [p.objectives[k3] for p in points]

        # Plot Point Cloud
        fig.add_trace(
            go.Scatter3d(
                x=p_x, y=p_y, z=p_z,
                mode='markers',
                marker=dict(size=3, color=labels, colorscale='Viridis', opacity=0.4),
                name=f"Iter {i+1} Points",
                showlegend=False,
                hovertext=[f"Cluster {L}" for L in labels],
                hoverinfo="text"
            ),
            row=1, col=i+1
        )

        # Plot Candidates
        cands = res.fair_solutions
        c_x = [c.objective_values[k1] for c in cands]
        c_y = [c.objective_values[k2] for c in cands]
        c_z = [c.objective_values[k3] for c in cands]

        # Determine the color: If it's a tie-breaker, the Compromise (Index 0) gets a special color (e.g., Orange)
        cand_colors = ['orange' if (i == 0 and "TIE" in phase_titles[i]) else 'red' for i in range(len(cands))]
        c_tags = [c.fairness_criterion for c in cands]

        fig.add_trace(
            go.Scatter3d(
                x=c_x, y=c_y, z=c_z,
                mode='markers+text',
                marker=dict(size=8, color=cand_colors, symbol='diamond', line=dict(width=2, color='black')),
                text=[f"C{idx}" for idx in range(len(cands))],
                textposition="top center",
                name=f"Iter {i+1} Candidates",
                hovertext=c_tags,
                hoverinfo="text"
            ),
            row=1, col=i+1
        )

        fig.update_scenes(xaxis_title=k1, yaxis_title=k2, zaxis_title=k3, row=1, col=i+1)

    fig.update_layout(title_text=title, height=700, margin=dict(l=0, r=0, b=0, t=50))
    fig.show(renderer="browser")

# ==========================================
# 1. DTLZ2 TEST
# ==========================================

def run_dtlz2():
    dtlz2_problem = dtlz2(8, 3)
    num_obj = len(dtlz2_problem.objectives)
    MAX_ITERS = 4

    reference_points = {
        "DM1": {"f_1": 0.0, "f_2": 0.9, "f_3": 0.5},
        "DM2": {"f_1": 0.5, "f_2": 0.0, "f_3": 0.9},
        "DM3": {"f_1": 0.9, "f_2": 0.5, "f_3": 0.0},
    }

    most_preferred_solutions = {}
    for dm in reference_points.keys():
        p, target = add_asf_diff(dtlz2_problem, symbol="asf", reference_point=reference_points[dm])
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        most_preferred_solutions[f"{dm}"] = res.optimal_objectives

    ipr_options = IPR_Options(most_preferred_solutions=most_preferred_solutions, num_initial_reference_points=10000, version="convex_hull")
    fav_options = FavOptions(
        GPRMoptions=GPRMOptions(method_options=ipr_options),
        candidate_generation_options="mm",
        zoom_options=ZoomOptions(num_steps_remaining=MAX_ITERS),
        original_most_preferred_solutions=most_preferred_solutions,
        votes=None, total_n_of_candidates=5
    )

    results_history = []
    labels_history = []
    phase_titles = []  # <--- Track the phase behavior for the Plotly titles
    current_options = fav_options

    # ==========================================
    # SCRIPTED VOTING SEQUENCE
    # ==========================================
    simulated_votes = [
        {"DM1": 0, "DM2": 0, "DM3": 1},  # Iter 1: Majority for Candidate 0 (NORMAL)
        {"DM1": 2, "DM2": 3, "DM3": 1},  # Iter 2: 3-Way Tie! (TIE-BREAKER)
        {"DM1": 1, "DM2": 1, "DM3": 0},  # Iter 3: Majority for Candidate 1 (NORMAL)
        {"DM1": 0, "DM2": 0, "DM3": 0},  # Iter 4: Unanimous for Candidate 0 (NORMAL)
    ]

    for iter_idx in range(MAX_ITERS):
        print(f"\n{'='*40}\n--- DTLZ2 RUNNING ITERATION {iter_idx + 1} ---\n{'='*40}")

        fav_results = favorite_method(problem=dtlz2_problem, options=current_options, results_list=results_history)
        results_history.append(fav_results)

        # Grab the scripted votes for this specific iteration
        votes = simulated_votes[iter_idx]
        candidates_pool = fav_results.fair_solutions
        winning_idx = majority_rule(votes)
        dynamic_fraction = calculate_fraction_to_keep(
            current_iter=iter_idx,
            max_iters=MAX_ITERS,
            num_objectives=num_obj
        )
        print(f"Mathematical Fraction to keep: {dynamic_fraction:.4f}")

        if winning_idx is None:
            print(">>> TIE DETECTED! Triggering compromise override... <<<")
            phase_titles.append(f"Iter {iter_idx+1}: TIE-BREAKER")

            compromise_solution = tie_breaker_avgproj(dtlz2_problem, votes, candidates_pool)

            # 3. USE YOUR NEW FUNCTION!
            next_mps = generate_next_mps_from_compromise(
                fav_results=fav_results,
                compromise_solution=compromise_solution,
                fraction_to_keep=dynamic_fraction
            )
            # ==========================================
            # 4. Formatting for the 3D Visualizer
            # (We manually highlight the N closest points to show the Sphere!)
            all_points = fav_results.GPRMResults.raw_results.evaluated_points
            obj_keys = list(all_points[0].objectives.keys())

            pts_mat = np.array([[p.objectives[k] for k in obj_keys] for p in all_points])
            comp_obj = np.array([compromise_solution.objective_values[k] for k in obj_keys])

            dists = np.linalg.norm(pts_mat - comp_obj, axis=1)
            n_keep = max(int(np.ceil(len(pts_mat) * dynamic_fraction)), len(obj_keys) + 1)
            top_indices = np.argsort(dists)[:n_keep]

            # Label Kept Points as '0' (Colored) and Discarded as '1' (Greyed out)
            active_labels = np.ones(len(pts_mat))
            active_labels[top_indices] = 0

            cents_mat = np.array([comp_obj])
            candidates_pool = [compromise_solution]  # Only the compromise is shown

            # history_data["tie"][iter_idx] = {
            #    "pts": pts_mat, "cents": cents_mat, "labels": active_labels, "cands": candidates_pool
            # }

        else:
            print(f">>> NORMAL MAJORITY! Candidate {winning_idx} wins. <<<")
            phase_titles.append(f"Iter {iter_idx+1}: NORMAL (Cand {winning_idx})")
            pts_mat, cents_mat, active_labels = cluster_points(fav_results)

            next_mps = generate_next_iteration_mps(
                fav_results=fav_results,
                cluster_labels=active_labels,
                winning_idx=winning_idx,
                fraction_to_keep=dynamic_fraction
            )

        labels_history.append(active_labels)

        visualize_3d_clusters(
            options=current_options.GPRMoptions, points_arr=pts_mat, centers_arr=cents_mat,
            labels=active_labels, n_predetermined=len(candidates_pool), iter_n=iter_idx + 1
        )

        current_options = current_options.model_copy(deep=True)
        current_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
        current_options.GPRMoptions.method_options.version = "convex_hull"
        current_options.zoom_options.num_steps_remaining = max(1, MAX_ITERS - iter_idx - 1)
        current_options.votes = votes

    # Pass the titles to the Plotly visualizer!
    visualize_iterations_3d(results_history, labels_history, phase_titles, title="DTLZ2 3D Objective Space Shrinking")


def run_dtlz2_old():
    dtlz2_problem = dtlz2(8, 3)
    num_obj = len(dtlz2_problem.objectives)
    MAX_ITERS = 4

    reference_points = {
        "DM1": {"f_1": 0.0, "f_2": 0.9, "f_3": 0.5},
        "DM2": {"f_1": 0.5, "f_2": 0.0, "f_3": 0.9},
        "DM3": {"f_1": 0.9, "f_2": 0.5, "f_3": 0.0},
    }

    most_preferred_solutions = {}
    for dm in reference_points.keys():
        p, target = add_asf_diff(dtlz2_problem, symbol="asf", reference_point=reference_points[dm])
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        most_preferred_solutions[f"{dm}"] = res.optimal_objectives

    ipr_options = IPR_Options(most_preferred_solutions=most_preferred_solutions, num_initial_reference_points=10000, version="convex_hull")
    fav_options = FavOptions(
        GPRMoptions=GPRMOptions(method_options=ipr_options),
        candidate_generation_options="mm",
        zoom_options=ZoomOptions(num_steps_remaining=MAX_ITERS),
        original_most_preferred_solutions=most_preferred_solutions,
        votes=None, total_n_of_candidates=5
    )

    results_history = []
    labels_history = []
    current_options = fav_options

    for iter_idx in range(MAX_ITERS):
        print(f"\n{'='*40}\n--- DTLZ2 RUNNING ITERATION {iter_idx + 1} ---\n{'='*40}")

        # 1. Run Core Method
        fav_results = favorite_method(problem=dtlz2_problem, options=current_options, results_list=results_history)
        results_history.append(fav_results)

        # 2. Simulate DM Voting (This creates a 3-way Tie!)
        votes = {"DM1": 2, "DM2": 0, "DM3": 1}
        candidates_pool = fav_results.fair_solutions
        winning_idx = majority_rule(votes)

        # 3. Handle Tie-Breaker and Clustering
        if winning_idx is None:
            print("Tie detected! Triggering compromise override...")
            compromise_solution = tie_breaker_avgproj(dtlz2_problem, votes, candidates_pool)
            all_points = fav_results.GPRMResults.raw_results.evaluated_points

            candidates_pool, active_labels, winning_idx = recluster_for_tie_breaker2(
                all_points=all_points,
                existing_candidates=candidates_pool,
                compromise_solution=compromise_solution
            )

            # Setup visualization matrices based on the override
            obj_keys = list(all_points[0].objectives.keys())
            pts_mat = np.array([[p.objectives[k] for k in obj_keys] for p in all_points])
            cents_mat = np.array([[c.objective_values[k] for k in obj_keys] for c in candidates_pool])
        else:
            pts_mat, cents_mat, active_labels = cluster_points(fav_results)

        labels_history.append(active_labels)

        # 4. Calculate Dynamic Shrinking Fraction!
        dynamic_fraction = calculate_fraction_to_keep(
            current_iter=iter_idx,
            max_iters=MAX_ITERS,
            num_objectives=num_obj
        )
        print(f"Mathematical Fraction to keep: {dynamic_fraction:.4f}")

        # 5. Generate Next Iteration MPS
        next_mps = generate_next_iteration_mps(
            fav_results=fav_results,
            cluster_labels=active_labels,
            winning_idx=winning_idx,
            fraction_to_keep=dynamic_fraction
        )

        # 6. Visualize the Current Iteration (Optional Single View)
        visualize_3d_clusters(
            options=current_options.GPRMoptions, points_arr=pts_mat, centers_arr=cents_mat,
            labels=active_labels, n_predetermined=len(candidates_pool), iter_n=iter_idx + 1
        )

        # 7. Prepare Options for Next Iteration
        current_options = current_options.model_copy(deep=True)
        current_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
        current_options.GPRMoptions.method_options.version = "convex_hull"
        current_options.zoom_options.num_steps_remaining = max(1, MAX_ITERS - iter_idx - 1)
        current_options.votes = votes

    # Show the full 3D historical progression!
    visualize_iterations_3d(results_history, labels_history, title="DTLZ2 3D Objective Space Shrinking")


# ==========================================
# 2. DMITRY FOREST TEST
# ==========================================
def run_dmitry_forest_problem():
    problem = dmitry_forest_problem_disc()
    obj_names = [obj.name for obj in problem.objectives]
    obj_symbols = [obj.symbol for obj in problem.objectives]
    num_obj = len(obj_symbols)
    MAX_ITERS = 4

    ideal = problem.get_ideal_point()
    nadir = problem.get_nadir_point()

    n_of_dms = 3
    most_preferred_solutions = {}
    for i in range(n_of_dms):
        dm_name = f"DM{i+1}"
        random_target = {name: np.random.uniform(ideal[name], nadir[name]) for name in obj_symbols}
        p, target = add_asf_nondiff(problem, symbol="asf", reference_point=random_target)
        solver = ProximalSolver(p)
        res = solver.solve(target)
        most_preferred_solutions[dm_name] = res.optimal_objectives

    ipr_options = IPR_Options(most_preferred_solutions=most_preferred_solutions, num_initial_reference_points=10000, version="convex_hull")
    fav_options = FavOptions(
        GPRMoptions=GPRMOptions(method_options=ipr_options),
        candidate_generation_options="mm",
        zoom_options=ZoomOptions(num_steps_remaining=MAX_ITERS),
        original_most_preferred_solutions=most_preferred_solutions,
        votes=None, total_n_of_candidates=5
    )

    results_history = []
    labels_history = []
    current_options = fav_options

    for iter_idx in range(MAX_ITERS):
        print(f"\n{'='*40}\n--- FOREST RUNNING ITERATION {iter_idx + 1} ---\n{'='*40}")

        fav_results = favorite_method(problem=problem, options=current_options, results_list=results_history)
        results_history.append(fav_results)

        # Unanimous voting: Tie-breaker will NOT trigger here
        votes = {"DM1": 2, "DM2": 2, "DM3": 2}
        candidates_pool = fav_results.fair_solutions
        winning_idx = majority_rule(votes)

        if winning_idx is None:
            compromise_solution = tie_breaker_avgproj(problem, votes, candidates_pool)
            all_points = fav_results.GPRMResults.raw_results.evaluated_points
            candidates_pool, active_labels, winning_idx = recluster_for_tie_breaker(all_points, candidates_pool, compromise_solution)
            obj_keys = list(all_points[0].objectives.keys())
            pts_mat = np.array([[p.objectives[k] for k in obj_keys] for p in all_points])
            cents_mat = np.array([[c.objective_values[k] for k in obj_keys] for c in candidates_pool])
        else:
            pts_mat, cents_mat, active_labels = cluster_points(fav_results)

        labels_history.append(active_labels)

        dynamic_fraction = calculate_fraction_to_keep(current_iter=iter_idx, max_iters=MAX_ITERS, num_objectives=num_obj)
        print(f"Mathematical Fraction to keep: {dynamic_fraction:.4f}")

        next_mps = generate_next_iteration_mps(
            fav_results=fav_results, cluster_labels=active_labels, winning_idx=winning_idx, fraction_to_keep=dynamic_fraction
        )

        visualize_3d_clusters(
            options=current_options.GPRMoptions, points_arr=pts_mat, centers_arr=cents_mat,
            labels=active_labels, n_predetermined=len(candidates_pool), iter_n=iter_idx + 1
        )

        current_options = current_options.model_copy(deep=True)
        current_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
        current_options.GPRMoptions.method_options.version = "convex_hull"
        current_options.zoom_options.num_steps_remaining = max(1, MAX_ITERS - iter_idx - 1)
        current_options.votes = votes

    visualize_iterations_3d(results_history, labels_history, title="Forest Problem Objective Space Shrinking")


if __name__ == "__main__":
    run_dtlz2()
    # run_dmitry_forest_problem()
    print("Favorite finished")
