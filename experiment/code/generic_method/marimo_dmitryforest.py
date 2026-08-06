import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import random
    from scipy.spatial.distance import cdist
    from desdeo.problem.testproblems import river_pollution_problem, dmitry_forest_problem_disc
    from desdeo.tools import PyomoIpoptSolver, ProximalSolver
    from desdeo.tools.scalarization import add_asf_diff, add_asf_nondiff
    from desdeo.gdm.favorite_method import (
        IPR_Options, GPRMOptions, ZoomOptions, FavOptions,
        favorite_method, generate_next_iteration_mps, cluster_points, select_final_candidates, tie_breaker_avgproj, recluster_for_tie_breaker, calculate_fraction_to_keep
    )
    from desdeo.gdm.voting_rules import majority_rule
    from visualizations import visualize_pcp_clusters

    return (
        FavOptions,
        GPRMOptions,
        IPR_Options,
        ProximalSolver,
        ZoomOptions,
        add_asf_nondiff,
        calculate_fraction_to_keep,
        cdist,
        cluster_points,
        dmitry_forest_problem_disc,
        favorite_method,
        generate_next_iteration_mps,
        mo,
        np,
        random,
        recluster_for_tie_breaker,
        select_final_candidates,
        tie_breaker_avgproj,
        visualize_pcp_clusters,
    )


@app.cell
def _(
    FavOptions,
    GPRMOptions,
    IPR_Options,
    ProximalSolver,
    ZoomOptions,
    add_asf_nondiff,
    dmitry_forest_problem_disc,
    mo,
):
    # --- 1. SETUP PROBLEM ---
    problem = dmitry_forest_problem_disc()
    obj_names = [obj.name for obj in problem.objectives]
    obj_symbols = [obj.symbol for obj in problem.objectives]

    # Fetch the problem's bounds for realistic random generation
    ideal = problem.get_ideal_point()
    nadir = problem.get_nadir_point()
    print(ideal, nadir)

    # fractions = [0.8, 0.6, 0.4, 0.2]
    # new_list = [1.**4, 0.8**4, 0.6**4, 0.4**4, 0.2**4, 0**4]
    # raise nautilus fractions to power of nunmber of obje
    # new2list = [(new_list[i+1]/new_list[i]) for i, _ in range(len(new_list)-1) ]
    MAX_ITERS = 5

    n_of_dms = 4
    rp = {
        "DM1": {'Rev': 230., 'HA': 20215, 'Carb': 2944, 'DW': 180},
        "DM2": {'Rev': 111, 'HA': 18225, 'Carb': 3200, 'DW': 200},
        "DM3": {'Rev': 160, 'HA': 11232, 'Carb': 4000, 'DW': 90},
        "DM4": {'Rev': 140, 'HA': 14232, 'Carb': 4100, 'DW': 190},
        # "DM5": {'Rev': 120, 'HA': 13232, 'Carb': 3300, 'DW': 140},
    }
    """
    for i in range(n_of_dms):
        dm_name = f"DM{i+1}"
        random_target = {
            name: np.random.uniform(ideal[name], nadir[name])
            for name in obj_symbols
        }
    """

    most_preferred_solutions = {}
    for i in range(n_of_dms):
        p, target = add_asf_nondiff(problem, symbol="asf", reference_point=rp[f"DM{i+1}"])
        solver = ProximalSolver(p)
        res = solver.solve(target)
        most_preferred_solutions[f"DM{i+1}"] = res.optimal_objectives

    # 3. Configure the Initial Engine Options
    ipr_options = IPR_Options(
        most_preferred_solutions=most_preferred_solutions,
        num_initial_reference_points=10000,
        version="box",
    )
    initial_fav_options = FavOptions(
        GPRMoptions=GPRMOptions(method_options=ipr_options),
        candidate_generation_options="mm",
        zoom_options=ZoomOptions(num_steps_remaining=4),
        original_most_preferred_solutions=most_preferred_solutions,
        votes=None,
        total_n_of_candidates=5
    )

    # 4. MARIMO STATE TRACKER - Split into two to prevent reactive computation loops
    get_state, set_state = mo.state({
        "iter_idx": 0,
        "current_options": initial_fav_options,
        "results_history": [],
        "final_candidates": None,
        "ultimate_winner": None,
        "current_dm_preferred": most_preferred_solutions
    })

    # NEW: Isolated state just for handling temporary UI tie-breaker re-votes
    get_tie_state, set_tie_state = mo.state(None)
    return (
        MAX_ITERS,
        get_state,
        get_tie_state,
        n_of_dms,
        obj_symbols,
        problem,
        set_state,
        set_tie_state,
    )


@app.cell
def _(
    MAX_ITERS,
    cluster_points,
    favorite_method,
    get_state,
    np,
    obj_symbols,
    problem,
):
    # Only listens to the core algorithmic state, ignoring tie_state UI changes
    _state = get_state()
    iter_idx = _state["iter_idx"]
    current_options = _state["current_options"]
    results_history = _state["results_history"]
    final_candidates = _state["final_candidates"]
    ultimate_winner = _state["ultimate_winner"]
    current_dm_preferred = _state["current_dm_preferred"]

    fav_results = None
    pts_mat, cents_mat, labels = None, None, None
    n_predetermined = 0

    if iter_idx < MAX_ITERS:
        fav_results = favorite_method(
            problem=problem,
            options=current_options,
            results_list=results_history
        )
        pts_mat, cents_mat, labels = cluster_points(fav_results)

    elif iter_idx >= MAX_ITERS and len(results_history) > 0:
        fav_results = results_history[-1]
        pts_mat, _, labels = cluster_points(fav_results)

        if ultimate_winner is not None:
            cents_mat = np.array([[ultimate_winner.objective_values[k] for k in obj_symbols]])
            n_predetermined = 1
        elif final_candidates is not None:
            cents_mat = np.array([[c.objective_values[k] for k in obj_symbols] for c in final_candidates])
            n_predetermined = len(final_candidates)
    return (
        cents_mat,
        current_dm_preferred,
        current_options,
        fav_results,
        final_candidates,
        iter_idx,
        labels,
        n_predetermined,
        pts_mat,
        results_history,
        ultimate_winner,
    )


@app.cell
def _(
    MAX_ITERS,
    cents_mat,
    current_dm_preferred,
    current_options,
    fav_results,
    final_candidates,
    iter_idx,
    labels,
    mo,
    n_predetermined,
    pts_mat,
    visualize_pcp_clusters,
):
    if fav_results is not None:
        # Determine how many "centers" we are drawing
        n_pred = len(fav_results.fair_solutions) if iter_idx < MAX_ITERS else n_predetermined

        plot = visualize_pcp_clusters(
            options=current_options.GPRMoptions,
            points_arr=pts_mat,
            centers_arr=cents_mat,
            labels=labels,
            n_predetermined=n_pred,
            iter_n=iter_idx + 1 if iter_idx < MAX_ITERS else "FINAL PHASE",
            # current_mps=fav_results.FavOptions.original_most_preferred_solutions
            current_mps=current_dm_preferred,
        )

        table_data = []
        # 1. Ideal & Nadir Bounds
        ideal_row = {"Name": "Ideal"}
        ideal_row.update({k: round(v, 3) for k, v in current_options.GPRMoptions.fake_ideal.items()})
        table_data.append(ideal_row)

        nadir_row = {"Name": "Nadir"}
        nadir_row.update({k: round(v, 3) for k, v in current_options.GPRMoptions.fake_nadir.items()})
        table_data.append(nadir_row)

        # Candidates
        candidates_pool_visu = fav_results.fair_solutions if iter_idx < MAX_ITERS else final_candidates
        if candidates_pool_visu is not None:
            for ic, cand in enumerate(candidates_pool_visu):
                cand_row = {"Name": f"Candidate {ic}"}
                cand_row.update({k: round(v, 3) for k, v in cand.objective_values.items()})
                table_data.append(cand_row)

        # DM Preferred Solutions
        if current_dm_preferred is not None:
            for dm_name, obj_vals in current_dm_preferred.items():
                dm_row = {"Name": f"{dm_name} Preferred"}
                dm_row.update({k: round(v, 3) for k, v in obj_vals.items()})
                table_data.append(dm_row)

        data_table = mo.ui.table(table_data, selection=None, pagination=False)
        plot_ui = mo.ui.plotly(plot)

        # Combine horizontally. [1, 2] means the plot gets twice the width of the table.
        output = mo.hstack([data_table, plot_ui], widths=[1, 3], align="center")

        # output = mo.ui.plotly(plot)
    else:
        output = mo.md("# Waiting...")

    output
    return


@app.cell
def _(
    MAX_ITERS,
    fav_results,
    final_candidates,
    get_tie_state,
    iter_idx,
    mo,
    n_of_dms,
    ultimate_winner,
):
    _tie_state = get_tie_state()

    ui_layout = None
    vote_form = None

    if ultimate_winner is None and fav_results is not None:
        _candidates_pool = fav_results.fair_solutions if iter_idx < MAX_ITERS else final_candidates
        _n_candidates = len(_candidates_pool)

        if _tie_state is None:
            _title = f"### Place Votes for Iteration {iter_idx + 1}" if iter_idx < MAX_ITERS else "### Place Votes for the FINAL Solution"
            _btn_label = "Submit Votes"
            _dropdown_options = {f"Candidate {i}": i for i in range(_n_candidates)}

        else:
            # We are in a Tie-Breaker Re-Vote
            _tied_cands = _tie_state["tied_indices"]
            _title = f"### ⚠️ TIE DETECTED! Re-Vote Among Candidates: {_tied_cands}"
            _btn_label = "Submit Re-Vote"
            _dropdown_options = {f"Candidate {i}": i for i in _tied_cands}

        _dmvote_arr = [
            mo.ui.dropdown(options=_dropdown_options, value=list(_dropdown_options.keys())[0], label=f"DM{ii+1} Vote")
            for ii in range(n_of_dms)
        ]

        _vote_inputs = mo.ui.array(_dmvote_arr)
        vote_form = mo.ui.form(element=_vote_inputs, submit_button_label=_btn_label)

        ui_layout = mo.vstack([
            mo.md("---"),
            mo.md(_title),
            vote_form
        ])

    elif ultimate_winner is not None:
        _winner_str = ", ".join([f"**{k}**: {v:.3f}" for k, v in ultimate_winner.objective_values.items()])
        ui_layout = mo.md(f"# 🎉 Optimization Finished! \n### The Final Selected Solution is:\n{_winner_str}")

    ui_layout
    return (vote_form,)


@app.cell
def _(
    MAX_ITERS,
    calculate_fraction_to_keep,
    cdist,
    current_options,
    fav_results,
    final_candidates,
    generate_next_iteration_mps,
    get_state,
    get_tie_state,
    iter_idx,
    labels,
    np,
    problem,
    pts_mat,
    random,
    recluster_for_tie_breaker,
    results_history,
    select_final_candidates,
    set_state,
    set_tie_state,
    tie_breaker_avgproj,
    vote_form,
):
    def _process_vote():
        if vote_form is not None and vote_form.value is not None:
            state = get_state()
            tie_state = get_tie_state()

            dm_names = list(fav_results.FavOptions.original_most_preferred_solutions.keys())
            votes = {dm_names[i]: vote_form.value[i] for i in range(len(dm_names))}

            candidates_pool = fav_results.fair_solutions if iter_idx < MAX_ITERS else final_candidates

            # Calculate vote distribution
            vote_counts = {}
            for v in votes.values():
                vote_counts[v] = vote_counts.get(v, 0) + 1

            max_votes = max(vote_counts.values()) if vote_counts else 0
            tied_indices = [cand for cand, count in vote_counts.items() if count == max_votes]

            winning_idx = tied_indices[0] if len(tied_indices) == 1 else None

            compromise_solution = None
            active_labels = labels

            new_dm_preferred = {}
            for dm, v_idx in votes.items():
                new_dm_preferred[dm] = candidates_pool[v_idx].objective_values

            pause_iteration = False

            # =========================================================
            # AUTOMATED TIE BREAKER ROUTING LOGIC
            # =========================================================
            if winning_idx is None:
                # 1. Check for Adjacency (if exactly 2 tied candidates)
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

                # 2. Route based on Adjacency
                if is_adjacent:
                    # Automatically combine adjacent clusters without re-voting
                    tied_votes = {dm: v for dm, v in votes.items() if v in [tied_indices[0], tied_indices[1]]}
                    compromise_solution = tie_breaker_avgproj(problem, tied_votes, candidates_pool)
                    set_tie_state(None)  # Ensure UI is cleared
                else:
                    # Clusters are not adjacent. Proceed to Re-Vote pipeline.
                    if tie_state is None:
                        # Pause iteration and trigger the Re-Vote UI
                        set_tie_state({"tied_indices": tied_indices})
                        pause_iteration = True
                    else:
                        # Re-vote tied again. Fallback to random to force progress.
                        winning_idx = random.choice(tied_indices)
                        set_tie_state(None)

            # =========================================================
            # PROCEED WITH ITERATION ADVANCEMENT
            # =========================================================
            if not pause_iteration:
                if compromise_solution is not None and iter_idx < MAX_ITERS:
                    all_points = fav_results.GPRMResults.raw_results.evaluated_points
                    candidates_pool, active_labels, winning_idx = recluster_for_tie_breaker(
                        all_points=all_points,
                        existing_candidates=candidates_pool,
                        compromise_solution=compromise_solution
                    )

                if iter_idx < MAX_ITERS - 1:
                    rs = MAX_ITERS - iter_idx
                    dynamic_fraction = calculate_fraction_to_keep(
                        current_iter=iter_idx,
                        max_iters=MAX_ITERS,
                        num_objectives=len(problem.objectives)
                    )

                    next_mps = generate_next_iteration_mps(
                        fav_results=fav_results, cluster_labels=active_labels, winning_idx=winning_idx, fraction_to_keep=dynamic_fraction
                    )

                    new_options = current_options.model_copy(deep=True)
                    new_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
                    new_options.GPRMoptions.method_options.version = "convex_hull"
                    new_options.zoom_options.num_steps_remaining = max(1, rs - 1)
                    new_options.votes = votes

                    set_state({
                        "iter_idx": iter_idx + 1,
                        "current_options": new_options,
                        "results_history": results_history + [fav_results],
                        "final_candidates": None,
                        "ultimate_winner": None,
                        "current_dm_preferred": new_dm_preferred
                    })

                elif iter_idx == MAX_ITERS - 1:
                    final_cands = select_final_candidates(problem, fav_results, active_labels, winning_idx, n_candidates=5)
                    if compromise_solution is not None:
                        final_cands[0] = compromise_solution

                    set_state({
                        "iter_idx": iter_idx + 1,
                        "current_options": current_options,
                        "results_history": results_history + [fav_results],
                        "final_candidates": final_cands,
                        "ultimate_winner": None,
                        "current_dm_preferred": new_dm_preferred
                    })

                elif iter_idx == MAX_ITERS:
                    winner = compromise_solution if compromise_solution is not None else final_candidates[winning_idx]

                    set_state({
                        "iter_idx": iter_idx + 1,
                        "current_options": current_options,
                        "results_history": results_history,
                        "final_candidates": final_candidates,
                        "ultimate_winner": winner,
                        "current_dm_preferred": new_dm_preferred
                    })

    _process_vote()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
