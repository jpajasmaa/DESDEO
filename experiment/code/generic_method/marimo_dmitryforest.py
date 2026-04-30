import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from desdeo.problem.testproblems import river_pollution_problem, dmitry_forest_problem_disc
    from desdeo.tools import PyomoIpoptSolver, ProximalSolver
    from desdeo.tools.scalarization import add_asf_diff, add_asf_nondiff
    from desdeo.gdm.favorite_method import (
        IPR_Options, GPRMOptions, ZoomOptions, FavOptions,
        favorite_method, generate_next_iteration_mps, cluster_points, select_final_candidates, tie_breaker_avgproj
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
        cluster_points,
        dmitry_forest_problem_disc,
        favorite_method,
        generate_next_iteration_mps,
        majority_rule,
        mo,
        np,
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

    fractions = [0.8, 0.6, 0.4, 0.2]
    MAX_ITERS = 4

    n_of_dms = 5
    rp = {
        "DM1": {'Rev': 240., 'HA': 12225, 'Carb': 2944, 'DW': 180},
        "DM2": {'Rev': 111, 'HA': 18225, 'Carb': 3200, 'DW': 200},
        "DM3": {'Rev': 160, 'HA': 15232, 'Carb': 4000, 'DW': 90},
        "DM4": {'Rev': 120, 'HA': 14232, 'Carb': 4100, 'DW': 190},
        "DM5": {'Rev': 120, 'HA': 13232, 'Carb': 3300, 'DW': 140},
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

    # 4. MARIMO STATE TRACKER
    get_state, set_state = mo.state({
        "iter_idx": 0,
        "current_options": initial_fav_options,
        "results_history": [],
        "final_candidates": None,
        "ultimate_winner": None,
        "current_dm_preferred": most_preferred_solutions
    })
    return (
        MAX_ITERS,
        fractions,
        get_state,
        n_of_dms,
        obj_symbols,
        problem,
        set_state,
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
    state = get_state()
    iter_idx = state["iter_idx"]
    current_options = state["current_options"]
    results_history = state["results_history"]
    final_candidates = state["final_candidates"]
    ultimate_winner = state["ultimate_winner"]
    current_dm_preferred = state["current_dm_preferred"]

    if iter_idx < MAX_ITERS:
        # Evaluate Points & Generate Candidates normally
        fav_results = favorite_method(
            problem=problem,
            options=current_options,
            results_list=results_history
        )
        pts_mat, cents_mat, labels = cluster_points(fav_results)

    elif iter_idx >= MAX_ITERS:
        # Fetch the last results to display the background points
        fav_results = results_history[-1]
        pts_mat, _, labels = cluster_points(fav_results)

        if ultimate_winner is not None:
            # Show only the final chosen winner
            cents_mat = np.array([[ultimate_winner.objective_values[k] for k in obj_symbols]])
            n_predetermined = 1
        elif final_candidates is not None:
            # Show the 5 randomly sampled candidates for the final vote
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
            #current_mps=fav_results.FavOptions.original_most_preferred_solutions
            current_mps=current_dm_preferred,
        )
        output = mo.ui.plotly(plot)
    else:
        output = mo.md("# Waiting...")

    output
    return


@app.cell
def _(
    MAX_ITERS,
    fav_results,
    final_candidates,
    iter_idx,
    mo,
    n_of_dms,
    ultimate_winner,
):
    if ultimate_winner is None and fav_results is not None:

        if iter_idx < MAX_ITERS:
            n_candidates = len(fav_results.fair_solutions)
            title = f"### Place Votes for Iteration {iter_idx + 1}"
            btn_label = "Submit Votes"
        else:
            n_candidates = len(final_candidates)
            title = "### Place Votes for the FINAL Solution"
            btn_label = "Select Ultimate Winner"

        dropdown_options = {f"Candidate {i}": i for i in range(n_candidates)}

        dmvote_arr = []
        for ii in range(n_of_dms):
            dmvote_arr.append(mo.ui.dropdown(options=dropdown_options, value="Candidate 0", label=f"DM{ii+1} Vote"))
        #dm2_vote = mo.ui.dropdown(options=dropdown_options, value="Candidate 0", label="DM2 Vote")
        #dm3_vote = mo.ui.dropdown(options=dropdown_options, value="Candidate 0", label="DM3 Vote")

        vote_inputs = mo.ui.array(dmvote_arr)

        vote_form = mo.ui.form(element=vote_inputs, submit_button_label=btn_label)
        ui_layout = mo.vstack([mo.md(title), vote_form])

    elif ultimate_winner is not None:
        # Print the final dictionary values cleanly!
        winner_str = ", ".join([f"**{k}**: {v:.3f}" for k, v in ultimate_winner.objective_values.items()])
        ui_layout = mo.md(f"# 🎉 Optimization Finished! \n### The Final Selected Solution is:\n{winner_str}")
    else:
        ui_layout = None

    ui_layout
    return (vote_form,)


@app.cell
def _(
    MAX_ITERS,
    current_options,
    fav_results,
    final_candidates,
    fractions,
    generate_next_iteration_mps,
    iter_idx,
    labels,
    majority_rule,
    np,
    obj_symbols,
    problem,
    results_history,
    select_final_candidates,
    set_state,
    tie_breaker_avgproj,
    vote_form,
):
    if vote_form is not None and vote_form.value is not None:
        dm_names = list(fav_results.FavOptions.original_most_preferred_solutions.keys())
        votes = {dm_names[i]: vote_form.value[i] for i in range(len(dm_names))}
        winning_idx = majority_rule(votes)
    
        compromise_solution = None
        candidates_pool = fav_results.fair_solutions if iter_idx < MAX_ITERS else final_candidates
        if winning_idx is None:
            compromise_solution = tie_breaker_avgproj(problem, votes, candidates_pool)

            # To seamlessly integrate the new compromise into our spatial expansion (Phase 1 & 2),
            # we find the existing cluster that is geometrically closest to this new mathematical compromise.
            if iter_idx < MAX_ITERS:
                candidates_arr = np.array([[c.objective_values[k] for k in obj_symbols] for c in candidates_pool])
                comp_arr = np.array([[compromise_solution.objective_values[k] for k in obj_symbols]])
                winning_idx = int(np.argmin(np.linalg.norm(candidates_arr - comp_arr, axis=1)))

        new_dm_preferred = {}
        for dm, v_idx in votes.items():
            new_dm_preferred[dm] = candidates_pool[v_idx].objective_values
    
        if iter_idx < MAX_ITERS - 1:
            # Phase 1: Normal zoom/expansion
            next_mps = generate_next_iteration_mps(
                fav_results=fav_results, cluster_labels=labels, winning_idx=winning_idx, fraction_to_keep=fractions[iter_idx]
            )
            new_options = current_options.model_copy(deep=True)
            new_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
            new_options.GPRMoptions.method_options.version = "convex_hull"
            new_options.zoom_options.num_steps_remaining = max(1, 4 - iter_idx - 1)
            new_options.votes = votes

            set_state({
                "iter_idx": iter_idx + 1, "current_options": new_options,
                "results_history": results_history + [fav_results],
                "final_candidates": None, "ultimate_winner": None,
                "current_dm_preferred": new_dm_preferred,
            })

        elif iter_idx == MAX_ITERS - 1:
            # Phase 2: We just voted on the LAST zoomed clusters. Generate the 5 final candidates!
            final_cands = select_final_candidates(fav_results, labels, winning_idx, n_candidates=5)
            # If a tie-brConduct a deepdive on ARE (the reit). Include evaluation of the latest earnings, guidance, analyst targets, evaluate how realistic is it to keep the current dividend (and how much is it). Evaluate intrinsic and fair value and create bear and bull scenarios with price targets for end of 2026 and 2030.eaker occurred, force the core candidate to be our new compromise!
            if compromise_solution is not None:
                final_cands[0] = compromise_solution

            set_state({
                "iter_idx": iter_idx + 1, "current_options": current_options,
                "results_history": results_history + [fav_results],
                "final_candidates": final_cands, "ultimate_winner": None, "current_dm_preferred": new_dm_preferred,
            })

        elif iter_idx == MAX_ITERS:
            # Phase 3: Pick the ultimate winner
            # If it tied here, the compromise IS the winner. Otherwise, use the voted index.
            winner = compromise_solution if compromise_solution is not None else final_candidates[winning_idx]

            set_state({
                "iter_idx": iter_idx + 1, "current_options": current_options,
                "results_history": results_history,
                "final_candidates": final_candidates, "ultimate_winner": winner, "current_dm_preferred": new_dm_preferred,
            })
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
