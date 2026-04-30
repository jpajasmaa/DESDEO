import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from desdeo.problem.testproblems import river_pollution_problem
    from desdeo.tools import PyomoIpoptSolver
    from desdeo.tools.scalarization import add_asf_diff

    from desdeo.gdm.favorite_method import (
        IPR_Options, GPRMOptions, ZoomOptions, FavOptions,
        favorite_method, generate_next_iteration_mps, cluster_points, select_final_candidates, select_final_candidates_simple
    )
    from desdeo.gdm.voting_rules import majority_rule
    from visualizations import visualize_pcp_clusters

    return (
        FavOptions,
        GPRMOptions,
        IPR_Options,
        ZoomOptions,
        cluster_points,
        favorite_method,
        generate_next_iteration_mps,
        majority_rule,
        mo,
        np,
        river_pollution_problem,
        select_final_candidates,
        visualize_pcp_clusters,
    )


@app.cell
def _(
    FavOptions,
    GPRMOptions,
    IPR_Options,
    ZoomOptions,
    mo,
    river_pollution_problem,
):
    # 1. Initialize 5-Objective River Pollution Problem
    river_problem = river_pollution_problem(five_objective_variant=False)
    obj_names = [obj.name for obj in river_problem.objectives]
    obj_symbols = [obj.symbol for obj in river_problem.objectives]
    print(obj_names)
    print(river_problem.get_ideal_point())
    print(river_problem.get_nadir_point())
    k_objectives = len(obj_names)
    m_dms = 3

    # Fetch the problem's bounds for realistic random generation
    ideal = river_problem.get_ideal_point()
    nadir = river_problem.get_nadir_point()

    fractions = [0.8, 0.6, 0.4, 0.2]
    MAX_ITERS = 4

    # 2. Dynamically Generate Feasible DMs' Most Preferred Solutions
    """
    most_preferred_solutions = {}
    for i in range(m_dms):
        dm_name = f"DM{i+1}"

        # Project random coordinates onto the front using the actual ideal and nadir
        random_target = {
            name: np.random.uniform(ideal[name], nadir[name]) 
            for name in obj_symbols
        }
        print(random_target)

        p, target = add_asf_diff(river_problem, symbol="asf", reference_point=random_target)
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        most_preferred_solutions[dm_name] = res.optimal_objectives
    """

    most_preferred_solutions = {
         "DM1": {obj_symbols[0]: 6.319, obj_symbols[1]: 3.038, obj_symbols[2]: 1.661, obj_symbols[3]: -0.906},
         "DM2": {obj_symbols[0]: 6.177, obj_symbols[1]: 3.404, obj_symbols[2]: 5.1, obj_symbols[3]: -8.008},
         "DM3": {obj_symbols[0]: 5.677, obj_symbols[1]: 3.342, obj_symbols[2]: 7.004, obj_symbols[3]: -6.150},
    }

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
        "ultimate_winner": None 
    })
    return (
        MAX_ITERS,
        fractions,
        get_state,
        obj_symbols,
        river_problem,
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
    river_problem,
):
    state = get_state()
    iter_idx = state["iter_idx"]
    current_options = state["current_options"]
    results_history = state["results_history"]
    final_candidates = state["final_candidates"]
    ultimate_winner = state["ultimate_winner"]

    if iter_idx < MAX_ITERS:
        # Evaluate Points & Generate Candidates normally
        fav_results = favorite_method(
            problem=river_problem,
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
            current_mps=fav_results.FavOptions.original_most_preferred_solutions
        )
        output = mo.ui.plotly(plot)
    else:
        output = mo.md("# Waiting...")

    output
    return


@app.cell
def _(MAX_ITERS, fav_results, final_candidates, iter_idx, mo, ultimate_winner):
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

        dm1_vote = mo.ui.dropdown(options=dropdown_options, value="Candidate 0", label="DM1 Vote")
        dm2_vote = mo.ui.dropdown(options=dropdown_options, value="Candidate 0", label="DM2 Vote")
        dm3_vote = mo.ui.dropdown(options=dropdown_options, value="Candidate 0", label="DM3 Vote")

        vote_inputs = mo.ui.array([dm1_vote, dm2_vote, dm3_vote])

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
    results_history,
    select_final_candidates,
    set_state,
    vote_form,
):
    if vote_form is not None and vote_form.value is not None:
        votes = {"DM1": vote_form.value[0], "DM2": vote_form.value[1], "DM3": vote_form.value[2]}
        winning_idx = majority_rule(votes)

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
                "final_candidates": None, "ultimate_winner": None
            })

        elif iter_idx == MAX_ITERS - 1:
            # Phase 2: We just voted on the LAST zoomed clusters. Generate the 5 final candidates!
            final_cands = select_final_candidates(fav_results, labels, winning_idx, n_candidates=5)

            set_state({
                "iter_idx": iter_idx + 1, "current_options": current_options,
                "results_history": results_history + [fav_results],
                "final_candidates": final_cands, "ultimate_winner": None
            })

        elif iter_idx == MAX_ITERS:
            # Phase 3: We just voted on the 5 final candidates. Pick the ultimate winner!
            winner = final_candidates[winning_idx]

            set_state({
                "iter_idx": iter_idx + 1, "current_options": current_options,
                "results_history": results_history,
                "final_candidates": final_candidates, "ultimate_winner": winner
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
