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
        favorite_method, generate_next_iteration_mps, cluster_points
    )
    from desdeo.gdm.voting_rules import majority_rule
    from visualizations import visualize_pcp_clusters

    return (
        FavOptions,
        GPRMOptions,
        IPR_Options,
        PyomoIpoptSolver,
        ZoomOptions,
        add_asf_diff,
        cluster_points,
        favorite_method,
        generate_next_iteration_mps,
        majority_rule,
        mo,
        np,
        river_pollution_problem,
        visualize_pcp_clusters,
    )


@app.cell
def _(
    FavOptions,
    GPRMOptions,
    IPR_Options,
    PyomoIpoptSolver,
    ZoomOptions,
    add_asf_diff,
    mo,
    np,
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
    most_preferred_solutions = {}
    for i in range(m_dms):
        dm_name = f"DM{i+1}"

        # ---------------------------------------------------------
        # MANUAL PREFERENCES (Commented Out)
        # Uncomment the block below to override the random generation 
        # and explicitly set the starting ideals for each DM.
        # ---------------------------------------------------------
        # most_preferred_solutions = {
        #     "DM1": {obj_names[0]: -3.0, obj_names[1]: -2.0, obj_names[2]: -0.5, obj_names[3]: 1.0, obj_names[4]: 4.0},
        #     "DM2": {obj_names[0]: -4.5, obj_names[1]: -1.5, obj_names[2]: -1.5, obj_names[3]: 2.0, obj_names[4]: 3.0},
        #     "DM3": {obj_names[0]: -2.5, obj_names[1]: -3.5, obj_names[2]: -2.0, obj_names[3]: 0.5, obj_names[4]: 5.0},
        # }
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
        "results_history": []
    })
    return MAX_ITERS, fractions, get_state, river_problem, set_state


@app.cell
def _(MAX_ITERS, cluster_points, favorite_method, get_state, river_problem):
    state = get_state()
    iter_idx = state["iter_idx"]
    current_options = state["current_options"]
    results_history = state["results_history"]

    if iter_idx < MAX_ITERS:
        # Evaluate Points & Generate Candidates
        fav_results = favorite_method(
            problem=river_problem,
            options=current_options,
            results_list=results_history
        )

        # Generate the geometry for the visualizations
        pts_mat, cents_mat, labels = cluster_points(fav_results)
    else:
        # TODO: here handle final solution voting, for now, lets just set fav_results to None so figures dont get updated.
        fav_results = None
    return (
        cents_mat,
        current_options,
        fav_results,
        iter_idx,
        labels,
        pts_mat,
        results_history,
    )


@app.cell(hide_code=True)
def _(
    cents_mat,
    current_options,
    fav_results,
    iter_idx,
    labels,
    mo,
    pts_mat,
    visualize_pcp_clusters,
):
    if fav_results is not None:
        # Generate the Plotly figure
        plot = visualize_pcp_clusters(
            options=current_options.GPRMoptions,
            points_arr=pts_mat,
            centers_arr=cents_mat,
            labels=labels,
            n_predetermined=len(fav_results.fair_solutions),
            iter_n=iter_idx + 1,
            current_mps=fav_results.FavOptions.original_most_preferred_solutions
        )
        # Wrap it in Marimo's native Plotly UI handler
        output = mo.ui.plotly(plot)
    else:
        # Render Markdown if finished
        output = mo.md("# 🎉 Optimization Finished!")

    # This MUST be the last line of the cell, completely unindented!
    output
    return


@app.cell(hide_code=True)
def _(fav_results, iter_idx, mo):
    if fav_results is not None:
        n_candidates = len(fav_results.fair_solutions)

        dropdown_options = {f"Candidate {i}": i for i in range(n_candidates)}

        dm1_vote = mo.ui.dropdown(options=dropdown_options, value="Candidate 0", label="DM1 Vote")
        dm2_vote = mo.ui.dropdown(options=dropdown_options, value="Candidate 0", label="DM2 Vote")
        dm3_vote = mo.ui.dropdown(options=dropdown_options, value="Candidate 0", label="DM3 Vote")

        # 1. Group the interactive inputs into a UI array
        vote_inputs = mo.ui.array([dm1_vote, dm2_vote, dm3_vote])

        # 2. Bind the form specifically to the inputs (NOT the markdown/layout)
        vote_form = mo.ui.form(
            element=vote_inputs,
            submit_button_label="Submit Votes & Expand Hull"
        )

        # 3. Construct the visual layout by stacking the text and the form
        ui_layout = mo.vstack([
            mo.md(f"### Place Votes for Iteration {iter_idx + 1}"),
            vote_form
        ])

    else:
        vote_form = None
        ui_layout = None

    # Output the visual layout at the bottom of the cell to render it
    print("success until here")
    ui_layout
    return (vote_form,)


@app.cell
def _(
    current_options,
    fav_results,
    fractions,
    generate_next_iteration_mps,
    iter_idx,
    labels,
    majority_rule,
    results_history,
    set_state,
    vote_form,
):
    if vote_form is not None and vote_form.value is not None:

        # 1. Gather votes and determine winner
        votes = {
            "DM1": vote_form.value[0], 
            "DM2": vote_form.value[1], 
            "DM3": vote_form.value[2]
        }

        winning_idx = majority_rule(votes)

        # 2. Expand the hull dynamically
        next_mps = generate_next_iteration_mps(
            fav_results=fav_results,
            cluster_labels=labels,
            winning_idx=winning_idx,
            fraction_to_keep=fractions[iter_idx]
        )

        # 3. Queue the options for the next loop
        new_options = current_options.model_copy(deep=True)
        new_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
        new_options.GPRMoptions.method_options.version = "convex_hull"
        new_options.zoom_options.num_steps_remaining = max(1, 4 - iter_idx - 1)
        new_options.votes = votes

        # 4. Update Marimo state (this loops execution back to Cell 3!)
        set_state({
            "iter_idx": iter_idx + 1,
            "current_options": new_options,
            "results_history": results_history + [fav_results]
        })
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Final solution voting
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
