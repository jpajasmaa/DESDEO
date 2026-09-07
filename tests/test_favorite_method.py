"""Tests related to the Favorite method."""

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from desdeo.gdm.favorite_method import (
    FairSolution,
    FavOptions,
    FavResults,
    GPRMOptions,
    GPRMResults,
    IPR_Options,
    IPR_Results,
    ProblemWrapper,
    ZoomOptions,
    calculate_fraction_to_keep,
    cluster_points,
    favorite_method,
    find_group_solutions,
    hausdorff_candidates,
    recluster_for_tie_breaker,
    select_final_candidates,
    tie_breaker_avgproj,
    adapt_all_dm_preferences,
    calculate_dm_utility,
    minimum_adjustment_mps,
)
from desdeo.problem.testproblems.dtlz_problems import dtlz2
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint

# ==========================================
# FIXTURES
# ==========================================


@pytest.fixture
def dummy_problem():
    """Returns a simple 3-objective DTLZ2 problem."""
    return dtlz2(n_variables=8, n_objectives=3)


@pytest.fixture
def dummy_mps():
    return {
        "DM1": {"f_1": 0.0, "f_2": 0.9, "f_3": 0.8},
        "DM2": {"f_1": 0.9, "f_2": 0.0, "f_3": 0.8},
        "DM3": {"f_1": 0.8, "f_2": 0.9, "f_3": 0.0},
        "DM4": {"f_1": 0.3, "f_2": 0.3, "f_3": 0.3},
    }


@pytest.fixture
def base_options(dummy_mps):
    """Creates a valid, fast FavOptions object for testing."""
    ipr_options = IPR_Options(
        most_preferred_solutions=dummy_mps,
        num_initial_reference_points=50,  # Keep small for tests
        version="convex_hull",
    )
    gprm_options = GPRMOptions(
        method_options=ipr_options,
        fake_ideal={"f_1": 0.0, "f_2": 0.0, "f_3": 0.0},
        fake_nadir={"f_1": 1.0, "f_2": 1.0, "f_3": 1.0},
        num_points_to_evaluate=5,  # Keep small for tests
    )
    return FavOptions(
        GPRMoptions=gprm_options,
        candidate_generation_options="mm",
        zoom_options=ZoomOptions(num_steps_remaining=4),
        original_most_preferred_solutions=dummy_mps,
        total_n_of_candidates=5,
    )


@pytest.fixture
def dummy_evaluated_points():
    """Generates a list of dummy evaluated points for clustering and hausdorff tests."""
    points = []
    for i in range(10):
        points.append(
            _EvaluatedPoint(
                reference_point={"f_1": 0.5, "f_2": 0.5, "f_3": 0.5},
                targets={"f_1": 0.5, "f_2": 0.5, "f_3": 0.5},
                objectives={"f_1": i * 0.1, "f_2": 1.0 - i * 0.1, "f_3": 0.5},
            )
        )
    return points


# ==========================================
# COMPONENT TESTS (Logic)
# ==========================================


def test_shrinking():
    """Tests that the shrinking fraction functions correctly."""
    max_iters = 5
    num_obj = 3

    # Iteration 0: (4 / 5) ^ 2 = 16 / 25 = 0.64
    frac_0 = calculate_fraction_to_keep(current_iter=0, max_iters=max_iters, num_objectives=num_obj)
    assert frac_0 == pytest.approx(0.64)

    # Iteration 1: (3 / 4) ^ 2 = 9 / 16 = 0.5625
    frac_1 = calculate_fraction_to_keep(current_iter=1, max_iters=max_iters, num_objectives=num_obj)
    assert frac_1 == pytest.approx(0.5625)

    # Final Iteration (4 out of 5): Should be exactly 0.0
    frac_final = calculate_fraction_to_keep(current_iter=4, max_iters=max_iters, num_objectives=num_obj)
    assert frac_final == 0.0


def test_recluster_tie_breaker_override():
    """TODO: see if this test makes sense anymore"""
    # Mock evaluated points
    mock_points = [
        _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.1, "f_2": 0.9, "f_3": 0.1}),
        _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.9, "f_2": 0.1, "f_3": 0.1}),
        _EvaluatedPoint(
            reference_point={}, targets={}, objectives={"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}
        ),  # The center point
        _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.2, "f_2": 0.8, "f_3": 0.1}),
    ]

    # Mock the existing deadlocked candidates
    existing_cands = [
        FairSolution(
            objective_values={"f_1": 0.0, "f_2": 1.0, "f_3": 0.0}, fairness_criterion="core", fairness_value=0
        ),
        FairSolution(objective_values={"f_1": 1.0, "f_2": 0.0, "f_3": 0.0}, fairness_criterion="mm", fairness_value=0),
        FairSolution(
            objective_values={"f_1": 0.2, "f_2": 0.2, "f_3": 0.8}, fairness_criterion="hausdorff_1", fairness_value=0
        ),
    ]

    # The New Tie-Breaker Compromise (Sits right in the middle)
    compromise = FairSolution(
        objective_values={"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}, fairness_criterion="tie_breaker", fairness_value=0.0
    )

    # EXECUTE
    updated_cands, new_labels, winning_idx = recluster_for_tie_breaker(
        all_points=mock_points, existing_candidates=existing_cands, compromise_solution=compromise
    )

    # ASSERTIONS
    assert len(updated_cands) == 2, "Override failed: Pool should only contain Compromise and MaxMin (2 items)."
    assert winning_idx == 0, "Winning index must be forced to 0."
    assert updated_cands[0].fairness_criterion == "tie_breaker", "Compromise must be the primary seed."
    assert updated_cands[1].fairness_criterion == "mm", "MaxMin fair solution must be retained as the alternative."
    assert len(new_labels) == 4, "Labels array should map to all 4 evaluated points."
    assert new_labels[2] == 0, "The center point was not correctly assigned to the compromise cluster!"


def test_hausdorff_candidates(dummy_evaluated_points):
    """Tests if Hausdorff selection correctly expands the candidate list."""
    # Seed with one fair solution
    seed_solution = FairSolution(
        objective_values={"f_1": 0.1, "f_2": 0.9, "f_3": 0.5}, fairness_criterion="mm", fairness_value=0.1
    )

    n_missing = 2
    results = hausdorff_candidates(dummy_evaluated_points, [seed_solution], n_missing)

    # Check lengths
    assert len(results) == 3, "Should return the 1 seed + 2 new candidates"
    assert all(isinstance(sol, FairSolution) for sol in results)
    assert results[0].fairness_criterion == "mm"
    assert results[1].fairness_criterion == "avg_hausdorff"


def test_cluster_points(dummy_evaluated_points, base_options):
    """Tests if Voronoi partitioning returns correctly shaped arrays."""
    mock_gprm = GPRMResults(
        raw_results=IPR_Results(evaluated_points=dummy_evaluated_points), solutions=None, outputs=pl.DataFrame()
    )

    candidates = [
        FairSolution(objective_values={"f_1": 0.1, "f_2": 0.9, "f_3": 0.5}, fairness_criterion="mm", fairness_value=0),
        FairSolution(
            objective_values={"f_1": 0.9, "f_2": 0.1, "f_3": 0.5}, fairness_criterion="nash", fairness_value=0
        ),
    ]

    mock_fav_results = FavResults(
        FavOptions=base_options, GPRMResults=mock_gprm, fair_solutions=candidates, status="success", tie_state=None
    )

    pts_arr, centers_arr, labels = cluster_points(mock_fav_results)

    assert pts_arr.shape == (10, 3), "Points array should be (n_points, k_objectives)"
    assert centers_arr.shape == (2, 3), "Centers array should be (n_candidates, k_objectives)"
    assert labels.shape == (10,), "Labels array should have one entry per point"
    assert set(labels).issubset({0, 1}), "Labels should only map to the 2 candidate indices"


def test_select_final_candidates(dummy_problem, dummy_evaluated_points, base_options):
    """Tests the final phase voting logic."""
    mock_gprm = GPRMResults(
        raw_results=IPR_Results(evaluated_points=dummy_evaluated_points), solutions=None, outputs=pl.DataFrame()
    )

    candidates = [
        FairSolution(objective_values=dummy_evaluated_points[0].objectives, fairness_criterion="mm", fairness_value=0),
        FairSolution(
            objective_values=dummy_evaluated_points[9].objectives, fairness_criterion="nash", fairness_value=0
        ),
    ]

    mock_fav_results = FavResults(
        FavOptions=base_options, GPRMResults=mock_gprm, fair_solutions=candidates, status="success", tie_state=None
    )

    labels = np.zeros(10, dtype=int)

    final_sols = select_final_candidates(
        problem=dummy_problem, fav_results=mock_fav_results, cluster_labels=labels, winning_idx=0, n_candidates=3
    )

    assert len(final_sols) == 3
    assert final_sols[0].fairness_criterion == "last_winner"
    assert final_sols[1].fairness_criterion == "final_mm"
    assert final_sols[2].fairness_criterion == "final_hausdorff"
    assert final_sols[0].objective_values == dummy_evaluated_points[0].objectives


# ==========================================
# DATA FLOW & PIPELINE TESTS
# ==========================================


@patch("desdeo.gdm.favorite_method.guess_best_solver")
def test_problem_wrapper_data_flow(mock_guess, dummy_problem):
    """Tests if the ProblemWrapper correctly formats the solver inputs/outputs."""
    mock_solver_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.optimal_objectives = {"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}
    mock_solver_instance.solve.return_value = mock_result
    mock_guess.return_value = MagicMock(return_value=mock_solver_instance)

    fake_ideal = {"f_1": 0.0, "f_2": 0.0, "f_3": 0.0}
    fake_nadir = {"f_1": 1.0, "f_2": 1.0, "f_3": 1.0}

    wrapper = ProblemWrapper(dummy_problem, fake_ideal, fake_nadir)
    res = wrapper.solve([0.2, 0.2, 0.2])

    assert len(res) == 1
    assert isinstance(res[0], _EvaluatedPoint)
    assert res[0].targets == {"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}


@patch("desdeo.gdm.favorite_method.get_representative_set_IPR")
@patch("desdeo.gdm.favorite_method.find_group_solutions")
def test_favorite_method_first_iteration(mock_find_group, mock_get_ipr, dummy_problem, base_options):
    """Tests the main orchestrator for a first iteration."""
    mock_ipr_res = GPRMResults(
        raw_results=IPR_Results(evaluated_points=[]), solutions=pl.DataFrame(), outputs=pl.DataFrame()
    )
    mock_get_ipr.return_value = mock_ipr_res

    mock_fair_sol = FairSolution(
        objective_values={"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}, fairness_criterion="mm", fairness_value=0.0
    )
    mock_find_group.return_value = [mock_fair_sol]

    with patch("desdeo.gdm.favorite_method.hausdorff_candidates") as mock_hausdorff:
        mock_hausdorff.return_value = [mock_fair_sol, mock_fair_sol, mock_fair_sol]

        final_results = favorite_method(dummy_problem, base_options, results_list=[])

        assert isinstance(final_results, FavResults)
        assert final_results.status == "success", "Initial iteration should inherently succeed."
        assert final_results.tie_state is None
        assert len(final_results.fair_solutions) == 3
        assert final_results.FavOptions.votes is None


def test_find_group_solutions_data_flow(dummy_problem, dummy_mps):
    """Tests that find_group_solutions works properly."""
    targets_df = pl.DataFrame({"f_1": [0.1, 0.9], "f_2": [0.9, 0.1], "f_3": [0.5, 0.5]})
    solutions_df = pl.DataFrame({"f_1": [0.1, 0.9], "f_2": [0.9, 0.1], "f_3": [0.5, 0.5]})

    fair_sols = find_group_solutions(
        problem=dummy_problem,
        solutions=solutions_df,
        targets=targets_df,
        most_preferred_solutions=dummy_mps,
        fairness_criterion="mm",
    )

    assert len(fair_sols) == 1
    assert isinstance(fair_sols[0], FairSolution)
    assert fair_sols[0].fairness_criterion == "mm"


@pytest.mark.slow
def test_favorite_method_e2e_integration(dummy_problem, dummy_mps):
    """END-TO-END INTEGRATION TEST."""
    ipr_options = IPR_Options(
        most_preferred_solutions=dummy_mps, num_initial_reference_points=15, version="convex_hull"
    )
    gprm_options = GPRMOptions(
        method_options=ipr_options,
        fake_ideal={"f_1": 0.0, "f_2": 0.0, "f_3": 0.0},
        fake_nadir={"f_1": 1.0, "f_2": 1.0, "f_3": 1.0},
        num_points_to_evaluate=3,
    )
    options = FavOptions(
        GPRMoptions=gprm_options,
        candidate_generation_options="mm",
        zoom_options=ZoomOptions(num_steps_remaining=4),
        original_most_preferred_solutions=dummy_mps,
        total_n_of_candidates=3,
    )

    results = favorite_method(dummy_problem, options, results_list=[])

    assert isinstance(results, FavResults)
    assert results.status == "success"
    assert results.tie_state is None
    assert len(results.GPRMResults.raw_results.evaluated_points) == 3
    assert len(results.fair_solutions) == 3

    for sol in results.fair_solutions:
        for val in sol.objective_values.values():
            assert isinstance(val, float)


def test_favorite_method_tie_routing(dummy_problem, base_options):
    """Tests that a simulated tie triggers the handle_ties mechanism and correctly
    flags the FavResults object as needing a re-vote if clusters are disjoint.
    """
    # Mock the first iteration results to provide previous candidates
    mock_points = [
        _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.1, "f_2": 0.9, "f_3": 0.1}),
        _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.9, "f_2": 0.1, "f_3": 0.1}),
    ]
    mock_gprm = GPRMResults(
        raw_results=IPR_Results(evaluated_points=mock_points), solutions=None, outputs=pl.DataFrame()
    )
    mock_candidates = [
        FairSolution(objective_values={"f_1": 0.0, "f_2": 1.0, "f_3": 0.0}, fairness_criterion="mm", fairness_value=0),
        FairSolution(
            objective_values={"f_1": 1.0, "f_2": 0.0, "f_3": 0.0}, fairness_criterion="nash", fairness_value=0
        ),
    ]

    res1 = FavResults(FavOptions=base_options, GPRMResults=mock_gprm, fair_solutions=mock_candidates, status="success")

    # Setup iteration 2 options with a strict 2-2 tie
    options2 = base_options.model_copy(deep=True)
    options2.votes = {"DM1": 0, "DM2": 0, "DM3": 1, "DM4": 1}

    # Note: Because the mock evaluated points are at extreme ends, adjacency will be false, triggering a re-vote.
    with patch("desdeo.gdm.favorite_method.get_representative_set_IPR") as mock_ipr:
        # Mock IPR to bypass heavy calc since we just want to test routing
        mock_ipr.return_value = mock_gprm

        res2 = favorite_method(dummy_problem, options2, results_list=[res1])

    assert res2.status == "revote_pending", "Tie logic failed to halt progression!"
    assert res2.tie_state is not None, "UI State payload is missing!"
    assert "tied_indices" in res2.tie_state
    assert 0 in res2.tie_state["tied_indices"] and 1 in res2.tie_state["tied_indices"]


@patch("desdeo.gdm.favorite_method.add_asf_diff")
@patch("desdeo.gdm.favorite_method.guess_best_solver")
def test_tie_breaker_avgproj(mock_guess, mock_add_asf, dummy_problem):
    """Tests the tie-breaker functionality: verifying the average is calculated correctly
    and the solver pipeline is triggered and routed.
    """
    mock_solver_instance = MagicMock()
    mock_result = MagicMock()

    mock_result.optimal_objectives = {"f_1": 2.5, "f_2": 2.5, "f_3": 2.5}
    mock_solver_instance.solve.return_value = mock_result
    mock_guess.return_value = MagicMock(return_value=mock_solver_instance)
    mock_add_asf.return_value = (MagicMock(), MagicMock())

    votes = {"DM1": 0, "DM2": 1}
    # imagine first candidades are adjacent
    candidates = [
        FairSolution(
            objective_values={"f_1": 0.0, "f_2": 2.0, "f_3": 4.0}, fairness_criterion="mm", fairness_value=0.0
        ),
        FairSolution(
            objective_values={"f_1": 3.0, "f_2": 1.0, "f_3": 5.0}, fairness_criterion="mm", fairness_value=0.0
        ),
        FairSolution(
            objective_values={"f_1": 6.0, "f_2": 6.0, "f_3": 0.0}, fairness_criterion="mm", fairness_value=0.0
        ),
    ]

    winning_sol = tie_breaker_avgproj(dummy_problem, votes, candidates)

    mock_add_asf.assert_called_once()
    passed_avg_point = mock_add_asf.call_args[0][2]
    print(passed_avg_point)

    assert passed_avg_point == {"f_1": 1.5, "f_2": 1.5, "f_3": 4.5}, "Calculated average is incorrect!"
    assert isinstance(winning_sol, FairSolution)
    assert winning_sol.objective_values == {"f_1": 2.5, "f_2": 2.5, "f_3": 2.5}
    assert winning_sol.fairness_criterion == "tie_breaker_average_projection"


def test_minimum_adjustment_mps_no_adjustment_when_top(dummy_problem):
    """Test that when a DM votes for their best candidate, no adjustment is made."""
    dm_mps = {"f_1": 0.0, "f_2": 0.9, "f_3": 0.8}
    best_cand = {"f_1": 0.0, "f_2": 0.9, "f_3": 0.8}
    other_cand = {"f_1": 0.8, "f_2": 0.2, "f_3": 0.2}

    new_mps, was_adjusted, lam = minimum_adjustment_mps(
        problem=dummy_problem,
        dm_mps=dm_mps,
        voted_candidate=best_cand,
        all_candidates=[best_cand, other_cand],
    )
    assert not was_adjusted
    assert lam == 0.0
    assert new_mps == dm_mps


def test_minimum_adjustment_mps_adjusted_when_suboptimal(dummy_problem):
    """Test that when a DM votes for a sub-optimal candidate, their MPS is shifted."""
    dm_mps = {"f_1": 0.0, "f_2": 0.9, "f_3": 0.8}
    top_cand = {"f_1": 0.0, "f_2": 0.9, "f_3": 0.8}
    voted_cand = {"f_1": 0.9, "f_2": 0.1, "f_3": 0.1}

    new_mps, was_adjusted, lam = minimum_adjustment_mps(
        problem=dummy_problem,
        dm_mps=dm_mps,
        voted_candidate=voted_cand,
        all_candidates=[top_cand, voted_cand],
    )
    assert was_adjusted
    assert 0.0 < lam <= 1.0

    # Under new_mps, voted_cand must have higher utility than top_cand
    u_voted = calculate_dm_utility(dummy_problem, new_mps, voted_cand)
    u_top = calculate_dm_utility(dummy_problem, new_mps, top_cand)
    assert u_voted >= u_top


def test_adapt_all_dm_preferences(dummy_problem):
    """Test adapting preferences across multiple DMs."""
    cands = [
        FairSolution(objective_values={"f_1": 0.0, "f_2": 0.9, "f_3": 0.8}, fairness_criterion="mm", fairness_value=0.0),
        FairSolution(objective_values={"f_1": 0.9, "f_2": 0.1, "f_3": 0.1}, fairness_criterion="mm", fairness_value=0.0),
    ]
    current_mps = {
        "DM1": {"f_1": 0.0, "f_2": 0.9, "f_3": 0.8},
        "DM2": {"f_1": 0.9, "f_2": 0.1, "f_3": 0.1},
    }
    # DM1 votes for candidate 1 (suboptimal for DM1)
    # DM2 votes for candidate 1 (optimal for DM2)
    votes = {"DM1": 1, "DM2": 1}

    updated_mps, summary = adapt_all_dm_preferences(
        problem=dummy_problem,
        current_mps=current_mps,
        candidates=cands,
        votes=votes,
    )
    assert summary["DM1"]["was_adjusted"] is True
    assert summary["DM2"]["was_adjusted"] is False
    assert updated_mps["DM2"] == current_mps["DM2"]
    assert updated_mps["DM1"] != current_mps["DM1"]

