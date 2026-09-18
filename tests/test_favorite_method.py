"""Tests related to the Favorite method."""

import copy
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
    ZoomOptions,
    adapt_all_dm_preferences,
    calculate_dm_utility,
    calculate_fraction_to_keep,
    check_adjacency,
    cluster_points,
    favorite_method,
    find_group_solutions,
    get_mm_candidate_index,
    get_tied_candidates,
    handle_ties,
    hausdorff_candidates,
    minimum_adjustment_mps,
    random_tie_breaker,
    recluster_for_tie_breaker,
    select_final_candidates,
    tie_breaker_avgproj,
)
from desdeo.gdm.voting_rules import borda_rule, calculate_borda_scores, plurality_rule
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
    """Returns dummy most preferred solutions for 4 decision makers."""
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
        fairness_criterion="mm",
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


def test_get_tied_candidates():
    """Test get_tied_candidates identifies all tied winners."""
    votes_single = {"dm1": 0, "dm2": 0, "dm3": 1}
    assert get_tied_candidates(votes_single) == [0]

    votes_tie = {"dm1": 0, "dm2": 1, "dm3": 2}
    assert sorted(get_tied_candidates(votes_tie)) == [0, 1, 2]

    votes_empty = {}
    assert get_tied_candidates(votes_empty) == []


def test_random_tie_breaker():
    """Test random_tie_breaker picks one candidate from tied indices."""
    candidates = [
        FairSolution(objective_values={"f1": 1.0}, fairness_criterion="c0", fairness_value=0.0),
        FairSolution(objective_values={"f1": 2.0}, fairness_criterion="c1", fairness_value=0.0),
        FairSolution(objective_values={"f1": 3.0}, fairness_criterion="c2", fairness_value=0.0),
    ]
    tied = [0, 2]
    winner_sol, winner_idx = random_tie_breaker(tied, candidates)
    assert winner_idx in [0, 2]
    assert winner_sol == candidates[winner_idx]


def test_check_adjacency():
    """Test check_adjacency correctly evaluates proximity between clusters."""
    pts = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.15, 0.15],
            [5.0, 5.0],
            [5.1, 5.1],
        ]
    )
    labels = np.array([0, 0, 1, 2, 2])
    assert check_adjacency(pts, labels, 0, 1) is True
    assert check_adjacency(pts, labels, 0, 2) is False
    assert check_adjacency(pts, labels, 0, 9) is False


def test_recluster_tie_breaker_override():
    """Test that recluster_for_tie_breaker updates candidate seeds and reclusters evaluated points."""
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
        fairness_criterion="mm",
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
    """Test that a simulated tie triggers the handle_ties mechanism and requests a revote."""
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
    assert "tied_candidate_indices" in res2.tie_state
    assert 0 in res2.tie_state["tied_candidate_indices"] and 1 in res2.tie_state["tied_candidate_indices"]


@patch("desdeo.gdm.favorite_method.add_asf_diff")
@patch("desdeo.gdm.favorite_method.guess_best_solver")
def test_tie_breaker_avgproj(mock_guess, mock_add_asf, dummy_problem):
    """Test tie-breaker average projection calculation and solver pipeline routing."""
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
        FairSolution(
            objective_values={"f_1": 0.0, "f_2": 0.9, "f_3": 0.8},
            fairness_criterion="mm",
            fairness_value=0.0,
        ),
        FairSolution(
            objective_values={"f_1": 0.9, "f_2": 0.1, "f_3": 0.1},
            fairness_criterion="mm",
            fairness_value=0.0,
        ),
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


def test_select_final_candidates_duplicate_merged(dummy_problem, dummy_evaluated_points, base_options):
    """Tests that when winning candidate is identical to group-fair solution, they merge into winner_and_mm."""
    mock_gprm = GPRMResults(
        raw_results=IPR_Results(evaluated_points=dummy_evaluated_points), solutions=None, outputs=pl.DataFrame()
    )

    targets_df = pl.DataFrame([p.targets for p in dummy_evaluated_points])
    outputs_df = pl.DataFrame([p.objectives for p in dummy_evaluated_points])
    fair_list = find_group_solutions(
        problem=dummy_problem,
        solutions=outputs_df,
        targets=targets_df,
        most_preferred_solutions=base_options.original_most_preferred_solutions,
        fairness_criterion="mm",
    )
    fair_pt = fair_list[0]

    candidates = [
        FairSolution(
            objective_values=fair_pt.objective_values,
            fairness_criterion="last_winner",
            fairness_value=fair_pt.fairness_value,
        ),
        FairSolution(
            objective_values=dummy_evaluated_points[9].objectives,
            fairness_criterion="nash",
            fairness_value=0,
        ),
    ]

    mock_fav_results = FavResults(
        FavOptions=base_options, GPRMResults=mock_gprm, fair_solutions=candidates, status="success", tie_state=None
    )
    labels = np.zeros(10, dtype=int)

    final_sols = select_final_candidates(
        problem=dummy_problem, fav_results=mock_fav_results, cluster_labels=labels, winning_idx=0, n_candidates=5
    )

    assert len(final_sols) == 5
    assert final_sols[0].fairness_criterion == "winner_and_mm"
    assert final_sols[1].fairness_criterion == "final_hausdorff"
    assert final_sols[2].fairness_criterion == "final_hausdorff"
    assert final_sols[3].fairness_criterion == "final_hausdorff"
    assert final_sols[4].fairness_criterion == "final_hausdorff"


def test_favorite_method_winner_duplicate_merging(dummy_problem, base_options):
    """Test that if previous winner matches newly computed fair solution, they merge into winner_and_mm."""
    winner_sol = FairSolution(
        objective_values={"f_1": 0.2, "f_2": 0.8, "f_3": 0.1},
        fairness_criterion="mm",
        fairness_value=0.5,
        variable_values={"x_1": 1.0, "x_2": 2.0},
    )
    initial_results = FavResults(
        FavOptions=base_options,
        GPRMResults=GPRMResults(raw_results=IPR_Results(evaluated_points=[]), solutions=None, outputs=pl.DataFrame()),
        fair_solutions=[winner_sol],
        status="success",
    )

    # Next iteration: voting for candidate 0 (the winner)
    iter2_options = base_options.model_copy(deep=True)
    iter2_options.votes = {"DM1": 0, "DM2": 0, "DM3": 0}
    iter2_options.total_n_of_candidates = 5

    # Mock new fair solution to be identical to winner_sol
    identical_fair_sol = FairSolution(
        objective_values={"f_1": 0.2, "f_2": 0.8, "f_3": 0.1},
        fairness_criterion="mm",
        fairness_value=0.45,
        variable_values={"x_1": 1.0, "x_2": 2.0},
    )

    with (
        patch("desdeo.gdm.favorite_method.get_representative_set") as mock_get_rep,
        patch("desdeo.gdm.favorite_method.find_group_solutions") as mock_find_group,
        patch("desdeo.gdm.favorite_method.hausdorff_candidates") as mock_hausdorff,
    ):
        mock_get_rep.return_value = GPRMResults(
            raw_results=IPR_Results(evaluated_points=[]), solutions=None, outputs=pl.DataFrame()
        )
        mock_find_group.return_value = [identical_fair_sol]
        mock_hausdorff.side_effect = lambda all_pts, fairs, n_missing: fairs + [
            FairSolution(
                objective_values={"f_1": 0.5, "f_2": 0.5, "f_3": 0.5},
                fairness_criterion="avg_hausdorff",
                fairness_value=1e6,
            )
            for _ in range(n_missing)
        ]

        res = favorite_method(dummy_problem, iter2_options, results_list=[initial_results])

        assert len(res.fair_solutions) == 5
        assert res.fair_solutions[0].fairness_criterion == "winner_and_mm"
        assert res.fair_solutions[0].fairness_value == 0.45
        assert res.fair_solutions[1].fairness_criterion == "avg_hausdorff"


def test_plurality_rule_unique_and_tied():
    """Test plurality_rule returns single winner for strict plurality and all tied candidates on tie."""
    votes_unique = {"dm1": 0, "dm2": 0, "dm3": 1, "dm4": 2}
    top = plurality_rule(votes_unique)
    assert top == [0]

    votes_tie = {"dm1": 0, "dm2": 1, "dm3": 1, "dm4": 0}
    top_tie = plurality_rule(votes_tie)
    assert sorted(top_tie) == [0, 1]


def test_calculate_borda_scores_and_rule():
    """Test weighted Borda scores (2*V1 + 1*V2) and borda_rule."""
    r1 = {"dm1": 0, "dm2": 1, "dm3": 1}  # cand 0 has 1 vote (2 pts), cand 1 has 2 votes (4 pts)
    r2 = {"dm1": 2, "dm2": 0, "dm3": 2}  # cand 0 has 1 vote (1 pt), cand 2 has 2 votes (2 pts)
    # Total scores: cand 0 = 3, cand 1 = 4, cand 2 = 2
    scores = calculate_borda_scores(r1, r2, n_candidates=3)
    assert scores == {0: 3, 1: 4, 2: 2}

    winners = borda_rule(r1, r2, n_candidates=3)
    assert winners == [1]

    # Test Borda tie
    r2_tie = {"dm1": 0, "dm2": 2, "dm3": 0}  # cand 0 gets +2 pts -> total 4; cand 1 gets +0 -> total 4
    scores_tie = calculate_borda_scores(r1, r2_tie, n_candidates=3)
    assert scores_tie[0] == 4
    assert scores_tie[1] == 4
    tied_winners = borda_rule(r1, r2_tie, n_candidates=3)
    assert sorted(tied_winners) == [0, 1]


def test_get_mm_candidate_index():
    """Test get_mm_candidate_index finds the group fair candidate."""
    cands = [
        FairSolution(objective_values={"f": 1.0}, fairness_criterion="last_winner", fairness_value=0.0),
        FairSolution(objective_values={"f": 2.0}, fairness_criterion="mm", fairness_value=0.0),
        FairSolution(objective_values={"f": 3.0}, fairness_criterion="avg_hausdorff", fairness_value=0.0),
    ]
    assert get_mm_candidate_index(cands) == 1

    cands_merged = [
        FairSolution(objective_values={"f": 1.0}, fairness_criterion="winner_and_mm", fairness_value=0.0),
        FairSolution(objective_values={"f": 2.0}, fairness_criterion="avg_hausdorff", fairness_value=0.0),
    ]
    assert get_mm_candidate_index(cands_merged) == 0


def test_handle_ties_global_forced_borda_and_mm_fallback(dummy_problem, base_options):
    """Test handle_ties transitions through Round 1 tie, Borda resolution, and MM fallback."""
    cands = [
        FairSolution(objective_values={"f_1": 0.0, "f_2": 1.0, "f_3": 0.0}, fairness_criterion="mm", fairness_value=0),
        FairSolution(
            objective_values={"f_1": 1.0, "f_2": 0.0, "f_3": 0.0}, fairness_criterion="avg_hausdorff", fairness_value=0
        ),
        FairSolution(
            objective_values={"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}, fairness_criterion="avg_hausdorff", fairness_value=0
        ),
    ]
    mock_gprm = GPRMResults(
        raw_results=IPR_Results(
            evaluated_points=[
                _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.0, "f_2": 1.0, "f_3": 0.0}),
                _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 1.0, "f_2": 0.0, "f_3": 0.0}),
            ]
        ),
        solutions=None,
        outputs=pl.DataFrame(),
    )
    res_prev = FavResults(FavOptions=base_options, GPRMResults=mock_gprm, fair_solutions=cands, status="success")

    # 1. Round 1 3-way tie: should initiate global_forced_borda revote
    votes_r1 = {"DM1": 0, "DM2": 1, "DM3": 2}
    winner, tie_state = handle_ties(
        problem=dummy_problem,
        votes=votes_r1,
        candidates=cands,
        fav_results_previous=res_prev,
        tie_state=None,
    )
    assert winner is None
    assert tie_state is not None
    assert tie_state["strategy"] == "global_forced_borda"
    assert tie_state["round_1_votes"] == votes_r1
    assert tie_state["eligible_candidates"] == [0, 1, 2]

    # 2. Round 2 revote: DM1->1, DM2->2, DM3->1
    # Borda: cand 0 = 2, cand 1 = 2 + 2 = 4, cand 2 = 2 + 1 = 3 -> cand 1 wins!
    votes_r2 = {"DM1": 1, "DM2": 2, "DM3": 1}
    winner_r2, tie_state_r2 = handle_ties(
        problem=dummy_problem,
        votes=votes_r2,
        candidates=cands,
        fav_results_previous=res_prev,
        tie_state=tie_state,
    )
    assert tie_state_r2 is None
    assert winner_r2 == cands[1]

    # 3. Round 2 revote resulting in a 3-way Borda tie:
    # All 3 candidates get 3 Borda points -> breaks tie by evaluating candidate fairness
    votes_r2_tie = {"DM1": 1, "DM2": 2, "DM3": 0}
    winner_tie, tie_state_tie = handle_ties(
        problem=dummy_problem,
        votes=votes_r2_tie,
        candidates=cands,
        fav_results_previous=res_prev,
        tie_state=tie_state,
    )
    assert tie_state_tie is None
    # Evaluates fairness among tied candidates: cand 2 has the highest group fairness score
    assert winner_tie == cands[2]


def test_borda_tie_between_non_mm_candidates_picks_fairer_leader(dummy_problem, base_options):
    """Test that when top Borda candidates are tied, candidate 0 with fewer points cannot win."""
    cands = [
        FairSolution(objective_values={"f_1": 0.0, "f_2": 1.0, "f_3": 0.0}, fairness_criterion="mm", fairness_value=0),
        FairSolution(
            objective_values={"f_1": 0.3, "f_2": 0.3, "f_3": 0.3}, fairness_criterion="avg_hausdorff", fairness_value=0
        ),
        FairSolution(
            objective_values={"f_1": 0.4, "f_2": 0.4, "f_3": 0.4}, fairness_criterion="avg_hausdorff", fairness_value=0
        ),
        FairSolution(
            objective_values={"f_1": 0.9, "f_2": 0.9, "f_3": 0.9}, fairness_criterion="avg_hausdorff", fairness_value=0
        ),
    ]
    res_prev = FavResults(
        FavOptions=base_options,
        GPRMResults=GPRMResults(
            raw_results=IPR_Results(evaluated_points=[]),
            solutions=None,
            outputs=pl.DataFrame(),
        ),
        fair_solutions=cands,
        status="success",
    )

    # 4-way tie in Round 1: each candidate gets 2 Borda points
    votes_r1 = {"DM1": 0, "DM2": 1, "DM3": 2, "DM4": 3}
    _, tie_state = handle_ties(
        problem=dummy_problem,
        votes=votes_r1,
        candidates=cands,
        fav_results_previous=res_prev,
        tie_state=None,
    )
    assert tie_state is not None

    # Round 2 revote:
    # DM1 -> 1, DM2 -> 2, DM3 -> 1, DM4 -> 2
    # Borda: Cand 0 has 2 pts, Cand 1 has 4 pts, Cand 2 has 4 pts, Cand 3 has 2 pts.
    votes_r2 = {"DM1": 1, "DM2": 2, "DM3": 1, "DM4": 2}
    winner, tie_state_r2 = handle_ties(
        problem=dummy_problem,
        votes=votes_r2,
        candidates=cands,
        fav_results_previous=res_prev,
        tie_state=tie_state,
    )
    assert tie_state_r2 is None
    # Cand 0 has only 2 Borda points and MUST NOT WIN. Winner must be cands[1] (fairer than cands[2]).
    assert winner != cands[0]
    assert winner == cands[1]


def test_round_2_revotes_private_no_preference_leak(dummy_problem, base_options):
    """Test that Round 2 concession votes do not leak into or alter DM preferred solutions."""
    initial_mps = {
        "DM1": {"f_1": 0.0, "f_2": 0.9, "f_3": 0.8},
        "DM2": {"f_1": 0.9, "f_2": 0.0, "f_3": 0.8},
        "DM3": {"f_1": 0.8, "f_2": 0.9, "f_3": 0.0},
        "DM4": {"f_1": 0.3, "f_2": 0.3, "f_3": 0.3},
    }
    base_options.original_most_preferred_solutions = copy.deepcopy(initial_mps)
    base_options.current_most_preferred_solutions = copy.deepcopy(initial_mps)

    mock_candidates = [
        FairSolution(objective_values={"f_1": 0.0, "f_2": 1.0, "f_3": 0.0}, fairness_criterion="mm", fairness_value=0),
        FairSolution(
            objective_values={"f_1": 1.0, "f_2": 0.0, "f_3": 0.0}, fairness_criterion="nash", fairness_value=0
        ),
    ]
    mock_gprm = GPRMResults(
        raw_results=IPR_Results(
            evaluated_points=[
                _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.0, "f_2": 1.0, "f_3": 0.0}),
                _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 1.0, "f_2": 0.0, "f_3": 0.0}),
            ]
        ),
        solutions=None,
        outputs=pl.DataFrame(),
    )
    res1 = FavResults(FavOptions=base_options, GPRMResults=mock_gprm, fair_solutions=mock_candidates, status="success")

    # Round 1 options: tie between 0 and 1
    options_r1 = base_options.model_copy(deep=True)
    options_r1.votes = {"DM1": 0, "DM2": 0, "DM3": 1, "DM4": 1}

    with patch("desdeo.gdm.favorite_method.get_representative_set_IPR") as mock_ipr:
        mock_ipr.return_value = mock_gprm
        res_r1 = favorite_method(dummy_problem, options_r1, results_list=[res1])

    assert res_r1.status == "revote_pending"
    # DMs preferences were adapted on Round 1 votes
    mps_after_r1 = copy.deepcopy(res_r1.FavOptions.current_most_preferred_solutions)
    assert res_r1.FavOptions.preferences_already_adapted is True

    # Now cast Round 2 revotes where DMs vote for concession candidates
    options_r2 = res_r1.FavOptions.model_copy(deep=True)
    options_r2.tie_state = res_r1.tie_state
    options_r2.votes = {"DM1": 1, "DM2": 1, "DM3": 0, "DM4": 0}  # Concession votes

    with (
        patch("desdeo.gdm.favorite_method.get_representative_set") as mock_rep,
        patch("desdeo.gdm.favorite_method.find_group_solutions") as mock_fairs,
    ):
        mock_rep.return_value = mock_gprm
        mock_fairs.return_value = [mock_candidates[0]]
        res_r2 = favorite_method(dummy_problem, options_r2, results_list=[res1])

    assert res_r2.status == "success"
    # Crucial assertion: MPS must NOT be altered by Round 2 concession votes!
    assert res_r2.FavOptions.current_most_preferred_solutions == mps_after_r1


def test_borda_scoring_custom_weights():
    """Test calculate_borda_scores and borda_rule with customizable weights."""
    r1 = {"DM1": 0, "DM2": 0, "DM3": 1}
    r2 = {"DM1": 1, "DM2": 1, "DM3": 0}

    # Default weights (2, 1):
    # Cand 0: 2*2 + 1*1 = 5
    # Cand 1: 1*2 + 2*1 = 4
    scores_default = calculate_borda_scores(r1, r2, 2, weights=(2, 1))
    assert scores_default[0] == 5
    assert scores_default[1] == 4
    assert borda_rule(r1, r2, 2, weights=(2, 1)) == [0]

    # Custom weights (1, 3): (higher weight on concession revote)
    # Cand 0: 2*1 + 1*3 = 5
    # Cand 1: 1*1 + 2*3 = 7
    scores_custom = calculate_borda_scores(r1, r2, 2, weights=(1, 3))
    assert scores_custom[0] == 5
    assert scores_custom[1] == 7
    assert borda_rule(r1, r2, 2, weights=(1, 3)) == [1]


def test_voting_rule_majority_vs_plurality(dummy_problem, base_options):
    """Test that majority rule requires >50% votes, while plurality only requires max votes."""
    mock_candidates = [
        FairSolution(objective_values={"f_1": 0.0, "f_2": 1.0, "f_3": 0.0}, fairness_criterion="mm", fairness_value=0),
        FairSolution(objective_values={"f_1": 1.0, "f_2": 0.0, "f_3": 0.0}, fairness_criterion="nash", fairness_value=0),
        FairSolution(objective_values={"f_1": 0.5, "f_2": 0.5, "f_3": 0.0}, fairness_criterion="avg_hausdorff", fairness_value=0),
    ]
    mock_gprm = GPRMResults(
        raw_results=IPR_Results(
            evaluated_points=[
                _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.0, "f_2": 1.0, "f_3": 0.0}),
                _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 1.0, "f_2": 0.0, "f_3": 0.0}),
                _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.5, "f_2": 0.5, "f_3": 0.0}),
            ]
        ),
        solutions=None,
        outputs=pl.DataFrame(),
    )
    res_prev = FavResults(FavOptions=base_options, GPRMResults=mock_gprm, fair_solutions=mock_candidates, status="success")

    # 4 voters: Cand 0 has 2 votes (50%), Cand 1 has 1 vote (25%), Cand 2 has 1 vote (25%)
    votes_no_maj = {"DM1": 0, "DM2": 0, "DM3": 1, "DM4": 2}

    # Under Plurality Rule: Candidate 0 wins immediately because 2 > 1
    options_plurality = base_options.model_copy(deep=True)
    options_plurality.voting_rule = "plurality"
    options_plurality.votes = votes_no_maj
    with (
        patch("desdeo.gdm.favorite_method.get_representative_set") as mock_rep,
        patch("desdeo.gdm.favorite_method.find_group_solutions") as mock_fairs,
    ):
        mock_rep.return_value = mock_gprm
        mock_fairs.return_value = [mock_candidates[0]]
        res_plurality = favorite_method(dummy_problem, options_plurality, results_list=[res_prev])

    assert res_plurality.status == "success"
    # Winner was Candidate 0, so fair_solutions starts with Candidate 0
    assert res_plurality.fair_solutions[0] == mock_candidates[0]

    # Under Majority Rule: Candidate 0 has 2/4 = 50% (not >50%), so no majority -> revote_pending
    options_majority = base_options.model_copy(deep=True)
    options_majority.voting_rule = "majority"
    options_majority.votes = votes_no_maj
    with patch("desdeo.gdm.favorite_method.get_representative_set_IPR") as mock_ipr:
        mock_ipr.return_value = mock_gprm
        res_majority = favorite_method(dummy_problem, options_majority, results_list=[res_prev])

    assert res_majority.status == "revote_pending"


def test_fav_options_fairness_criterion(base_options):
    """Test that fairness_criterion is the single field and backward-compat validator works."""
    # 1. model_copy with fairness_criterion works directly
    opt1 = base_options.model_copy(update={"fairness_criterion": "nash"})
    assert opt1.fairness_criterion == "nash"

    # 2. Backward-compat: old candidate_generation_options key in dict input is promoted
    raw = base_options.model_dump()
    del raw["fairness_criterion"]
    raw["candidate_generation_options"] = "utilitarian"
    opt2 = FavOptions(**raw)
    assert opt2.fairness_criterion == "utilitarian"
