"""Tests related to the Favorite method."""

import pytest
import numpy as np
import polars as pl
from unittest.mock import patch, MagicMock

from desdeo.problem.testproblems.dtlz2_problem import dtlz2
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint

from desdeo.gdm.favorite_method import (
    IPR_Options, GPRMOptions, ZoomOptions, FavOptions, FavResults, GPRMResults, IPR_Results, FairSolution,
    ProblemWrapper, find_group_solutions, hausdorff_candidates, cluster_points,
    generate_next_iteration_mps, select_final_candidates, favorite_method, tie_breaker_avgproj, calculate_fraction_to_keep, recluster_for_tie_breaker
)

# ==========================================
# 1. FIXTURES (Reusable Test Data)
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
    }

@pytest.fixture
def base_options(dummy_mps):
    """Creates a valid, fast FavOptions object for testing."""
    ipr_options = IPR_Options(
        most_preferred_solutions=dummy_mps,
        num_initial_reference_points=50,  # Keep small for tests
        version="convex_hull"
    )
    gprm_options = GPRMOptions(
        method_options=ipr_options,
        fake_ideal={"f_1": 0.0, "f_2": 0.0, "f_3": 0.0},
        fake_nadir={"f_1": 1.0, "f_2": 1.0, "f_3": 1.0},
        num_points_to_evaluate=5  # Keep small for tests
    )
    return FavOptions(
        GPRMoptions=gprm_options,
        candidate_generation_options="mm",
        zoom_options=ZoomOptions(num_steps_remaining=4),
        original_most_preferred_solutions=dummy_mps,
        total_n_of_candidates=5
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
                objectives={"f_1": i*0.1, "f_2": 1.0 - i*0.1, "f_3": 0.5}
            )
        )
    return points

# ==========================================
# 2. COMPONENT TESTS (Logic)
# ==========================================

def test_fractional_decay_logic():
    """
    Tests that the volume decay fraction calculates correctly 
    based on iterations and dimensionality (k-1).
    """
    max_iters = 5
    num_obj = 3  # E.g., DTLZ2 has 3 objectives

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
    """
    Tests the Voronoi Cannibalization fix. Proves that the override
    logic completely replaces the old candidate pool and safely assigns points.
    """
    # Mock evaluated points
    mock_points = [
        _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.1, "f_2": 0.9, "f_3": 0.1}),
        _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.9, "f_2": 0.1, "f_3": 0.1}),
        _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}),  # The center point
        _EvaluatedPoint(reference_point={}, targets={}, objectives={"f_1": 0.2, "f_2": 0.8, "f_3": 0.1}),
    ]

    # Mock the existing deadlocked candidates
    existing_cands = [
        FairSolution(objective_values={"f_1": 0.0, "f_2": 1.0, "f_3": 0.0}, fairness_criterion="core", fairness_value=0),
        FairSolution(objective_values={"f_1": 1.0, "f_2": 0.0, "f_3": 0.0}, fairness_criterion="mm", fairness_value=0),
        FairSolution(objective_values={"f_1": 0.2, "f_2": 0.2, "f_3": 0.8}, fairness_criterion="hausdorff_1", fairness_value=0),
    ]

    # The New Tie-Breaker Compromise (Sits right in the middle)
    compromise = FairSolution(
        objective_values={"f_1": 0.5, "f_2": 0.5, "f_3": 0.5},
        fairness_criterion="tie_breaker",
        fairness_value=0.0
    )

    # EXECUTE
    updated_cands, new_labels, winning_idx = recluster_for_tie_breaker(
        all_points=mock_points,
        existing_candidates=existing_cands,
        compromise_solution=compromise
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
        objective_values={"f_1": 0.1, "f_2": 0.9, "f_3": 0.5},
        fairness_criterion="mm",
        fairness_value=0.1
    )

    n_missing = 2
    results = hausdorff_candidates(dummy_evaluated_points, [seed_solution], n_missing)

    # Check lengths
    assert len(results) == 3, "Should return the 1 seed + 2 new candidates"

    # Check types
    assert all(isinstance(sol, FairSolution) for sol in results)

    # Check that the seed wasn't mutated or moved from index 0
    assert results[0].fairness_criterion == "mm"

    # Check that new candidates got the correct tag
    assert results[1].fairness_criterion == "avg_hausdorff"

def test_cluster_points(dummy_evaluated_points, base_options):
    """Tests if Voronoi partitioning returns correctly shaped arrays."""
    # Mock a FavResults object
    mock_gprm = GPRMResults(
        raw_results=IPR_Results(evaluated_points=dummy_evaluated_points),
        solutions=None, outputs=pl.DataFrame()
    )

    candidates = [
        FairSolution(objective_values={"f_1": 0.1, "f_2": 0.9, "f_3": 0.5}, fairness_criterion="mm", fairness_value=0),
        FairSolution(objective_values={"f_1": 0.9, "f_2": 0.1, "f_3": 0.5}, fairness_criterion="nash", fairness_value=0)
    ]

    mock_fav_results = FavResults(
        FavOptions=base_options, GPRMResults=mock_gprm, fair_solutions=candidates
    )

    pts_arr, centers_arr, labels = cluster_points(mock_fav_results)

    assert pts_arr.shape == (10, 3), "Points array should be (n_points, k_objectives)"
    assert centers_arr.shape == (2, 3), "Centers array should be (n_candidates, k_objectives)"
    assert labels.shape == (10,), "Labels array should have one entry per point"
    assert set(labels).issubset({0, 1}), "Labels should only map to the 2 candidate indices"

def test_select_final_candidates(dummy_problem, dummy_evaluated_points, base_options):
    """Tests the final phase voting logic with Hausdorff mapping."""
    mock_gprm = GPRMResults(
        raw_results=IPR_Results(evaluated_points=dummy_evaluated_points),
        solutions=None, outputs=pl.DataFrame()
    )

    # Create two dummy fair solutions
    candidates = [
        FairSolution(objective_values=dummy_evaluated_points[0].objectives, fairness_criterion="mm", fairness_value=0),
        FairSolution(objective_values=dummy_evaluated_points[9].objectives, fairness_criterion="nash", fairness_value=0)
    ]

    mock_fav_results = FavResults(
        FavOptions=base_options, GPRMResults=mock_gprm, fair_solutions=candidates
    )

    # Simulate that all points belong to cluster 0
    labels = np.zeros(10, dtype=int)

    final_sols = select_final_candidates(
        problem=dummy_problem, fav_results=mock_fav_results, cluster_labels=labels, winning_idx=0, n_candidates=3
    )

    assert len(final_sols) == 3
    assert final_sols[0].fairness_criterion == "last_winner", "Core candidate must be at index 0"
    assert final_sols[1].fairness_criterion == "final_mm", "Next candidate should be maxmin at index 1"
    assert final_sols[2].fairness_criterion == "final_hausdorff", "Subsequent candidates should be hausdorff"
    # Ensure the core candidate's values match the winner
    assert final_sols[0].objective_values == dummy_evaluated_points[0].objectives

# ==========================================
# 3. DATA FLOW & PIPELINE TESTS
# ==========================================

@patch("desdeo.gdm.favorite_method.guess_best_solver")
def test_problem_wrapper_data_flow(mock_guess, dummy_problem):
    """Tests if the ProblemWrapper correctly formats the solver inputs/outputs."""

    # Setup the mock solver to return a fake result
    mock_solver_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.optimal_objectives = {"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}
    mock_solver_instance.solve.return_value = mock_result

    # guess_best_solver returns a class, which is immediately instantiated.
    mock_guess.return_value = MagicMock(return_value=mock_solver_instance)

    fake_ideal = {"f_1": 0.0, "f_2": 0.0, "f_3": 0.0}
    fake_nadir = {"f_1": 1.0, "f_2": 1.0, "f_3": 1.0}

    wrapper = ProblemWrapper(dummy_problem, fake_ideal, fake_nadir)

    # Pass a normalized reference point array
    res = wrapper.solve([0.2, 0.2, 0.2])

    assert len(res) == 1
    assert isinstance(res[0], _EvaluatedPoint)
    assert res[0].targets == {"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}, "Targets should be normalized based on ideal/nadir"


@patch("desdeo.gdm.favorite_method.get_representative_set_IPR")
@patch("desdeo.gdm.favorite_method.find_group_solutions")
def test_favorite_method_first_iteration(mock_find_group, mock_get_ipr, dummy_problem, base_options):
    """
    Tests the main orchestrator (favorite_method) for a first iteration.
    We mock the heavy computations to ensure the Pydantic data flows correctly.
    """

    # 1. Mock the IPR GPRMResults
    mock_ipr_res = GPRMResults(
        raw_results=IPR_Results(evaluated_points=[]),
        solutions=pl.DataFrame(),
        outputs=pl.DataFrame()
    )
    mock_get_ipr.return_value = mock_ipr_res

    # 2. Mock the Fairness Algorithm returning 1 candidate
    mock_fair_sol = FairSolution(
        objective_values={"f_1": 0.5, "f_2": 0.5, "f_3": 0.5},
        fairness_criterion="mm", fairness_value=0.0
    )
    mock_find_group.return_value = [mock_fair_sol]

    # 3. Mock Hausdorff (otherwise it fails trying to calculate distances on empty points)
    with patch("desdeo.gdm.favorite_method.hausdorff_candidates") as mock_hausdorff:
        mock_hausdorff.return_value = [mock_fair_sol, mock_fair_sol, mock_fair_sol]  # Total of 3 candidates

        # ACT: Run the method
        final_results = favorite_method(dummy_problem, base_options, results_list=[])

        # ASSERT: Check types and data routing
        assert isinstance(final_results, FavResults)
        assert len(final_results.fair_solutions) == 3, "Should have expanded to total_n_of_candidates"
        assert final_results.FavOptions.votes is None, "First iteration should not have votes"

        # Ensure setup() populated the method_options with the original MPS
        assert final_results.FavOptions.GPRMoptions.method_options.most_preferred_solutions == base_options.original_most_preferred_solutions

def test_find_group_solutions_data_flow(dummy_problem, dummy_mps):
    """Tests if find_group_solutions safely normalizes and handles Polars DataFrames."""

    # Create fake outputs from the solver
    targets_df = pl.DataFrame({"f_1": [0.1, 0.9], "f_2": [0.9, 0.1], "f_3": [0.5, 0.5]})

    solutions_df = pl.DataFrame({"f_1": [0.1, 0.9], "f_2": [0.9, 0.1], "f_3": [0.5, 0.5]})

    fair_sols = find_group_solutions(
        problem=dummy_problem,
        solutions=solutions_df,
        targets=targets_df,
        most_preferred_solutions=dummy_mps,
        fairness_criterion="mm"
    )

    assert len(fair_sols) == 1
    assert isinstance(fair_sols[0], FairSolution)
    assert "f_1" in fair_sols[0].objective_values
    assert fair_sols[0].fairness_criterion == "mm"


@pytest.mark.slow
def test_favorite_method_e2e_integration(dummy_problem, dummy_mps):
    """
    END-TO-END INTEGRATION TEST.
    No mocks. Runs the real solvers to ensure the 
    math, constraints, and API contracts are fully functional.
    """
    # 1. Setup a very lightweight configuration to keep the test fast
    ipr_options = IPR_Options(
        most_preferred_solutions=dummy_mps,
        num_initial_reference_points=15,  # Very small sample
        version="convex_hull"
    )
    gprm_options = GPRMOptions(
        method_options=ipr_options,
        fake_ideal={"f_1": 0.0, "f_2": 0.0, "f_3": 0.0},
        fake_nadir={"f_1": 1.0, "f_2": 1.0, "f_3": 1.0},
        num_points_to_evaluate=3  # Only run the real solver 3 times!
    )
    options = FavOptions(
        GPRMoptions=gprm_options,
        candidate_generation_options="mm",
        zoom_options=ZoomOptions(num_steps_remaining=4),
        original_most_preferred_solutions=dummy_mps,
        total_n_of_candidates=3
    )

    # 2. ACT: Run the real pipeline
    results = favorite_method(dummy_problem, options, results_list=[])

    # 3. ASSERT: Verify the solver actually produced real floating-point math
    assert isinstance(results, FavResults)
    assert len(results.GPRMResults.raw_results.evaluated_points) == 3
    assert len(results.fair_solutions) == 3

    # Prove the solver didn't just return Nones or crash silently
    for sol in results.fair_solutions:
        for val in sol.objective_values.values():
            assert isinstance(val, float)


@patch("desdeo.gdm.favorite_method.add_asf_diff")
@patch("desdeo.gdm.favorite_method.guess_best_solver")
def test_tie_breaker_avgproj(mock_guess, mock_add_asf, dummy_problem):
    """
    Tests the tie-breaker functionality: verifying the average is calculated correctly
    and the solver pipeline is triggered and routed.
    """
    # Force the problem to use the differentiable path for predictable mocking
    # dummy_problem.is_twice_differentiable = True

    # 1. Setup mock solver to return a fake projected point on the Pareto front
    mock_solver_instance = MagicMock()
    mock_result = MagicMock()

    # Pretend the solver drops the point onto the Pareto optimal coordinates of (2.5, 2.5, 2.5)
    mock_result.optimal_objectives = {"f_1": 2.5, "f_2": 2.5, "f_3": 2.5}
    mock_solver_instance.solve.return_value = mock_result
    mock_guess.return_value = MagicMock(return_value=mock_solver_instance)

    # Mock add_asf_diff to return dummy variables so the code doesn't crash during setup
    mock_add_asf.return_value = (MagicMock(), MagicMock())

    # 2. Setup a perfect 3-way tie
    votes = {
        "DM1": 0,  # Votes for Candidate 0
        "DM2": 1,  # Votes for Candidate 1
        "DM3": 2   # Votes for Candidate 2
    }

    # 3. Mock 3 candidates with mathematically clean values to easily verify the average
    candidates = [
        FairSolution(objective_values={"f_1": 0.0, "f_2": 2.0, "f_3": 4.0}, fairness_criterion="mm", fairness_value=0.0),
        FairSolution(objective_values={"f_1": 3.0, "f_2": 1.0, "f_3": 5.0}, fairness_criterion="mm", fairness_value=0.0),
        FairSolution(objective_values={"f_1": 6.0, "f_2": 6.0, "f_3": 0.0}, fairness_criterion="mm", fairness_value=0.0),
    ]

    # 4. ACT: Run the tie-breaker!
    winning_sol = tie_breaker_avgproj(dummy_problem, votes, candidates)

    # 5. ASSERT A: Did it calculate the average correctly?
    # Expected Average:
    # f_1: (0 + 3 + 6) / 3 = 3.0
    # f_2: (2 + 1 + 6) / 3 = 3.0
    # f_3: (4 + 5 + 0) / 3 = 3.0
    mock_add_asf.assert_called_once()

    # Extract the 3rd argument passed into `add_asf_diff(problem, "target", avg_point)`
    passed_avg_point = mock_add_asf.call_args[0][2]

    assert passed_avg_point == {"f_1": 3.0, "f_2": 3.0, "f_3": 3.0}, "The tie-breaker failed to calculate the correct average!"

    # 6. ASSERT B: Did it return the projected FairSolution correctly?
    assert isinstance(winning_sol, FairSolution)
    assert winning_sol.objective_values == {"f_1": 2.5, "f_2": 2.5, "f_3": 2.5}, "Failed to route the solver's projected coordinates!"
    assert winning_sol.fairness_criterion == "tie_breaker_average_projection", "Tag was not applied correctly."
