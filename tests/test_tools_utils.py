"""Tests the utils in the desdeo.tools package."""

import shutil

import pytest
from fixtures import dtlz2_5x_3f_data_based  # noqa: F401

from desdeo.problem.testproblems import (
    dtlz2,
    re21,
    river_pollution_problem,
    simple_constrained_quadratic_tensor_test_problem,
)
from desdeo.tools.utils import (
    available_solvers,
    filter_duplicate_solutions,
    find_compatible_solvers,
    guess_best_solver,
    is_duplicate_solution,
    payoff_table_method,
)


@pytest.mark.utils
def test_guess_best_solver(dtlz2_5x_3f_data_based):  # noqa: F811
    """Test that the best solver guesser guesses as expected for different problem types."""
    analytical_problem = river_pollution_problem()
    data_problem = dtlz2_5x_3f_data_based

    analytical_guess = guess_best_solver(analytical_problem)

    assert analytical_guess is available_solvers["nevergrad"]["constructor"]

    data_guess = guess_best_solver(data_problem)

    assert data_guess is available_solvers["proximal"]["constructor"]


@pytest.mark.utils
def test_find_compatible_solvers():
    """Test that find_compatible_solvers works as intended."""
    problem = re21()

    solvers = find_compatible_solvers(problem)

    correct_solvers = [
        available_solvers["pyomo_ipopt"]["constructor"],
        available_solvers["nevergrad"]["constructor"],
        available_solvers["scipy_minimize"]["constructor"],
        available_solvers["scipy_de"]["constructor"],
    ]

    # check that the solvers found are the correct ones
    if shutil.which("ipopt"):
        assert len(solvers) == 4
        assert all(solver in correct_solvers for solver in solvers) and all(
            solver in solvers for solver in correct_solvers
        )
    else:
        assert len(solvers) == 3

    problem = simple_constrained_quadratic_tensor_test_problem(dqp=True)

    solvers = find_compatible_solvers(problem)

    correct_solvers = [
        available_solvers["pyomo_ipopt"]["constructor"],
        available_solvers["cvxpy"]["constructor"],
    ]

    # check that the solvers found are the correct ones
    if shutil.which("ipopt"):
        assert len(solvers) == 2
        assert all(solver in correct_solvers for solver in solvers) and all(
            solver in solvers for solver in correct_solvers
        )
    else:
        assert len(solvers) == 1


@pytest.mark.utils
def test_payoff_dtlz2():
    """Tests the payoff-table method with the dtlz2 problem."""
    problem = dtlz2(6, 4)

    ideal, nadir = payoff_table_method(problem)  # noqa: RUF059


@pytest.mark.utils
def test_is_duplicate_solution():
    """Test generic solution equivalence across objective and decision spaces."""
    sol1 = {"optimal_objectives": {"f1": 1.0, "f2": 2.0}, "optimal_variables": {"x1": 0.5, "x2": 0.5}}
    sol2 = {"optimal_objectives": {"f1": 1.0, "f2": 2.0}, "optimal_variables": {"x1": 0.5, "x2": 0.5}}
    sol_multimodal = {"optimal_objectives": {"f1": 1.0, "f2": 2.0}, "optimal_variables": {"x1": 0.9, "x2": 0.1}}
    sol_diff_obj = {"optimal_objectives": {"f1": 1.5, "f2": 2.0}, "optimal_variables": {"x1": 0.5, "x2": 0.5}}

    # Identical in both spaces
    assert is_duplicate_solution(sol1, sol2, check_variables=True) is True
    assert is_duplicate_solution(sol1, sol2, check_variables=False) is True

    # Multi-modal: identical objectives, distinct decision variables
    assert is_duplicate_solution(sol1, sol_multimodal, check_variables=True) is False
    assert is_duplicate_solution(sol1, sol_multimodal, check_variables=False) is True

    # Different objectives
    assert is_duplicate_solution(sol1, sol_diff_obj, check_variables=True) is False

    # Filtering
    distinct = filter_duplicate_solutions([sol1, sol2, sol_multimodal, sol_diff_obj], check_variables=True)
    assert len(distinct) == 3
    assert distinct[0] == sol1
    assert distinct[1] == sol_multimodal
    assert distinct[2] == sol_diff_obj

    # Filtering without variable checking merges multi-modal solutions
    distinct_obj_only = filter_duplicate_solutions([sol1, sol2, sol_multimodal, sol_diff_obj], check_variables=False)
    assert len(distinct_obj_only) == 2
