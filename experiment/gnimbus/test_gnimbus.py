"""Tests related to the GNIMBUS method."""

import numpy as np
import numpy.testing as npt
import pytest

# from desdeo.mcdm.gnimbus import infer_classifications, solve_intermediate_solutions, solve_sub_problems
from gnimbus import infer_classifications, infer_ordinal_classifications, solve_intermediate_solutions, solve_sub_problems, convert_to_nimbus_classification
from desdeo.problem import dtlz2, nimbus_test_problem
from desdeo.tools import IpoptOptions, PyomoIpoptSolver, add_asf_diff

from aggregate_classifications import aggregate_classifications

@pytest.mark.gnimbus
def test_aggregate_classifications():
    # example = np.array([[2, 0, 1, 2], [1, 0, 2, 1], [2, 1, 0, 1]])
    example = np.array([[2, 0, 1], [1, 0, 2], [2, 1, 0]])
    result = aggregate_classifications(example)
    print(result)  # results look resonable
    # assert 0 == 1


@pytest.mark.nimbus
def test_convert_classification():
    """
    TODO: convert the DMs RPs to 2 = improve, 1 = keep same, 0 = worsen, for aggregate_classification
    """
   # problem: Problem, current_objectives: dict[str, float], reference_points: dict[str, dict[str, float]]
# ) -> dict[str, tuple[str, float | None]]:

    problem = dtlz2(8, 3)

    current_point = {"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}

    rps = {
        "DM1": {"f_1": 0.0, "f_2": 0.5, "f_3": 1},
        "DM2": {"f_1": 0.0, "f_2": 1, "f_3": 0.5},
        "DM3": {"f_1": 0.5, "f_2": 1, "f_3": 0.0},
    }

    res = infer_ordinal_classifications(problem, current_point, rps)
    print(res)
    # should work


@pytest.mark.nimbus
def test_convert_to_nimbus_classification():
    example = np.array([[2, 0, 1], [1, 0, 2], [2, 1, 0]])
    compromise_classification = aggregate_classifications(example)["compromise"][0]  # get the compromise classif vector and only the first one here.
    print("compromise class:", compromise_classification)  # results look resonable [2,1,0]

    problem = dtlz2(8, 3)
    classif = convert_to_nimbus_classification(problem, compromise_classification)
    print("classic", classif)
    assert isinstance(classif, dict)
    # TODO: should work until this.
    # assert 0 == 1


@pytest.mark.nimbus
def test_infer_classifications():
    """Test that classifications are inferred correctly."""
    problem = nimbus_test_problem()

    current_objectives = {"f_1": 4.5, "f_2": 3.2, "f_3": -5.2, "f_4": -1.2, "f_5": 120.0, "f_6": 9001.0}

    # f_1: improve until
    # f_2: keep as it is
    # f_3: improve without limit
    # f_4: let change freely
    # f_5: impair until
    # f_6: improve until
    reference_point = {"f_1": 6.9, "f_2": 3.2, "f_3": -6.0, "f_4": 2.0, "f_5": 160.0, "f_6": 9000.0}

    classifications = infer_classifications(problem, current_objectives, reference_point)

    # f_1: improve until
    assert classifications["f_1"][0] == "<="
    assert np.isclose(classifications["f_1"][1], reference_point["f_1"])

    # f_2: keep as it is
    assert classifications["f_2"][0] == "="
    assert classifications["f_2"][1] is None

    # f_3: improve without limit
    assert classifications["f_3"][0] == "<"
    assert classifications["f_3"][1] is None

    # f_4: let change freely
    assert classifications["f_4"][0] == "0"
    assert classifications["f_4"][1] is None

    # f_5: impair until
    assert classifications["f_5"][0] == ">="
    assert np.isclose(classifications["f_5"][1], reference_point["f_5"])

    # f_6: improve until
    assert classifications["f_6"][0] == "<="
    assert np.isclose(classifications["f_6"][1], reference_point["f_6"])


@pytest.mark.nimbus
@pytest.mark.slow
def test_solve_sub_problems():
    """Test that the scalarization problems in GNIMBUS are solved as expected."""
    n_variables = 8
    n_objectives = 3

    problem = dtlz2(n_variables, n_objectives)

    solver_options = IpoptOptions()

    # get some initial solution
    initial_rp = {
        "f_1": 0.4, "f_2": 0.5, "f_3": 0.8
    }
    problem_w_sf, target = add_asf_diff(problem, "target", initial_rp)
    solver = PyomoIpoptSolver(problem_w_sf, solver_options)
    initial_result = solver.solve(target)

    # f1: 0.4355, f2: 0.3355, f3: 0.8355
    initial_fs = initial_result.optimal_objectives

    # let f1 worsen until 0.6, keep f2, improve f3 until 0.6
    # irst_rp = {"f_1": 0.6, "f_2": initial_fs["f_2"], "f_3": 0.6}
    dms_rps = [
        {"f_1": 0.4, "f_2": 0.5, "f_3": 0.8},
        {"f_1": 0.4, "f_2": 0.3, "f_3": 0.2},
    ]

    num_desired = 3
    solutions = solve_sub_problems(
        problem, initial_fs, dms_rps, num_desired, create_solver=PyomoIpoptSolver, solver_options=solver_options
    )

    # TODO: WORks until here because missing nimbus scala
    assert len(solutions) == num_desired

    # check that the solutions are Pareto optimal
    for solution in solutions:
        assert solution.success
        npt.assert_almost_equal(
            [solution.optimal_variables[f"x_{i+1}"] for i in range(n_objectives - 1, n_variables)], 0.5
        )
        npt.assert_almost_equal(
            sum(solution.optimal_objectives[f"{obj.symbol}"] ** 2 for obj in problem.objectives), 1.0
        )

    # check that solutions make sense TODO: update this does not correspond anymore as more DMs, different results
    for i, solution in enumerate(solutions):
        fs = solution.optimal_objectives

        # f1 should have worsened, but only until 0.6
        assert fs["f_1"] > initial_fs["f_1"]
        assert np.isclose(fs["f_1"], 0.6) or fs["f_1"] > 0.6

        # f2 should be same or better
        if i == 0:
            # NIMBUS scalarization, f_2 must be either as good or better
            assert np.isclose(fs["f_2"], initial_fs["f_2"]) or fs["f_2"] < initial_fs["f_2"]
        else:
            # other scalarization functions are more lenient, f2 is close to current point
            assert abs(fs["f_2"] - initial_fs["f_2"]) < 0.1

        # f3 should have improved
        assert fs["f_3"] < initial_fs["f_3"]
