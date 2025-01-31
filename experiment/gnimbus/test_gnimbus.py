"""Tests related to the GNIMBUS method."""

from aggregate_classifications import aggregate_classifications
from desdeo.tools import IpoptOptions, PyomoIpoptSolver, add_asf_diff
from desdeo.problem import dtlz2, nimbus_test_problem, zdt1, zdt2
import numpy as np
import numpy.testing as npt
import pytest

# from desdeo.mcdm.gnimbus import infer_classifications, solve_intermediate_solutions, solve_sub_problems
from gnimbus import (explain, voting_procedure, infer_classifications, agg_cardinal, infer_ordinal_classifications,
                     solve_intermediate_solutions, solve_sub_problems, convert_to_nimbus_classification, add_group_nimbusv2_sf_diff,
                     list_of_rps_to_dict_of_rps, dict_of_rps_to_list_of_rps)


@pytest.mark.gnimbus
def test_voting_procedure():

    problem = dtlz2(8, 3)

    solver_options = IpoptOptions()
    # get some initial solution
    initial_rp = {
        "f_1": 0.4, "f_2": 0.5, "f_3": 0.8
    }
    problem_w_sf, target = add_asf_diff(problem, "target", initial_rp)
    solver = PyomoIpoptSolver(problem_w_sf, solver_options)
    initial_result = solver.solve(target)

    # f1: 0.385, f2: 0.485, f3: 0.776
    initial_fs = initial_result.optimal_objectives

    print(initial_fs)
    # let f1 worsen until 0.6, keep f2, improve f3 until 0.6
    # irst_rp = {"f_1": 0.6, "f_2": initial_fs["f_2"], "f_3": 0.6}
    dms_rps = {
        "DM1": {"f_1": 0.35, "f_2": initial_fs["f_2"], "f_3": 1},  # improve f_1, keep f_2 same, impair f_3
        "DM2": {"f_1": 0.3, "f_2": 0.8, "f_3": 0.5},  # improve f_1 to 0.3, impair f_2, improve f_3 to 0.5
        "DM3": {"f_1": 0.5, "f_2": 0.6, "f_3": 0.0},  # impair f_1 to 0.5, impair f_2 to 0.6, improve f_3
    }

    num_desired = 3
    solution_results = solve_sub_problems(
        problem, initial_fs, dms_rps, num_desired, False, create_solver=PyomoIpoptSolver, solver_options=solver_options
    )
    # TODO: remove duplicates
    print()
    print(solution_results)
    print()
    # remove solutions without votes
    # TODO:DO I actually even need this?
    # voted_solutions = [s for i, s in enumerate(solution_results) if i not in list(votes_idxs.values())]
    # assert len(solution_results) != len(voted_solutions)

    voted_solutions = solution_results

    # TEST MAJORITY WINS
    # make different votes
    votes_idxs = {
        "DM1": 1,
        "DM2": 2,
        "DM3": 2
    }
    res = voting_procedure(problem, voted_solutions, votes_idxs)
    assert res == solution_results[2]
    next_current_solution = res.optimal_objectives
    print(next_current_solution)

    # TEST PLULARITY WINS
    # make different votes
    votes_idxs = {
        "DM1": 1,
        "DM2": 0,
        "DM3": 0,
        "DM4": 3,
        "DM5": 2

    }
    res = voting_procedure(problem, voted_solutions, votes_idxs)
    print("here", res)
    assert res == solution_results[0]
    next_current_solution = res.optimal_objectives
    print(next_current_solution)

    # TEST intermediate
    votes_idxs = {
        "DM1": 1,
        "DM2": 1,
        "DM3": 2,
        "DM4": 2
    }
    res = voting_procedure(problem, voted_solutions, votes_idxs)
    # assert res == solution_results[2] does not apply as we generate a new one.
    next_current_solution = res.optimal_objectives
    print(next_current_solution)
    # TEST according to gnimbus WINS
    votes_idxs = {
        "DM1": 1,
        "DM2": 0,
        "DM3": 2
    }
    res = voting_procedure(problem, voted_solutions, votes_idxs)
    print(res)
    print(solution_results[0])
    assert res == solution_results[0]  # for now tie-breaking rule is taking the first one. Error what?
    next_current_solution = res.optimal_objectives
    print(next_current_solution)

    # assert 0 == 1


@pytest.mark.nimbus
def test_agg_cardinal():
    reference_points = [
        {"f_1": ("<=", 0.25), "f_2": (">=", 0.8), "f_3": (">", 1)},
        {"f_1": ("=>", 0.5), "f_2": ("<=", 0.2), "f_3": ("<", 0)},
        # {"f_1": ("0", None), "f_2": ("<=", 0.4)},
    ]
    """
    classifications = {
        "f_1": ("<", None),
        "f_2": ("<=", 42.1),
        "f_3": (">=", 22.2),
        "f_4": ("0", None)
        }
    """
    problem = dtlz2(8, 3)

    current_point = {"f_1": 0.4, "f_2": 0.3, "f_3": 0.5}

    # for i in range(len(reference_points)):
    # for DM i
    res = agg_cardinal(reference_points, problem, current_point)
    print(res)
    # assert 0 == 1


@pytest.mark.gnimbus
def test_aggregate_classifications():
    # example = np.array([[2, 0, 1, 2], [1, 0, 2, 1], [2, 1, 0, 1]])
    example = np.array([[2, 0, 1], [1, 0, 2], [2, 1, 0]])
    result = aggregate_classifications(example)
    print(result)  # results look resonable
    # assert 0 == 1

@pytest.mark.gnimbus
def test_dict_to_list_and_back():
    rps = {
        "DM1": {"f_1": 0.0, "f_2": 0.5, "f_3": 1},
        "DM2": {"f_1": 0.0, "f_2": 1, "f_3": 0.5},
        "DM3": {"f_1": 0.5, "f_2": 1, "f_3": 0.0},
    }
    result = dict_of_rps_to_list_of_rps(rps)
    print(result)  # results look resonable
    back_to = list_of_rps_to_dict_of_rps(result)
    print(back_to)


@pytest.mark.nimbus
def test_convert_classification():
    """
    TODO: convert the DMs RPs to 2 = improve, 1 = keep same, 0 = worsen, for aggregate_classification
   # problem: Problem, current_objectives: dict[str, float], reference_points: dict[str, dict[str, float]]
 ) -> dict[str, tuple[str, float | None]]:

    """

    problem = dtlz2(8, 3)

    current_point = {"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}

    # Simple case wheere only ordinal information
    rps = {
        "DM1": {"f_1": 0.0, "f_2": 0.5, "f_3": 1},
        "DM2": {"f_1": 0.0, "f_2": 1, "f_3": 0.5},
        "DM3": {"f_1": 0.5, "f_2": 1, "f_3": 0.0},
    }
    res = infer_ordinal_classifications(problem, current_point, rps)
    print(res)

    # Testing that cardinal parts are converted to ordinal information
    rps = {
        "DM1": {"f_1": 0.0, "f_2": 0.5, "f_3": 1},
        "DM2": {"f_1": 0.0, "f_2": 1, "f_3": 0.5},
        "DM3": {"f_1": 0.5, "f_2": 0.6, "f_3": 0.0},  # TODO: should fail because of the 0.6
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

@pytest.mark.skip
@pytest.mark.nimbus
# TODO: this is not working right now
def test_convert_to_nimbus_classification_aggregation():
    example = np.array([[2, 0, 1], [1, 0, 2], [2, 1, 0]])
    compromise_classification = aggregate_classifications(example)["compromise"][0]  # get the compromise classif vector and only the first one here.
    print("compromise class:", compromise_classification)  # results look resonable [2,1,0]

    classification_list = compromise_classification
    problem = dtlz2(8, 3)

    print("classic", classification_list)
    # TESTABLE PART
    reference_points = [
        {"f_1": ("<=", 0.3), "f_2": ("0", None)},
        {"f_1": ("=>", 0.9), "f_2": ("<", None)},
        {"f_1": ("0", None), "f_2": ("<=", 0.4)},
    ]
    for obj in problem.objectives:
        for i in classification_list:
            if classification_list[i][obj.symbol] not in [0, 1, 2]:
                agg_value = 0
                agg = [reference_points[rp][obj.symbol] for rp in reference_points]
                agg_value = np.sum(agg)/3  # number of DMs, taking mean value.

                classification = {obj.symbol: str(classification_list[i][obj.symbol], agg_value)}

            classification_list.append(classification)

    classification_list = convert_to_nimbus_classification(problem, compromise_classification)

    print("classic", classification_list)
    assert isinstance(classification_list, dict)
    # TODO: should work until this.
    # assert 0 == 1


def test_gnimbus_scala():
    """Test that the multiple decision maker NIMBUS scalarization function works."""
    problem = zdt1(30)
    current_rp = {"f_1": 0.6, "f_2": 0.6, }

    # TODO: classifications to be converted to a ordered list; DM1 as first item of the list.
    classifications_for_all_DMs = [
        {"f_1": ("<=", 0.3), "f_2": ("0", None)},
        {"f_1": ("=>", 0.9), "f_2": ("<", None)},
    ]
    classifications_for_all_DMs2 = [
        {"f_1": ("<=", 0.3), "f_2": ("0", None)},
        {"f_1": ("=>", 0.9), "f_2": ("<", None)},
        {"f_1": ("0", None), "f_2": ("<=", 0.4)},
    ]
    problem_w_group_sf, group_sf = add_group_nimbusv2_sf_diff(problem, "group_sf", classifications_for_all_DMs, current_rp)
    solver_group_sf = PyomoIpoptSolver(problem_w_group_sf)
    res_group_sf = solver_group_sf.solve(group_sf)
    assert res_group_sf.success

    problem_w_group_sf2, group_sf2 = add_group_nimbusv2_sf_diff(problem, "group_sf2", classifications_for_all_DMs2, current_rp)
    solver_group_sf2 = PyomoIpoptSolver(problem_w_group_sf2)
    res_group_sf2 = solver_group_sf2.solve(group_sf2)
    assert res_group_sf2.success
    # solver_group_sf_3rp = PyomoIpoptSolver(problem_w_group_sf_3rp)
    # res_group_sf_3rp = solver_group_sf_3rp.solve(group_sf_3rp)
    # ssert res_group_sf_3rp.success

    fs_group_sf = res_group_sf.optimal_objectives
    fs_group_sf2 = res_group_sf2.optimal_objectives
    # fs_group_sf_3rp = res_group_sf_3rp.optimal_objectives

    print(fs_group_sf)
    print(fs_group_sf2)
    # print(fs_group_sf_3rp)

    # optimal objective values should be close
    # for obj in problem.objectives:
    # assert np.isclose(fs_group_sf_3rp[obj.symbol], fs_group_sf[obj.symbol], atol=1e-3)
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

    # f1: 0.385, f2: 0.485, f3: 0.776
    initial_fs = initial_result.optimal_objectives

    print(initial_fs)
    # let f1 worsen until 0.6, keep f2, improve f3 until 0.6
    # irst_rp = {"f_1": 0.6, "f_2": initial_fs["f_2"], "f_3": 0.6}
    dms_rps = {
        "DM1": {"f_1": 0.0, "f_2": initial_fs["f_2"], "f_3": 1},  # improve f_1, keep f_2 same, impair f_3
        "DM2": {"f_1": 0.3, "f_2": 1, "f_3": 0.5},  # improve f_1 to 0.3, impair f_2, improve f_3 to 0.5
        "DM3": {"f_1": 0.5, "f_2": 0.6, "f_3": 0.0},  # impair f_1 to 0.5, impair f_2 to 0.6, improve f_3
    }

    num_desired = 4
    solutions = solve_sub_problems(
        problem, initial_fs, dms_rps, num_desired, False, create_solver=PyomoIpoptSolver, solver_options=solver_options
    )

    # TODO: WORks until here because missing nimbus scala
    assert len(solutions) == num_desired
    print(solutions[0].optimal_objectives)
    print(solutions[1].optimal_objectives)
    print(solutions[2].optimal_objectives)
    print(solutions[3].optimal_objectives)

    # assert 0 == 1

    """ lets ignore these for now because there are things to fix with the group scalarizaiton fucntions too.
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
        """

@pytest.mark.nimbus
@pytest.mark.slow
def test_2solve_sub_problems_decision_phase():
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
    dms_rps = {
        "DM1": {"f_1": 0.0, "f_2": 0.5, "f_3": 1},
        "DM2": {"f_1": 0.0, "f_2": 1, "f_3": 0.5},
        "DM3": {"f_1": 0.5, "f_2": 1, "f_3": 0.0},
    }

    # TODO: only simple not realistic test case to test stuff
    initial_rp = {"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}

    # Simple case wheere only ordinal information
    dms_rps = {
        "DM1": {"f_1": 0.0, "f_2": initial_fs["f_2"], "f_3": 1},
        "DM2": {"f_1": 0.0, "f_2": 1, "f_3": initial_fs["f_3"]},
        "DM3": {"f_1": initial_fs["f_1"], "f_2": 1, "f_3": 0.0},
    }
    # let f1 worsen until 0.6, keep f2, improve f3 until 0.6
    # irst_rp = {"f_1": 0.6, "f_2": initial_fs["f_2"], "f_3": 0.6}
    dms_rps = {
        "DM1": {"f_1": 0.0, "f_2": 0.8, "f_3": 1},
        "DM2": {"f_1": 0.0, "f_2": 1, "f_3": 0.3},
        "DM3": {"f_1": 0.5, "f_2": 1, "f_3": 0.0},
    }

    classifications_for_all_DMs = [
        {"f_1": ("<=", 0.3), "f_2": ("0", None)},
        {"f_1": ("=>", 0.9), "f_2": ("<", None)},
        {"f_1": ("0", None), "f_2": ("<=", 0.4)},
    ]
    num_desired = 1
    solutions = solve_sub_problems(
        problem, initial_fs, dms_rps, num_desired, True, create_solver=PyomoIpoptSolver, solver_options=solver_options
    )

    print(solutions[0].optimal_objectives)
    # TODO: WORks until here because missing nimbus scala
    assert len(solutions) == num_desired
    # assert 0 == 1

    """ lets ignore these for now because there are things to fix with the group scalarizaiton fucntions too.
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
    """

@pytest.mark.skip
@pytest.mark.nimbus
@pytest.mark.slow
def test_forest_problem():
    """Test that the scalarization problems in GNIMBUS are solved as expected."""

    from desdeo.problem.testproblems import forest_problem, forest_problem_discrete

    from desdeo.tools.utils import guess_best_solver

    from desdeo.tools import ProximalSolver, GurobipySolver

    problem = forest_problem_discrete()

    # get some initial solution
    initial_rp = {
        "stock": 2000, "harvest_value": 30500, "npv": 75000
    }
    problem_w_sf, target = add_asf_diff(problem, "target", initial_rp)
    # solver = PyomoIpoptSolver(problem_w_sf, solver_options)
    # solver = guess_best_solver(problem)
    solver = ProximalSolver(problem)
    print(solver)
    print(target)
    initial_result = solver.solve("stock")

    # TODO to figrue out how this meess works

    print(initial_result)
    # f1: 0.4355, f2: 0.3355, f3: 0.8355
    initial_fs = initial_result.optimal_objectives

    DM_rps = {
        "DM1": {
            "stock": 2800,
            "harvest_value": 30000,
            "npv": 80000,
        },
        "DM2": {
            "stock": 3000,
            "harvest_value": 25200.5,
            "npv": 76500,
        },
        "DM3": {
            "stock": 2900,
            "harvest_value": 50005,
            "npv": 90000,
        },
        "DM4": {
            "stock": 3100,
            "harvest_value": 22000,
            "npv": 90000,
        },
        "DM5": {
            "stock": 3200,
            "harvest_value": 29000,
            "npv": 76000,
        },
    }


# TODO: only simple not realistic test case to test stuff
    initial_rp = {"f_1": 0.5, "f_2": 0.5, "f_3": 0.5}

# Simple case wheere only ordinal information
    dms_rps = {
        "DM1": {"f_1": 0.0, "f_2": initial_fs["f_2"], "f_3": 1},
        "DM2": {"f_1": 0.0, "f_2": 1, "f_3": initial_fs["f_3"]},
        "DM3": {"f_1": initial_fs["f_1"], "f_2": 1, "f_3": 0.0},
    }
# let f1 worsen until 0.6, keep f2, improve f3 until 0.6
# irst_rp = {"f_1": 0.6, "f_2": initial_fs["f_2"], "f_3": 0.6}
    dms_rps = {
        "DM1": {"f_1": 0.0, "f_2": 0.8, "f_3": 1},
        "DM2": {"f_1": 0.0, "f_2": 1, "f_3": 0.3},
        "DM3": {"f_1": 0.5, "f_2": 1, "f_3": 0.0},
    }

    classifications_for_all_DMs = [
        {"f_1": ("<=", 0.3), "f_2": ("0", None)},
        {"f_1": ("=>", 0.9), "f_2": ("<", None)},
        {"f_1": ("0", None), "f_2": ("<=", 0.4)},
    ]
    num_desired = 1
    solutions = solve_sub_problems(
        problem, initial_fs, dms_rps, num_desired, True, create_solver=PyomoIpoptSolver, solver_options=solver_options
    )

    print(solutions[0].optimal_objectives)
# TODO: WORks until here because missing nimbus scala
    assert len(solutions) == num_desired
# assert 0 == 1
