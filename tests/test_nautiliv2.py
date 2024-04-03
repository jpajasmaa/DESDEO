"""Tests related to the NAUTILI method."""
import numpy as np
import numpy.testing as npt
import pytest
from fixtures import dtlz2_5x_3f_data_based  # noqa: F401

from desdeo.mcdm.nautiliv2 import (
    calculate_distance_to_front,
    calculate_navigation_point,
    solve_reachable_bounds,
    solve_reachable_solution,
    agg_maxmin, 
    agg_maxmin_cones,
    nautili_all_steps,
    NAUTILI_Response, 
)

from desdeo.problem import (
    binh_and_korn,
    objective_dict_to_numpy_array,
    river_pollution_problem,
    get_nadir_dict,
)


@pytest.mark.slow
@pytest.mark.nautili
def test_nautili_aggregation():
    """TODO: Test nautili aggregation aggregation """

    #problem = dtlz2_5x_3f_data_based
    #prev_nav_point = get_nadir_dict(problem)
    #ideal = problem.ideal

    #rps = {
    #    "DM1": {"f1": 0.8, "f2": 0.7, "f3": 0.6},
    #    "DM2": {"f1": 0.7, "f2": 0.8, "f3": 0.5},
    #    "DM3": {"f1": 0.5, "f2": 0.6, "f3": 0.8},
    #}


@pytest.mark.slow
@pytest.mark.nautili
def test_nautili_all_steps():
    #problem = dtlz2_5x_3f_data_based 
    problem = binh_and_korn(maximize=(False, False))

    nav_point = {"f_1": 60.0, "f_2": 20.1}
    ideal = {"f_1": 0.0, "f_2": 0.0}

    rps = {
        "DM1": {"f_1": 50, "f_2": 5},
        "DM2": {"f_1": 30, "f_2": 3},
        "DM3": {"f_1": 20, "f_2": 14},
    }

    #prev_nav_point = problem.nadir
    #nav_point = get_nadir_dict(problem)
    #nav_point = {"f1": 1.0, "f2": 1.0, "f3": 1.0}
    #ideal = {"f1": 0.0, "f2": 0.0, "f3": 0.0}

    #rps = {
    #    "DM1": {"f1": 0.8, "f2": 0.7, "f3": 0.6},
    #    "DM2": {"f1": 0.7, "f2": 0.8, "f3": 0.5},
    #    "DM3": {"f1": 0.5, "f2": 0.6, "f3": 0.8},
    #}
    lower_bounds, upper_bounds = solve_reachable_bounds(problem, nav_point)
    prev_resp = [NAUTILI_Response(
        distance_to_front=0,
        navigation_point=nav_point,
        reachable_bounds={"lower_bounds": lower_bounds, "upper_bounds": upper_bounds},
        reachable_solution=None,
        reference_points=None,
        improvement_directions=None,
        group_improvement_direction=None,
        step_number=0,
        pref_agg_method="mean"
    )]

    res_1 = nautili_all_steps(problem, 5, rps, prev_resp, nav_point)
    print(res_1)
    #assert res_1[-1].reference_points["DM1"].values < [0,0]
    assert res_1[-1].navigation_point["f_1"] < 0
    # res_1 = solve_reachable_solution(problem, rps, prev_nav_point)

    assert res_1.success
