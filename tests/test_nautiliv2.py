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
    aggregate,
)

from desdeo.problem import (
    binh_and_korn,
    objective_dict_to_numpy_array,
    river_pollution_problem,
    get_nadir_dict,
)


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.nautili
def test_nautili_aggregation_mean():
    """TODO: Test nautili aggregation aggregation """

    #test do not make sense rn
    problem = dtlz2_5x_3f_data_based
    #nav_point = get_nadir_dict(problem)
    nav_point = [1.,1.,1.]
    ideal = [0.,0.,0.]

    improvement_directions = {
        "DM1": np.array([0.9, 0.8, 0.4]),
        "DM2": np.array([0.8, 0.8, 0.5]),
        "DM3": np.array([0.5, 0.6, 0.8]),
    }
    nav_point_arr = np.array([0.9, 0.9, 0.9])
    g_improvement_direction = aggregate("mean", improvement_directions, nav_point_arr, None, None, None, None)
    assert g_improvement_direction is not None
    g_improvement_direction2 = aggregate("median", improvement_directions, nav_point_arr, None, None, None, None)
    assert g_improvement_direction2 is not None


@pytest.mark.slow
@pytest.mark.nautili
def test_nautili_aggregation_maxmin():
    """TODO: Test nautili aggregation aggregation """

    problem = dtlz2_5x_3f_data_based
    nadir = [1.,1.,1.]
    ideal = [0.,0.,0.]

    rps = {
        "DM1": np.array([0.9, 0.8, 0.4]),
        "DM2": np.array([0.8, 0.8, 0.5]),
        "DM3": np.array([0.5, 0.6, 0.8]),
    }
    nav_point_arr = np.array([0.9, 0.9, 0.9])
    g_improvement_direction = aggregate("maxmin", rps, nav_point_arr, 3, 3, ideal, nadir)
    assert g_improvement_direction is not None
    g_improvement_direction2 = aggregate("maxmin_cones", rps, nav_point_arr, 3, 3, ideal, nadir)
    assert g_improvement_direction2 is not None

    #g_improvement_direction = aggregate("mean", improvement_directions, nav_point, nav_point_arr, None, None, None, None)
    #assert g_improvement_direction is not None



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
        #pref_agg_method=None,
    )]

    rps2 = {"f_1": 30, "f_2": 3}
    #  test the aggregation erikseen.
    group_dir = rps2

    res_1 = solve_reachable_solution(problem, group_dir, nav_point)
    assert res_1 is not None

    res_2 = nautili_all_steps(problem, 5, rps, prev_resp, nav_point, "mean")
    print(res_2)
    assert res_2 is not None
    #assert res_1[-1].reference_points["DM1"].values < [0,0]
    #assert res_1[-1].navigation_point["f_1"] < 0

    assert res_1.success
