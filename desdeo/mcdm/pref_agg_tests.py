import numpy as np
import numpy.testing as npt
#from fixtures import dtlz2_5x_3f_data_based  # noqa: F401


import polars as pl

import pandas as pd

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
    nautili_all_steps,
    nautili_init
)

from desdeo.problem import (
    binh_and_korn,
    objective_dict_to_numpy_array,
    river_pollution_problem,
    get_nadir_dict,
    get_ideal_dict,
    dtlz2,
    #variable_dict_to_numpy_array,
    objective_dict_to_numpy_array
)


def test_dtzl2(pref_agg_method):
    #problem = river_pollution_problem
    problem = dtlz2(10,3)
    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)

    total_steps = 3
    initial_response = nautili_init(problem)

    rps = {
        "DM1": {"f_1": 0.8, "f_2": 0.7, "f_3": 0.6},
        "DM2": {"f_1": 0.7, "f_2": 0.8, "f_3": 0.5},
        "DM3": {"f_1": 0.5, "f_2": 0.6, "f_3": 0.8},
    }

    all_resp = nautili_all_steps(
        problem,
        total_steps,
        rps,
        [initial_response], # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )
    print(all_resp)
    print(all_resp[-1].reference_points)
    print(all_resp[-1].group_improvement_direction)
    print(all_resp[-1].navigation_point)

# TODO:
def test_river_pollution(pref_agg_method):
    #problem = river_pollution_problem
    problem = dtlz2(10,3)
    nadir = [2.,2.,2.]
    ideal = [0.,0.,0.]

    print("ideal", ideal)

    total_steps = 5


    initial_response = nautili_init(problem)

    """
    rps = {
            "DM1": np.array([0.9, 0.8, 0.4]),
            "DM2": np.array([0.8, 0.8, 0.5]),
            "DM3": np.array([0.5, 0.6, 0.8]),
        }
    """
    rps = {
            "DM1": {"f_1": 0.9, "f_2": 0.9, "f_3": 0.01},
            "DM2": {"f_1": 0.8, "f_2": 0.8, "f_3": 0.5},
            "DM3": {"f_1": 0.5, "f_2": 0.6, "f_3": 0.6},
        }

    all_resp = nautili_all_steps(
        problem,
        total_steps,
        rps,
        [initial_response], # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )

    del all_resp[-1:]
    rps2 = {
            "DM1": {"f_1": 0.79, "f_2": 0.79, "f_3": 0.01},
            "DM2": {"f_1": 0.38, "f_2": 0.58, "f_3": 0.5},
            "DM3": {"f_1": 0.25, "f_2": 0.6, "f_3": 0.6},
        }
    print(all_resp)
    print(all_resp[-1].reference_points)
    print(all_resp[-1].group_improvement_direction)
    print(all_resp[-1].navigation_point)

    print("changing prefs========================================")

    all_resp = nautili_all_steps(
        problem,
        1,
        rps2,
        [all_resp], # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )

    print(all_resp)
    print(all_resp[-1].reference_points)
    print(all_resp[-1].group_improvement_direction)
    print(all_resp[-1].navigation_point)


def solve_prob(problem, pref_agg_method):

    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)

    rps = {
        "DM1": {"f_1": 0.8, "f_2": 0.7, "f_3": 0.6},
        "DM2": {"f_1": 0.7, "f_2": 0.8, "f_3": 0.5},
        "DM3": {"f_1": 0.5, "f_2": 0.6, "f_3": 0.8},
    }
    initial_response = nautili_init(problem)
    all_resp = nautili_all_steps(problem, 5, rps, [initial_response], pref_agg_method)
    return all_resp


def visu(problem, all_resp):
    lower_bounds = pl.DataFrame(
        [response.reachable_bounds["lower_bounds"] for response in all_resp]
    )
    upper_bounds = pl.DataFrame(
        [response.reachable_bounds["upper_bounds"] for response in all_resp]
    )
    reference_points = pl.DataFrame(
        [response.reference_points for response in all_resp[1:]]
    )
    navigation_points = pl.DataFrame(
        [response.navigation_point for response in all_resp]
    )
    reachable_sols = pl.DataFrame(
        [response.reachable_solution for response in all_resp]
    )
    group_dirs = pl.DataFrame(
        [all_resp[0].group_improvement_direction]
    )

    """
    rps = [] 
    for dm in reference_points:
        max_multiplier = [-1 if obj.maximize else 1 for obj in problem.objectives]
        rp = (
            np.array([reference_points[dm][obj.symbol] for obj in problem.objectives]) * max_multiplier
        )
        rps.append(rp)
    #rps
    """

    import plotly.express as ex

    #iteration_points = pd.DataFrame(
    #    navigation_points,
    #    columns=["f1", "f2", "f3"]
    #)
    fig2 = ex.scatter_3d(
        x=navigation_points[navigation_points.columns[0]],
        y=navigation_points[navigation_points.columns[1]],
        z=navigation_points[navigation_points.columns[2]],
    )
    # TODO: use brain to do this
    # TODO: add RPs
    # TODO: add PF?
    # TODO: add the reachable solutions solution

    #prefs = np.array(
    #    [pref[1] for pref in rps.items()]
    #)
    #prefs = np.vstack((prefs, prefs[0]))
    fig2.add_scatter3d(
        x=[0.8, 0.7, 0.6], 
        y= [0.7, 0.8, 0.5],
        z =[0.5, 0.6, 0.8],)
    #prefs = [
    #    [0.8, 0.7, 0.6],
    #    [0.7, 0.8, 0.5],
    #    [0.5, 0.6, 0.8],
    #]

    #fig2.add_scatter3d(
    #    x=prefs[:, 0],
    #    y=prefs[:, 1],
    #    z=prefs[:, 2],
    #)

    fig2.add_scatter3d(
        x=[group_dirs[0]],
        y=[group_dirs[1]],
        z=[group_dirs[2]],
    )


if __name__ == "__main__":
    #test_dtzl2("mean")
    #test_dtzl2("maxmin")
    #test_dtzl2("maxmin_cones")
    problem = dtlz2(10,3)
    all_resp = solve_prob(problem, "maxmin_cones")
    #visu(problem, all_resp)