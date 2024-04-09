import numpy as np
import numpy.testing as npt
#from fixtures import dtlz2_5x_3f_data_based  # noqa: F401

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
    nautili_init,
)

from desdeo.problem import (
    binh_and_korn,
    objective_dict_to_numpy_array,
    river_pollution_problem,
    get_nadir_dict,
    dtlz2
)


def test_dtzl2(pref_agg_method):
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
            "DM1": {"f_1": 0.9, "f_2": 0.8, "f_3": 0.4},
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
    print(all_resp)
    print(all_resp[-1].reference_points)
    print(all_resp[-1].group_improvement_direction)
    print(all_resp[-1].navigation_point)


    #del all_resp[-1:]
    rps2 = {
            "DM1": {"f_1": 0.79, "f_2": 0.79, "f_3": 0.01},
            "DM2": {"f_1": 0.38, "f_2": 0.58, "f_3": 0.5},
            "DM3": {"f_1": 0.25, "f_2": 0.6, "f_3": 0.6},
        }
    print(all_resp)

    print("changing prefs========================================")

    all_resp = nautili_all_steps(
        problem,
        5,
        rps2,
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



if __name__ == "__main__":
    test_dtzl2("mean")
