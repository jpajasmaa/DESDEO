""" Tests related to preference_aggregation of reference points and maxmin fairness """
import numpy as np
import numpy.testing as npt
import pytest
# from fixtures import dtlz2_5x_3f_data_based  # noqa: F401

from preference_aggregation import find_GRP, find_GRP2, maxmin_criterion, maxmin_cones_criterion, aggregate, not_normalized_maxmin_criterion


from desdeo.mcdm.nautili import (
    calculate_distance_to_front,
    calculate_navigation_point,
    solve_reachable_bounds,
    solve_reachable_solution,
    nautili_all_steps,
    nautili_init
)
from desdeo.problem import (
    binh_and_korn,
    dtlz2,
    zdt1,
    zdt2,
    zdt3,
    momip_ti7,
    get_nadir_dict,
    get_ideal_dict,
    objective_dict_to_numpy_array,
    river_pollution_problem,
)


@pytest.mark.pref_agg
def test_maxmin_diff_scales():
    """Test nautili all steps"""

    n_objectives = 2
    problem = binh_and_korn()
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    rp1_np = [50, 20]
    rp2_np = [100., 25]
    rp3_np = [30, 50]
    all_rps = np.array([rp1_np, rp2_np, rp3_np])

    cip = nadir
    # cip = np.array([0.7, 0.8]) # if wanting to use some other cip

    k = n_objectives
    q = len(all_rps)
    pa = "maxmin"

    # grp = [rp1, rp2, rp3]
    # print(all_rps)
    GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("MAXMIN GRP", GRP)
    assert len(GRP) == n_objectives
    assert not np.isnan(GRP).any()  # "GRP should consist of real numbers."

    P = [50, 20]  # DM1 and DM2 should be happy
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]  # should be [-1, -1, 0]
    print("values", values)
    assert values[0] == 1
    assert values[1] > 1
    assert values[2] < 1

    P = [100, 25]  # DM2 should be max happy.
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]  # should be [0, x, x]

    print("values", values)
    assert values[0] < 1
    assert values[1] == 1
    assert values[2] < 1

    P = [30, 50]  # all DMs should have 1 or better.
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]

    print("values", values)
    assert values[0] > 1
    assert values[1] > 1
    assert values[2] == 1

    # assert 0 == 1  # to compare maxmin values


@pytest.mark.slow
@pytest.mark.pref_agg
def test_find_GRP():
    """Test this is now testing find_GRP """

    n_variables = 30
    problem = zdt1(n_variables)
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    prefs = {
        "DM1": {"f_1": 0.5, "f_2": 0.5},
        "DM2": {"f_1": 1., "f_2": 0},
        "DM3": {"f_1": 1, "f_2": 1},
    }
    # rp = {"f_1": 0.5, "f_2": 0.6 }
    # rp1 = {"f_1": 0.9, "f_2": 0.6, }
    # rp2 = {"f_1": 0.55, "f_2": 0.6, }
    # rp3 = {"f_1": 0.0, "f_2": 0.1, }
    rp1 = {"f_1": 0.5, "f_2": 0.5}
    rp2 = {"f_1": 0., "f_2": 1.}
    rp3 = {"f_1": 1., "f_2": 1.0}
    # rps in lists
    # rp_np =  [0.5, 0.6]
    rp1_np = [0.5, 0.5]
    rp2_np = [0., 1]
    rp3_np = [1, 1.0]

    cip = np.array([1., 1.])

    k = 2
    q = 3
    pa = "maxmin"

    # grp = [rp1, rp2, rp3]
    all_rps = np.array([rp1_np, rp2_np, rp3_np])
    print(all_rps)
    GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("MAXMIN GRP", GRP)

    assert len(GRP) == 2
    assert not np.isnan(GRP).any()  # "GRP should consist of real numbers."
    # assert type(all_resp[0]) is NAUTILI_Response

# @pytest.mark.skip
@pytest.mark.pref_agg
def test_maxmin_same_scales():
    """Test maxmin with same scaled objective functions"""

    n_variables = 30
    n_objectives = 2
    problem = zdt2(n_variables)
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    rp1_np = [0.5, 0.5]
    rp2_np = [0., 1]
    rp3_np = [1, 0]
    all_rps = np.array([rp1_np, rp2_np, rp3_np])

    cip = nadir
    # cip = np.array([0.7, 0.8]) # if wanting to use some other cip

    k = n_objectives
    q = 3
    pa = "maxmin"

    # grp = [rp1, rp2, rp3]
    # print(all_rps)
    GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("MAXMIN GRP", GRP)
    assert len(GRP) == n_objectives
    assert not np.isnan(GRP).any()  # "GRP should consist of real numbers."

    P = [.5, .5]  # testing that if GRP would be at DM1's point, he would have max value and others not
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]  # should be [-1, -1, 0]
    print("values", values)
    assert values[0] == 1
    assert values[1] != 1
    assert values[2] != 1

    P = [0., 1.]  # DM2 and DM1 should be max happy.
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]  # should be [0, x, x]

    print("values", values)
    assert values[0] == 1
    assert values[1] == 1
    assert values[2] != 1

    P = [1., 0]  # DM3 and DM1 should be max happy.
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]

    print("values", values)
    assert values[0] == 1
    assert values[1] != 1
    assert values[2] == 1

    # assert 0 == 1  # to see the values


@pytest.mark.slow
@pytest.mark.pref_agg
def test_maxmin_ext():
    """Test maxmin with same scaled objective functions"""

    n_variables = 30
    n_objectives = 2
    problem = zdt2(n_variables)
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    # rp_np =  [0.5, 0.6]
    rp1_np = [0.5, 0.5]
    rp2_np = [0.6, 0.8]
    rp3_np = [0.8, 0.6]

    cip = np.array([1., 1.])
    k = 2
    q = 3
    pa = "maxmin_ext"
    # grp = [rp1, rp2, rp3]
    all_rps = np.array([rp1_np, rp2_np, rp3_np])
    print(all_rps)
    GRP2 = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("MAXMIN GRP2", GRP2)
    assert not np.isnan(GRP2).any()  # "GRP should consist of real numbers."

    # GET normal maxmin compared to maxmin_ext
    cip = np.array([1., 1.])
    k = 2
    q = 3
    pa = "maxmin"
    # grp = [rp1, rp2, rp3]
    all_rps = np.array([rp1_np, rp2_np, rp3_np])
    print(all_rps)
    GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("MAXMIN GRP", GRP)
    assert not np.isnan(GRP).any()  # "GRP should consist of real numbers."

    assert GRP[0] == GRP2[0]
    assert GRP[1] == GRP2[1]

@pytest.mark.slow
@pytest.mark.pref_agg
def test_not_norm_maxmin_same_scales():
    """Test maxmin with same scaled objective functions"""

    n_variables = 30
    n_objectives = 2
    problem = zdt1(n_variables)
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    rp1_np = [0.5, 0.5]
    rp2_np = [0., 1]
    rp3_np = [1, 0]

    cip = np.array([1., 1.])

    k = 2
    q = 3
    pa = "maxmin"

    # grp = [rp1, rp2, rp3]
    all_rps = np.array([rp1_np, rp2_np, rp3_np])
    print(all_rps)
    GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("MAXMIN GRP", GRP)

    # TODO: check as they are not normalized, we need to multiply them by -1 to be suitalbe for scipy minimize?
    P = [1., 1.]  # testing that if GRP would be at DM3's point, he would have max value and others not
    values = [not_normalized_maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]
    print(values)
    assert values[0] != 1
    assert values[1] != 1
    assert np.isclose(values[2], 0)

    P = [0.5, 0.5]  # testing that if GRP would be at DM1's point, he wou

    # DM1 have max value and others not
    values = [not_normalized_maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]
    print("VV", values)

    assert np.isclose(values[0], 1)
    assert values[1] != 1
    assert values[2] != 1
