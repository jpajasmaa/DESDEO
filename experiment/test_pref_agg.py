""" Tests related to preference_aggregation of reference points and maxmin fairness """
import numpy as np
import numpy.testing as npt
import pytest
# from fixtures import dtlz2_5x_3f_data_based  # noqa: F401

from preference_aggregation import find_GRP, find_GRP_simple, find_GRP_old, maxmin_criterion, maxmin_cones_criterion, aggregate, not_normalized_maxmin_criterion

from desdeo.tools.scalarization import add_asf_diff, add_guess_sf_diff, add_stom_sf_diff, add_asf_generic_diff

from desdeo.tools.utils import guess_best_solver, PyomoIpoptSolver, NevergradGenericSolver

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
    numpy_array_to_objective_dict,
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

    rp1_np = [0.6, 0.4]
    rp2_np = [0.1, .1]
    rp3_np = [0.4, 0.6]

    result = [0.1, 0.1]

    cip = np.array([1., 1.])

    k = 2
    q = 3
    pa = "maxmin"
    all_rps = np.array([rp1_np, rp2_np, rp3_np])
    print(all_rps)
    GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("MAXMIN GRP", GRP)

    assert len(GRP) == 2
    assert not np.isnan(GRP).any()  # "GRP should consist of real numbers."
    assert np.allclose(GRP, np.array([0.1, 0.1]))

    pa = "maxmin_ext"
    GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("MAXMIN ext GRP", GRP)

    assert len(GRP) == 2
    assert not np.isnan(GRP).any()
    assert np.allclose(GRP, np.array([0.1, 0.1]))

    pa = "maxmin_cones"
    GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("MAXMIN cones GRP", GRP)

    assert len(GRP) == 2
    assert not np.isnan(GRP).any()
    assert np.allclose(GRP, np.array([0.1, 0.1]))

    pa = "maxmin_cones_ext"
    GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    print("MAXMIN cones ext GRP", GRP)

    assert len(GRP) == 2
    assert not np.isnan(GRP).any()
    assert np.allclose(GRP, np.array([0.1, 0.1]))

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

@pytest.mark.pref_agg
def test_maxmin_cones_same_scales():
    """Test maxmin with same scaled objective functions"""

    n_variables = 30
    n_objectives = 2
    problem = zdt2(n_variables)
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    rp1_np = [0.5, 0.8]
    rp2_np = [0.3, 0.6]
    rp3_np = [0.8, 0.4]
    all_rps = np.array([rp1_np, rp2_np, rp3_np])

    cip = nadir
    # cip = np.array([0.7, 0.8]) # if wanting to use some other cip

    k = n_objectives
    q = 3
    pa = "maxmin_cones"

    # grp = [rp1, rp2, rp3]
    # print(all_rps)
    # GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    # print("MAXMIN GRP", GRP)
    # assert len(GRP) == n_objectives
    # assert not np.isnan(GRP).any()  # "GRP should consist of real numbers."

    R = [.573, 0.533]  # DM1 should be max happy, DM2-3 also kind of.
    values = [maxmin_cones_criterion(all_rps[j, :], cip, R) for j in range(q)]
    print("rudolf code values", values)

    R = [.5, .8]  # testing that if GRP would be at DM1's point, he would have max value and others not
    values = [maxmin_cones_criterion(all_rps[j, :], cip, R) for j in range(q)]  # should be [-1, -1, 0]
    print("values", values)
    assert np.isclose(values[0], 1)  # should be 1 for DM that has RP at the same point as suggested.
    assert values[1] < 1
    assert values[2] < 1

    R = [.3, .6]  # DM2 and DM1 should be max happy.
    values = [maxmin_cones_criterion(all_rps[j, :], cip, R) for j in range(q)]  # should be [0, x, x]

    print("values", values)
    assert values[0] > 1
    assert np.isclose(values[1], 1)  # should be 1 for DM that has RP at the same point as suggested.
    assert values[2] < 1

    R = [.8, 0.4]  # DM3 and DM1 should be max happy.
    values = [maxmin_cones_criterion(all_rps[j, :], cip, R) for j in range(q)]

    print("values", values)
    assert values[0] < 1
    assert values[1] < 1
    assert np.isclose(values[2], 1)  # should be 1 for DM that has RP at the same point as suggested.


@pytest.mark.skip
@pytest.mark.pref_agg
def test_maxmin_cones_extended():
    # TODO:
    """Test maxmin with same scaled objective functions"""

    n_variables = 30
    n_objectives = 2
    problem = zdt1(n_variables)
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    rp1_np = [0.5, 0.8]
    rp2_np = [0.3, 0.6]
    rp3_np = [0.8, 0.4]
    all_rps = np.array([rp1_np, rp2_np, rp3_np])

    cip = nadir
    # cip = np.array([0.7, 0.8]) # if wanting to use some other cip

    k = n_objectives
    q = 3
    pa = "maxmin_cones"

    # grp = [rp1, rp2, rp3]
    # print(all_rps)
    # GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)
    # print("MAXMIN GRP", GRP)
    # assert len(GRP) == n_objectives
    # assert not np.isnan(GRP).any()  # "GRP should consist of real numbers."

    R = [.573, 0.533]  # DM1 should be max happy, DM2-3 also kind of.
    values = [maxmin_cones_criterion(all_rps[j, :], cip, R) for j in range(q)]
    print("rudolf code values", values)

    R = [.5, .8]  # testing that if GRP would be at DM1's point, he would have max value and others not
    values = [maxmin_cones_criterion(all_rps[j, :], cip, R) for j in range(q)]  # should be [-1, -1, 0]
    print("values", values)
    assert np.isclose(values[0], 1)  # should be 1 for DM that has RP at the same point as suggested.
    assert values[1] < 1
    assert values[2] < 1

    R = [.3, .6]  # DM2 and DM1 should be max happy.
    values = [maxmin_cones_criterion(all_rps[j, :], cip, R) for j in range(q)]  # should be [0, x, x]

    print("values", values)
    assert values[0] > 1
    assert np.isclose(values[1], 1)  # should be 1 for DM that has RP at the same point as suggested.
    assert values[2] < 1

    R = [.8, 0.4]  # DM3 and DM1 should be max happy.
    values = [maxmin_cones_criterion(all_rps[j, :], cip, R) for j in range(q)]

    print("values", values)
    assert values[0] < 1
    assert values[1] < 1
    assert np.isclose(values[2], 1)  # should be 1 for DM that has RP at the same point as suggested.


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

    assert np.isclose(GRP[0], GRP2[0])
    assert np.isclose(GRP[1], GRP2[1])

@pytest.mark.skip
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

@pytest.mark.pref_agg
def test_maxmin_PO_same_scales():
    """Test maxmin with same scaled objective functions"""

    n_variables = 30
    n_objectives = 2
    problem = zdt2(n_variables)
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    cip = nadir
    cip_dict = {"f_1": cip[0], "f_2": cip[1]}

    rp1_np = [0.5, 0.5]
    rp2_np = [0., 0.9]
    rp3_np = [0.9, 0]
    rp_arr = np.array([rp1_np, rp2_np, rp3_np])

    # TODO: find projections of the RPs
    converted_prefs = []
    for rp in range(len(rp_arr)):
        dm_rp = cip - rp_arr[rp]  # convert to improvement direction
        # should convert numpy array (rp) of dm to dict.
        dm_rp = numpy_array_to_objective_dict(problem, dm_rp)
        # p, target = add_asf_diff(problem, f"target{rp}", dm_rp)
        p, target = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol=f"asf{rp}",
            reference_point=cip_dict,
            weights=dm_rp,
            reference_point_aug=cip_dict,
        )
        # for 1 RP
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        # xs = res.optimal_variables
        fs = res.optimal_objectives
        # print(fs)
        converted_prefs.append(fs)

    print(converted_prefs)

    all_rps = np.array([[col["f_1"], col["f_2"]] for col in converted_prefs])

    # cip = np.array([0.7, 0.8]) # if wanting to use some other cip

    k = n_objectives
    q = 3
    pa = "eq_maxmin"

    # grp = [rp1, rp2, rp3]
    # print(all_rps)
    GRP = find_GRP(all_rps, cip, k, q, ideal, rp_arr, pa)
    print("MAXMIN GRP", GRP)
    assert len(GRP) == n_objectives
    assert not np.isnan(GRP).any()  # "GRP should consist of real numbers."

    P = all_rps[0, :]  # testing that if GRP would be at DM1's point, he would have max value and others not
    # values = [not_normalized_maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]  # not_norm works bad
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]  # works
    print("values", values)
    assert values[0] == 1
    assert values[1] < 1
    assert values[2] < 1

    P = all_rps[1, :]  # testing that if GRP would be at DM1's point, he would have max value and others not
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]  # should be [0, x, x]

    print("values", values)
    assert values[0] > 1
    assert values[1] == 1
    assert values[2] < 1

    P = all_rps[2, :]  # testing that if GRP would be at DM1's point, he would have max value and others not
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]

    print("values", values)
    assert values[0] > 1
    assert values[1] < 1
    assert values[2] == 1


""" TODO write test for testing these too !
   # The s_m(R) for all DMs
    # TODO: standardization for maxmin (especially if not solving zdt1 and zdt2 only)
    if pref_agg_method == "maxmin" or pref_agg_method == "eq_maxmin":
        def fx(X):
            # TODO: recheck, if -1* should be inside maxmin terms or not!
            # max S. For scipy we need to convert to minimization.
            maxmin_terms = [maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            return -1*np.min(maxmin_terms)  # + rho * np.sum(maxmin_terms)
    if pref_agg_method == "maxmin_ext" or pref_agg_method == "eq_maxmin_ext":
        def fx(X):
            maxmin_terms = [maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            return -1*np.min(maxmin_terms) + rho * np.sum(maxmin_terms)
    if pref_agg_method == "maxmin_cones" or pref_agg_method == "eq_maxmin_cones":
        def fx(X):
            # all_s_m = [-maxmin_cones_criterion(cip, rps[j,:], X[:k]) for j in range(q)]
            # print(all_s_m)
            # s_m = np.max(all_s_m)
            # TODO: these are equivalent.. not sure if it matters which to use.. in otherwords, which is closest to the og formulation i guess
            all_s_m = [maxmin_cones_criterion(cip, rps[j, :], X[:k]) for j in range(q)]
            s_m = -1*np.min(all_s_m)
            return s_m
"""

@pytest.mark.pref_agg
def test_maxmin_S():
    n_variables = 30
    n_objectives = 2
    problem = zdt2(n_variables)
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    cip = nadir
    cip_dict = {"f_1": cip[0], "f_2": cip[1]}

    rp1_np = [0.5, 0.5]
    rp2_np = [0., 0.9]
    rp3_np = [0.9, 0]
    rp_arr = np.array([rp1_np, rp2_np, rp3_np])

    # TODO: find projections of the RPs
    converted_prefs = []
    for rp in range(len(rp_arr)):
        dm_rp = cip - rp_arr[rp]  # convert to improvement direction
        # should convert numpy array (rp) of dm to dict.
        dm_rp = numpy_array_to_objective_dict(problem, dm_rp)
        # p, target = add_asf_diff(problem, f"target{rp}", dm_rp)
        p, target = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol=f"asf{rp}",
            reference_point=cip_dict,
            weights=dm_rp,
            reference_point_aug=cip_dict,
        )
        # for 1 RP
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        # xs = res.optimal_variables
        fs = res.optimal_objectives
        # print(fs)
        converted_prefs.append(fs)

    print(converted_prefs)

    all_rps = np.array([[col["f_1"], col["f_2"]] for col in converted_prefs])

    # cip = np.array([0.7, 0.8]) # if wanting to use some other cip

    k = 2
    q = 3

    # grp = [rp1, rp2, rp3]
    # print(all_rps)
    # GRP = find_GRP(all_rps, cip, k, q, ideal, nadir, pa)

    # maxmin addtiive is minimizing Sm, and we are maximinzing S. so max S min Sm. -> maxmin.
    # scipy needs in minimization so, we should -1* S min Sm, right?
    P = all_rps[0, :]  # testing that if GRP would be at DM1's point, he would have max value and others not
    # values = [not_normalized_maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]  # not_norm works bad
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]  # works
    print("values", values)
    assert -1*np.max(values) == -1  #
    assert values[0] == 1
    assert values[1] < 1
    assert values[2] < 1

    P = all_rps[1, :]  # testing that if GRP would be at DM1's point, he would have max value and others not
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]  # should be [0, x, x]

    print("values", values)
    assert values[0] > 1
    assert values[1] == 1
    assert values[2] < 1

    P = all_rps[2, :]  # testing that if GRP would be at DM1's point, he would have max value and others not
    values = [maxmin_criterion(all_rps[j, :], cip, P) for j in range(q)]

    print("values", values)
    assert values[0] > 1
    assert values[1] < 1
    assert values[2] == 1


@pytest.mark.pref_agg
def test_maxmin_ext_S():
    pass

@pytest.mark.pref_agg
def test_maxmin_cones_S():
    n_variables = 30
    n_objectives = 2
    problem = zdt2(n_variables)
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))

    cip = nadir
    cip_dict = {"f_1": cip[0], "f_2": cip[1]}

    rp1_np = [0.5, 0.5]
    rp2_np = [0., 0.9]
    rp3_np = [0.9, 0]
    rp_arr = np.array([rp1_np, rp2_np, rp3_np])

    # TODO: find projections of the RPs
    converted_prefs = []
    for rp in range(len(rp_arr)):
        dm_rp = cip - rp_arr[rp]  # convert to improvement direction
        # should convert numpy array (rp) of dm to dict.
        dm_rp = numpy_array_to_objective_dict(problem, dm_rp)
        # p, target = add_asf_diff(problem, f"target{rp}", dm_rp)
        p, target = add_asf_generic_diff(  # use nondiff as default like in nautili
            problem,
            symbol=f"asf{rp}",
            reference_point=cip_dict,
            weights=dm_rp,
            reference_point_aug=cip_dict,
        )
        # for 1 RP
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        # xs = res.optimal_variables
        fs = res.optimal_objectives
        # print(fs)
        converted_prefs.append(fs)

    print(converted_prefs)

    all_rps = np.array([[col["f_1"], col["f_2"]] for col in converted_prefs])

    # cip = np.array([0.7, 0.8]) # if wanting to use some other cip
    k = 2
    q = 3

    # maxmin addtiive is minimizing Sm, and we are maximinzing S. so max S min Sm. -> maxmin.
    # scipy needs in minimization so, we should -1* S min Sm, right?
    P = all_rps[0, :]  # testing that if GRP would be at DM1's point, he would have max value and others not
    values = [maxmin_cones_criterion(all_rps[j, :], cip, P) for j in range(q)]  # Eli täältä puuttuisi - ?
    print(values)
    assert values[0] == 1
    assert values[1] < 1
    assert values[2] < 1

    # samalla logiikalla TODO maxmin cones
    # max S min Sm -> -1* S min Sm.

    P = all_rps[1, :]  # testing that if GRP would be at DM1's point, he would have max value and others not
    values = [maxmin_cones_criterion(all_rps[j, :], cip, P) for j in range(q)]  # Eli täältä puuttuisi - ?

    print("values", values)
    assert values[0] < 1
    assert values[1] == 1
    assert values[2] < 1

    # P = all_rps[2, :]  # testing that if GRP would be at DM1's point, he would have max value and others not
    P = [0, 0]  # testing that at ideal, all would have over 1 values
    values = [maxmin_cones_criterion(all_rps[j, :], cip, P) for j in range(q)]  # Eli täältä puuttuisi - ?
    print("values", values)
    assert values[0] > 1
    assert values[1] > 1
    assert values[2] > 1
