
"""Tests related to the Favorite method."""

import pytest
from desdeo.gdm.favorite_method import find_group_solutions, scale_rp, itp_mps
from desdeo.problem.testproblems.dtlz2_problem import dtlz2
from desdeo.problem import (
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)


@pytest.mark.favorite
def test_find_group_solutions():
    dtlz2_problem = dtlz2(8, 3)
    saved_solutions = []
    ideal = dtlz2_problem.get_ideal_point()
    nadir = dtlz2_problem.get_nadir_point()
    dtlz2_problem = dtlz2_problem.update_ideal_and_nadir(new_ideal=ideal, new_nadir=nadir)
    print(ideal)
    print(nadir)

    most_preferred_solutions = {
        'DM1': {'f_1': 0.17049589013991726, 'f_2': 0.17049589002331159, 'f_3': 0.9704959056742878},
        'DM2': {'f_1': 0.17049589008489896, 'f_2': 0.9704959056849697, 'f_3': 0.17049589001752685},
        'DM3': {'f_1': 0.9704959057874635, 'f_2': 0.17049588971897997, 'f_3': 0.1704958898000307}
    }
    # need to scale the mpses for fairness
    mps = {}
    for dm in most_preferred_solutions:
        mps.update({dm: scale_rp(dtlz2_problem, most_preferred_solutions[dm], ideal, nadir, False)})

    # RPs as array for methods to come
    rp_arr = []
    for i, dm in enumerate(mps):
        rp_arr.append(objective_dict_to_numpy_array(dtlz2_problem, mps[dm]).tolist())
    normalized_most_preferred_solutions = rp_arr
    optimizer = "itp"
    eval_points = itp_mps(dtlz2_problem, normalized_most_preferred_solutions)

    assert eval_points is not None

    solution_selector = "regret"
    aggregator = "sum"
    fair_sols = find_group_solutions(dtlz2_problem, eval_points, normalized_most_preferred_solutions, solution_selector, aggregator)
    print(fair_sols)

    assert fair_sols is not None


@pytest.mark.favorite
def test_shift_fakenadir():
    pass
