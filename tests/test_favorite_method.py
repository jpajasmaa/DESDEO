"""Tests related to the Favorite method."""

import pytest

import polars as pl
from desdeo.gdm import favorite_method
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint, choose_reference_point
from desdeo.gdm.favorite_method import (
    FairSolution,
    GPRMResults,
    IPR_Results,
    ZoomOptions,
    setup,
    FavOptions,
    FavResults,
    favorite_method,
    GPRMOptions,
    find_group_solutions,
    scale_rp,
    get_representative_set,
    IPR_Options
)
from desdeo.problem.testproblems.dtlz2_problem import dtlz2
from desdeo.problem import (
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint


@pytest.mark.favorite
def test_fairness_funcs():
    pass

@ pytest.mark.favorite
def test_zooming():
    pass

@pytest.mark.favorite
def test_setup():
    pass
    # dtlz2_problem = dtlz2(8, 3)


@pytest.mark.favorite
def test_favorite_method_iteration():

    dtlz2_problem = dtlz2(8, 3)
    ideal = dtlz2_problem.get_ideal_point()
    nadir = dtlz2_problem.get_nadir_point()

    evaluated_points = []
    most_preferred_solutions = {
        "DM1": {"f_1": 0.17049589013991726, "f_2": 0.17049589002331159, "f_3": 0.9704959056742878},
        "DM2": {"f_1": 0.17049589008489896, "f_2": 0.9704959056849697, "f_3": 0.17049589001752685},
        "DM3": {"f_1": 0.9704959057874635, "f_2": 0.17049588971897997, "f_3": 0.1704958898000307},
    }
    ipr_options = IPR_Options(
        most_preferred_solutions=most_preferred_solutions,
        num_initial_reference_points=10000,
        version="box",
    )
    grpmoptions = GPRMOptions(
        method_options=ipr_options,
    )
    zoomoptions = ZoomOptions(num_steps_remaining=5)

    fav_options = FavOptions(
        GPRMoptions=grpmoptions,
        zoom_options=zoomoptions,
        original_most_preferred_solutions=most_preferred_solutions,
        votes=None,  # none in the first iteration

    )

    grp_results = GPRMResults(
        raw_results=IPR_Results(evaluated_points=[
            _EvaluatedPoint(
                reference_point=most_preferred_solutions["DM1"],
                targets=most_preferred_solutions["DM1"],
                objectives=most_preferred_solutions["DM1"],
            )],
        ),
        solutions=None,
        outputs=pl.DataFrame(),
    )
    # just to satisfy types, lets take the DM1's MPS as the fair solution
    objs = most_preferred_solutions["DM1"]
    fair_sols = [FairSolution(
        objective_values=objs,
        fairness_criterion="no_regret",
        fairness_value=0.5,
    )]
    fav_results_e = [
        FavResults(
            FavOptions=fav_options,
            GPRMResults=grp_results,
            fair_solutions=fair_sols,
        )
    ]

    # favorite_method(problem=dtlz2_problem, options=fav_options, results_list=fav_results)  # results_list is None in the first iteration
    fav_results = favorite_method(problem=dtlz2_problem, options=fav_options, results_list=[])  # results_list is None in the first iteration

    # Test second iteration
    fav_options_2 = FavOptions(
        GPRMoptions=fav_options.GPRMoptions,
        zoom_options=fav_options.zoom_options,
        original_most_preferred_solutions=fav_options.original_most_preferred_solutions,
        votes={"DM1": 0, "DM2": 0, "DM3": 1},
    )
    fav_results_2 = favorite_method(problem=dtlz2_problem, options=fav_options_2, results_list=[fav_results])  # results_list is None in the first iteration
    print(fav_results_2)
