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
    """Set up the problem."""
    dtlz2_problem = dtlz2(8, 3)

    """ The most preferred solutions for each DM found in the learning phase."""
    most_preferred_solutions = {
        "DM1": {"f_1": 0.17049589013991726, "f_2": 0.17049589002331159, "f_3": 0.9704959056742878},
        "DM2": {"f_1": 0.17049589008489896, "f_2": 0.9704959056849697, "f_3": 0.17049589001752685},
        "DM3": {"f_1": 0.9704959057874635, "f_2": 0.17049588971897997, "f_3": 0.1704958898000307},
    }

    """Set up the options for IPR method, give them to get representative set function as options (GPRMOptions)."""
    """Then, fill ZoomOptions, in addition, finish all that is needed for FavOptions."""
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
        candidate_generation_options="utilitarian",
        zoom_options=zoomoptions,
        original_most_preferred_solutions=most_preferred_solutions,
        votes=None,  # none in the first iteration

    )

    """Call favorite_method for one iteration."""
    fav_results = favorite_method(problem=dtlz2_problem, options=fav_options, results_list=[])  # results_list is None in the first iteration

    # Test second iteration
    fav_options_2 = FavOptions(
        GPRMoptions=fav_options.GPRMoptions,
        candidate_generation_options="utilitarian",
        zoom_options=fav_options.zoom_options,
        original_most_preferred_solutions=fav_options.original_most_preferred_solutions,
        votes={"DM1": 0, "DM2": 0, "DM3": 1},
    )
    fav_results_2 = favorite_method(problem=dtlz2_problem, options=fav_options_2, results_list=[fav_results])
    print(fav_results_2)

    # test failing to not implemented error
    fav_options_to_fail = FavOptions(
        GPRMoptions=fav_options.GPRMoptions,
        candidate_generation_options="bestfairnesscriterion",
        zoom_options=fav_options.zoom_options,
        original_most_preferred_solutions=fav_options.original_most_preferred_solutions,
        votes={"DM1": 0, "DM2": 0, "DM3": 1},
    )
    with pytest.raises(NotImplementedError) as e_info:
        _ = favorite_method(problem=dtlz2_problem, options=fav_options_to_fail, results_list=[fav_results_2])
        print(e_info)
