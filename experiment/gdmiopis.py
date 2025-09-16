from typing import Literal

import warnings

import plotly.graph_objects as go

from desdeo.emo.hooks.archivers import NonDominatedArchive
from desdeo.emo.methods.EAs import nsga3, rvea
from desdeo.problem.testproblems import dtlz2

# Suppress Userwarnings from not having solvers installed
# This is not a problem, as we are not using any solvers in this example.
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from desdeo.problem import (
    Constraint,
    ConstraintTypeEnum,
    Problem,
    ScalarizationFunction,
    Variable,
    VariableTypeEnum,
)
from desdeo.gdm.gdmtools import dict_of_rps_to_list_of_rps
from desdeo.tools.utils import (
    flip_maximized_objective_values,
    get_corrected_ideal,
    get_corrected_nadir,
)
from desdeo.tools.scalarization import (
    add_group_guess_diff,
    add_group_guess,
    add_group_stom_diff, 
    add_group_stom,
    add_group_nimbus_diff,
    add_group_nimbus,
    add_stom_sf_nondiff,
    add_guess_sf_nondiff,
)


# VERSION such as multiDM/IOPIS
def add_iopis_funcs(
    problem: Problem,
    reference_points: dict[str, dict[str, float]],
    ideal: dict[str, float] | None = None,
    nadir: dict[str, float] | None = None,
    rho: float = 1e-6,
    delta: float = 1e-6,
) -> tuple[Problem, list[str]]:
    #symbols = ["gdmiopis_guess_diff", "gdmiopis_stom_diff", "gdm_nimbus"]
    symbols = ["DM1_stom", "DM2_stom", "DM3_stom"]
    _problem, _ = add_guess_sf_nondiff(
        problem=problem,
        symbol=symbols[0],
        reference_point=reference_points["DM1"],
        ideal=ideal,
        nadir=nadir,
        rho=rho,
    )

    _problem, _ = add_stom_sf_nondiff(
        problem=_problem,
        symbol=symbols[1],
        reference_point=reference_points["DM2"],
        ideal=ideal,
        delta=delta,
    )
    """
    _problem, _ = add_stom_sf_nondiff( 
        problem=problem,
        symbol=symbols[2],
        reference_point=reference_points["DM3"],
        ideal=ideal,
        delta=delta,
    )
    """
 
    return _problem, symbols

def add_gdmiopis_funcs(
    problem: Problem,
    reference_points: dict[str, dict[str, float]],
    ideal: dict[str, float] | None = None,
    nadir: dict[str, float] | None = None,
    rho: float = 1e-6,
    delta: float = 1e-6,
) -> tuple[Problem, list[str]]:
    #symbols = ["gdmiopis_guess_diff", "gdmiopis_stom_diff", "gdm_nimbus"]
    symbols = ["gdmiopis_guess", "gdmiopis_stom"]
    _problem, _ = add_group_guess(
        problem=problem,
        symbol=symbols[0],
        reference_points=dict_of_rps_to_list_of_rps(reference_points),
        #ideal=ideal,
        nadir=nadir,
        rho=rho,
    )

    _problem, _ = add_group_stom( # TODO: error, if providing two diff scalas to iopis, then probably alpha brings the error "
# ValueError: A symbol was provided for a new variable that already exists in the problem definition"
        problem=_problem,
        symbol=symbols[1],
        reference_points=dict_of_rps_to_list_of_rps(reference_points),
        ideal=ideal,
        delta=delta,
    )
    """
    _problem, _ = add_group_guess(
        problem=problem,
        symbol=symbols[2],
        reference_points=dict_of_rps_to_list_of_rps(reference_points),
        #ideal=ideal,
        nadir=nadir,
        rho=1e-12,
    )
    """
    return _problem, symbols

if __name__=="__main__":
    n_variables = 8
    n_objectives = 3

    problem = dtlz2(n_variables, n_objectives)
    #solver_options = IpoptOptions()
    ideal = {"f_1": 0.0, "f_2": 0.0, "f_3": 0.0}
    nadir = {"f_1": 1., "f_2": 1., "f_3": 1.}

    dms_rps = {
        "DM1": {"f_1": 0.99, "f_2": 0.5, "f_3": 0.03},  
        "DM2": {"f_1": 0.5, "f_2": 0.1, "f_3": 0.98},  
        "DM3": {"f_1": 0.02, "f_2": 0.9, "f_3": 0.5}, 
        "DM4": {"f_1": 0.3, "f_2": 0.2, "f_3": 0.5}, 
        "DM5": {"f_1": 0.2, "f_2": 0.26, "f_3": 0.145}, 
        "DM6": {"f_1": 0.4, "f_2": 0.9, "f_3": 0.8},  
    }

    #problem, syms = add_iopis_funcs(problem, dms_rps, ideal, nadir)
    problem, syms = add_gdmiopis_funcs(problem, dms_rps, ideal, nadir)
    solver, publisher = nsga3(problem=problem)

    result = solver()
    print(result.outputs.head())  # Contains the objective values, target values (values that are minimized), and constraints, and extra functions.
    print(len(result.outputs)) # scala func values are in here!

    # TODO: do profiling for the slow. take emo evaluator, 
    # return wrap_df is 15secs with 4 DMs and 113secs with 5 DMs

    fig = go.Figure(
        go.Scatter3d(
            x=result.outputs["f_1"],
            y=result.outputs["f_2"],
            z=result.outputs["f_3"],
            mode="markers",
            marker=dict(size=2),
            )
        )

    fig.add_scatter3d(
        x=[dms_rps["DM1"]["f_1"]],
        y=[dms_rps["DM1"]["f_2"]],
        z=[dms_rps["DM1"]["f_3"]],
        mode="markers",
        marker=dict(size=5),
        name="Reference point 1",
    )
    fig.add_scatter3d(
        x=[dms_rps["DM2"]["f_1"]],
        y=[dms_rps["DM2"]["f_2"]],
        z=[dms_rps["DM2"]["f_3"]],
        mode="markers",
        marker=dict(size=5),
        name="Reference point 2",
    )

    fig.add_scatter3d(
        x=[dms_rps["DM3"]["f_1"]],
        y=[dms_rps["DM3"]["f_2"]],
        z=[dms_rps["DM3"]["f_3"]],
        mode="markers",
        marker=dict(size=5),
        name="Reference point 3",
    )
    fig.add_scatter3d(
        x=[dms_rps["DM4"]["f_1"]],
        y=[dms_rps["DM4"]["f_2"]],
        z=[dms_rps["DM4"]["f_3"]],
        mode="markers",
        marker=dict(size=5),
        name="Reference point 4",
    )
    fig.add_scatter3d(
        x=[dms_rps["DM5"]["f_1"]],
        y=[dms_rps["DM5"]["f_2"]],
        z=[dms_rps["DM5"]["f_3"]],
        mode="markers",
        marker=dict(size=5),
        name="Reference point 5",
    )
    """
    fig.add_scatter3d(
        x=[dms_rps["DM6"]["f_1"]],
        y=[dms_rps["DM6"]["f_2"]],
        z=[dms_rps["DM6"]["f_3"]],
        mode="markers",
        marker=dict(size=5),
        name="Reference point 6",
    )
    """

    fig.show()