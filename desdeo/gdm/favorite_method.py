"""The Favorite method, a general method for group decision making in multiobjective optimization"""

from typing import Literal

import numpy as np
import plotly.express as ex
import polars as pl
import pydantic
from pydantic import ConfigDict, Field

from desdeo.emo import DesirableRangesOptions, emo_constructor, nsga3_options
from desdeo.emo.options.generator import ArchiveGeneratorOptions
from desdeo.emo.options.templates import EMOOptions
from desdeo.gdm.gdmtools import (
    agg_aspbounds,
    alpha_fairness,
    average_pareto_regret,
    dict_of_rps_to_list_of_rps,
    get_top_n_fair_solutions,
    get_top_n_fair_solutions_pl,
    inequality_in_pareto_regret,
    max_min_regret,
    min_max_regret,
    min_max_regret_no_impro,
    scale_rp,
)
from desdeo.gdm.voting_rules import majority_rule
from desdeo.gdm.preference_aggregation import find_GRP
from desdeo.mcdm.nautilus_navigator import calculate_navigation_point
from desdeo.problem import (
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)
from desdeo.problem.schema import Problem
from desdeo.tools import IpoptOptions, PyomoIpoptSolver
from desdeo.tools.GenerateReferencePoints import generate_points
from desdeo.tools.generics import EMOResult, SolverResults
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint, choose_reference_point
from desdeo.tools.scalarization import add_asf_diff

# TODO: udpate to handle the new format
def visualize_3d(options, evaluated_points, fair_sols, n):
    fig = ex.scatter_3d()

    # Add reference points
    chosen_refps = pl.DataFrame([point.reference_point for point in evaluated_points])
    # rescale reference points
    chosen_refps = chosen_refps.with_columns(
        [
            (pl.col(obj) * (options.fake_nadir[obj] - options.fake_ideal[obj]) + options.fake_ideal[obj]).alias(obj)
            for obj in options.fake_ideal.keys()
        ]
    )

    # rescale reference points # TODO: need to fix this
    # chosen_refps = chosen_refps.with_columns(
    #    [(pl.col(obj) * (problem.nadir[obj] - problem.ideal[obj]) + problem.ideal[obj]).alias(obj) for obj in problem.ideal.keys()]
    # )
    fig = fig.add_scatter3d(
        x=chosen_refps["f_1"].to_numpy(),
        y=chosen_refps["f_2"].to_numpy(),
        z=chosen_refps["f_3"].to_numpy(),
        name="Reference Points",
        mode="markers",
        marker_symbol="circle",
        opacity=0.8,
    )
    # TODO: add color for the agg. sum fairness value for each solution. Find in some bhupinder notebook how it was done exactly.
    # Add front
    front = pl.DataFrame([point.objectives for point in evaluated_points])
    fig = fig.add_scatter3d(
        x=front["f_1"].to_numpy(),
        y=front["f_2"].to_numpy(),
        z=front["f_3"].to_numpy(),
        mode="markers",
        name="Front",
        marker_symbol="circle",
        opacity=0.9,
    )
    fig = fig.add_scatter3d(
        x=[options.fake_ideal["f_1"]],
        y=[options.fake_ideal["f_2"]],
        z=[options.fake_ideal["f_3"]],
        mode="markers",
        name="fake_ideal",
        marker_symbol="diamond",
        opacity=0.9,
    )
    fig = fig.add_scatter3d(
        x=[options.fake_nadir["f_1"]],
        y=[options.fake_nadir["f_2"]],
        z=[options.fake_nadir["f_3"]],
        mode="markers",
        name="fake_nadir",
        marker_symbol="diamond",
        opacity=0.9,
    )
    DMs = options.most_preferred_solutions.keys()
    for dm in DMs:
        fig = fig.add_scatter3d(
            x=[options.most_preferred_solutions[dm]["f_1"]],
            y=[options.most_preferred_solutions[dm]["f_2"]],
            z=[options.most_preferred_solutions[dm]["f_3"]],
            mode="markers",
            name=dm,
            marker_symbol="square",
            opacity=0.9,
        )

    # TODO: improve, this only works if n == 1
    fair_crits = [fair_sols[i].fairness_criterion for i in range(len(fair_sols))]
    for i, fc in enumerate(fair_crits):
        fig = fig.add_scatter3d(
            x=[fair_sols[i].objective_values["f_1"]],
            y=[fair_sols[i].objective_values["f_2"]],
            z=[fair_sols[i].objective_values["f_3"]],
            mode="markers",
            name=fc,
            marker_symbol="x",
            opacity=0.9,
        )

    """
    # Add maxfair points
    fig = fig.add_scatter3d(
        x=fair_sols[:n, 0],
        y=fair_sols[:n, 1],
        z=fair_sols[:n, 2],
        mode="markers", name="min_regret_no_impro", marker_symbol="x", opacity=0.9)

    fig = fig.add_scatter3d(
        x=fair_sols[n:n+n, 0],
        y=fair_sols[n:n+n, 1],
        z=fair_sols[n:n+n, 2],
        mode="markers", name="min_regret", marker_symbol="x", opacity=0.9)

    fig = fig.add_scatter3d(
        x=fair_sols[2*n:3*n, 0],
        y=fair_sols[2*n:3*n, 1],
        z=fair_sols[2*n:3*n, 2],
        mode="markers", name="avg_regret", marker_symbol="x", opacity=0.9)

    fig = fig.add_scatter3d(
        x=fair_sols[3*n:4*n, 0],
        y=fair_sols[3*n:4*n, 1],
        z=fair_sols[3*n:4*n, 2],
        mode="markers", name="gini_regret", marker_symbol="x", opacity=0.9)

    fig = fig.add_scatter3d(
        x=fair_sols[4*n:5*n, 0],
        y=fair_sols[4*n:5*n, 1],
        z=fair_sols[4*n:5*n, 2],
        mode="markers", name="utilitarian", marker_symbol="x", opacity=0.9)
    fig = fig.add_scatter3d(
        x=fair_sols[5*n:6*n, 0],
        y=fair_sols[5*n:6*n, 1],
        z=fair_sols[5*n:6*n, 2],
        mode="markers", name="nash", marker_symbol="x", opacity=0.9)

    fig = fig.add_scatter3d(
        x=fair_sols[6*n:, 0],
        y=fair_sols[6*n:, 1],
        z=fair_sols[6*n:, 2],
        mode="markers", name="cones", marker_symbol="x", opacity=0.9)
    """
    """
    fig.update_layout(
        scene={
            "xaxis_title": problem.problem.objectives[0].name,
            "yaxis_title": problem.problem.objectives[1].name,
            "zaxis_title": problem.problem.objectives[2].name,
        })
    # NOW can see everything!
    fig.update_layout(scene=dict(
        xaxis_range=(chosen_refps["f_1"].min(), ideal["f_1"]),
        yaxis_range=(chosen_refps["f_2"].min(), ideal["f_2"]),
        zaxis_range=(chosen_refps["f_3"].min(), ideal["f_3"]),
    ))
    # change this to ideal and nadir
    fig.update_layout(scene=dict(
        xaxis_range=(nadir["f_1"], ideal["f_1"]),
        yaxis_range=(nadir["f_2"], ideal["f_2"]),
        zaxis_range=(nadir["f_3"], ideal["f_3"]),
    ))
    """
    fig.layout.scene.camera.projection.type = "orthographic"
    # fig.write_html(f"/home/jp/tyot/mop/desdeo/DESDEO/experiment/code/generic_method/dtlz2.html")
    # fig.write_image(f"/home/jp/tyot/mop/desdeo/DESDEO/experiment/code/generic_method/test.pdf")
    fig.update_layout(
        autosize=False,
        width=1200,
        height=1200,
    )

    fig.show(renderer="browser")


class IPR_Options(pydantic.BaseModel):
    """Options specific to iterative_pareto_representer applied with the favorite method."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    num_initial_reference_points: int = Field(default=10000, ge=1)
    """Big number"""
    version: Literal["convex_hull", "box"] = "convex_hull"
    (
        """Version "convex_hull": evaluate in the convex hull of MPS."""
        """ Version "box": evaluate in the box of fake_ideal and fake_nadir."""
    )
    most_preferred_solutions: dict[str, dict[str, float]] | None = None
    """Most preferred solutions of the decision makers. Should be filled in by code, not by user."""


class GPRMOptions(pydantic.BaseModel):
    """Pydantic model to contain options for the `get_representative_set` function."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    method_options: IPR_Options | None = Field(default_factory=IPR_Options)
    """Options specific to the selected method. None for EMO"""
    fake_ideal: dict[str, float] | None = None
    """Fake ideal point. Should be filled in by code, not by user."""
    fake_nadir: dict[str, float] | None = None
    """Fake nadir point. Should be filled in by code, not by user."""
    num_points_to_evaluate: int = Field(default=100, ge=1)
    """Number of points to evaluate in the IPR method, or population size in EMO methods."""


class IPR_Results(pydantic.BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    evaluated_points: list[_EvaluatedPoint]  # I dont know how to use this properly.
    # fair_solutions: list[dict[str, float]]  # TODO: decide the type
    # fair_solutions: list  # TODO: decide the type
    # fairness_metrics: list[str]  # string to indicate the fair solution type


class GPRMResults(pydantic.BaseModel):
    """Pydantic model to contain results from the `get_representative_set` function."""

    model_config = ConfigDict(use_attribute_docstrings=True, arbitrary_types_allowed=True)

    raw_results: IPR_Results | EMOResult
    """Raw results from the selected method."""
    solutions: pl.DataFrame | None
    """DataFrame containing the evaluated solutions (inputs)."""
    outputs: pl.DataFrame
    """DataFrame containing the evaluated outputs."""


# IMPLEMENt below to tag the solutions in find fair solutions?
class FairSolution(pydantic.BaseModel):
    objective_values: dict[str, float]
    fairness_criterion: str
    fairness_value: float  # could add fairness value of this solution.


# get a problem
class ProblemWrapper:
    def __init__(self, problem: Problem, fake_ideal: dict[str, float], fake_nadir: dict[str, float]):
        """Initialize the problem wrapper for a DESDEO problem.

        Args:
        problem = a DESDEO problem
        fake_ideal
        fake_nadir
        """
        self.problem = problem
        self.ideal, self.nadir = fake_ideal, fake_nadir
        self.problem = problem.update_ideal_and_nadir(new_ideal=self.ideal, new_nadir=self.nadir)
        self.evaluated_points: list[_EvaluatedPoint] = []

    # TODO: set solver
    def solve(self, scaled_refp: np.ndarray) -> list[_EvaluatedPoint]:
        refp = {
            obj: val * (self.nadir[obj] - self.ideal[obj]) + self.ideal[obj]
            for obj, val in zip(self.ideal.keys(), scaled_refp)
        }
        scaled_problem, target = add_asf_diff(self.problem, "target", refp)
        solver = PyomoIpoptSolver(scaled_problem)
        # solver = guess_best_solver(scaled_problem)
        results = solver.solve(target)
        objs = results.optimal_objectives
        scaled_objs = {obj: (objs[obj] - self.ideal[obj]) / (self.nadir[obj] - self.ideal[obj]) for obj in objs.keys()}
        self.evaluated_points.append(
            _EvaluatedPoint(
                reference_point=dict(zip(self.ideal.keys(), scaled_refp)), targets=scaled_objs, objectives=objs
            )
        )
        return self.evaluated_points

    """
     TODO: would need some sort of setter or other tool to set a desired solver for ProblemWrapper
    def set_solver():
        pass
    """


def helper(fairness_criterion):
    """
    TODO: name properly
    would be useful (or at least help with readability of the code) to have helper function that would work as below. Not sure how to use with alpha_fairness that requires different parameters..
    """
    match fairness_criterion:
        case "utilitarian":
            return alpha_fairness  # TODO: adjust to the new fucntion call and param=s
        case "nash":
            return alpha_fairness
        case "regret":
            return min_max_regret_no_impro

        case _:
            return "Not one of the implemented fairness_criteria."


def find_group_solutions(problem: Problem, solutions: pl.DataFrame, targets, most_preferred_solutions: dict[str, dict[str, float]]):
    """Find n fair group solutions according to different fairness criteria.

    Assumes everything has been properly converted to minimization already.
    solutions contains the solutions non-normalized. Targets contains solutions in IPR normalized version.

        Returns a list of FairSolutions

    Args:
        problem: DESDEO Problem object

    Returns:
        tuple: (list of FairSolution objects, dict of regret values for each solution)
    """

    # If I will deal with polars.dataframes I do not need to convert stuff to numpy arrayss here.

    # TODO: Normalize MPSes
    normalized_mpses = most_preferred_solutions

    # convert to numpy array for numba in UFs
    normalized_mpses_arr = []
    for i, dm in enumerate(normalized_mpses):
        normalized_mpses_arr.append(objective_dict_to_numpy_array(problem, normalized_mpses[dm]).tolist())

    # TODO: change this to match how many FairSolution we want
    # For now, we just get a single one
    ranking = alpha_fairness(targets, normalized_mpses_arr, alpha=0.0)  # utilitarian
    # ranking = alpha_fairness(targets, normalized_mpses_arr, alpha=1)  # nash
    # ranking = min_max_regret_no_impro(targets, normalized_mpses_arr)  # minmax regret no improvements

    print(ranking)

    """
    fairness_criteria = ["utilitarian", "nash", "minmax"]
    for i in range(len(fairness_criteria)):
        fairness_ranking = []
        fairness_function = helper(fairness_criterion=fairness_criteria[i])
        if i == "utilitarian":
            fairness_ranking.append(fairness_function(targets, normalized_mpses_arr, alpha=0.0))
        if i == "nash":
            fairness_ranking.append(fairness_function(targets, normalized_mpses_arr, alpha=1.0))
        else:
            fairness_ranking.append(fairness_function(targets, normalized_mpses_arr))
        # TODO: figure out how to configure the helper
        fairness_function = helper(fairness_criterion=fairness_criteria[i])

        print(fairness_function)
        fairness_ranking = fairness_function(targets, normalized_mpses_arr, alpha=1.0)
        print("fair sols ranmking", fairness_ranking)
        """

    print("SOLUTIONS:\r\n", solutions)

    # if solutions are dict, we need to convert them to array for top_n_fair_solutions
    eval_sols_in_objs = []
    # evaluated_points["targets"]
    for i in range(len(solutions)):
        eval_sols_in_objs.append(objective_dict_to_numpy_array(problem, solutions[i]))  # HOW THE HELL YOU ARE NOT A LIST???
        # eval_sols_in_objs.append(objective_dict_to_numpy_array(problem, evaluated_points[i].objectives))
        eval_sols_in_objs = np.stack(eval_sols_in_objs)

    # Return top fair solutions
    ranking_r, ranking_i = get_top_n_fair_solutions(solutions, ranking, 1)
    print("fairness rankings")
    print(ranking_r, ranking_i)

    FairSolutions_arr = []
    # Loop
    FairSolutions_arr.append(
        FairSolution(
            objective_values=numpy_array_to_objective_dict(problem, ranking_r[0]),
            fairness_criterion="test",
            fairness_value=ranking_i[0],
        )
    )
    # TODO: regret values needed or?
    regret_values = {
        "mean": [],
    }

    return FairSolutions_arr, regret_values

# TODO: this to be re-written after UFs are implemented as polars expressions
def find_group_solutions_mess(
    problem: Problem, evaluated_points: pl.DataFrame, mps: list[np.ndarray], selectors: list[str], n: int = 1
) -> tuple[list[FairSolution], list[float]]:
    """Find n fair group solutions according to different fairness criteria.

    Assumes everything has been properly converted to minimization already.
    TODO: extend to return any amount of fair solutions with some order such as:
        1. Maxmin-cones solution
        2. regret sum
        3. regret maxmin

        Maxmin-cones is a singular, for regret sum and regret maxmin can return multiple solutions.

        mps must be np array for the below code !

        Returns a list of FairSolutions

    Args:
        problem: DESDEO Problem object
        evaluated_points: DataFrame of objective values of evaluated points
        mps: list of most preferred solutions as numpy arrays
        selectors: list of strings indicating which fairness metrics to apply
        n: number of fair solutions to return for each fairness metric

    Returns:
        tuple: (list of FairSolution objects, dict of regret values for each solution)
    """
    # TODO: normalize the MPSses. TO its own function?
    norm_mps = mps

    norm_eval_sols = []
    # evaluated_points["targets"]
    for i in range(len(evaluated_points)):
        norm_eval_sols.append(objective_dict_to_numpy_array(problem, evaluated_points[i].targets))
    norm_eval_sols = np.stack(norm_eval_sols)

    eval_sols_in_objs = []
    # evaluated_points["targets"]
    for i in range(len(evaluated_points)):
        eval_sols_in_objs.append(objective_dict_to_numpy_array(problem, evaluated_points[i].objectives))
    eval_sols_in_objs = np.stack(eval_sols_in_objs)

    norm_mps_arr = np.array(norm_mps)  # P needs to be np.array for solve_UFs

    #  apply the selectors string to determine which of the fairness_metrics to apply
    print(selectors)
    # add maxmin-cones, maybe maxmin also?

    #  would be convenient to get all the parameter to some parameter dictionary etc that I can apply.
    # Look for inspiration in Bhupinder's EMO stuff?
    # Regret UFs, no achievement for aspiration. Sum over DMs.
    #
    # TODO: here the UF calculations. Start with something simple to get the structure right?
    # Regret UFs, no achievement for aspiration. Take the DM that is worst-off. Min()
    min_no_regrets = min_max_regret_no_impro(norm_eval_sols, norm_mps_arr)
    print("min regrets no impro:", min(min_no_regrets), max(min_no_regrets))

    # Regret UFs, achievement for aspiration.
    min_regrets = min_max_regret(norm_eval_sols, norm_mps_arr)
    print("chebyshev regret:", min(min_regrets), max(min_regrets))

    # Avererage Pareto regret
    avg_regrets = average_pareto_regret(norm_eval_sols, norm_mps_arr)
    print("avg pareto regrets:", min(avg_regrets), max(avg_regrets))

    # inequality in pareto regret
    gini_regrets = inequality_in_pareto_regret(norm_eval_sols, norm_mps_arr)
    print("gini pareto regrets:", min(gini_regrets), max(gini_regrets))

    # maxmin in pareto regret
    # maxmin_regrets = max_min_regret(norm_eval_sols, norm_mps_arr)
    # print("maxmin regrets:", maxmin_regrets)

    # alpha fairness with pareto regret
    # alpha_regrets = alpha_fairness(norm_eval_sols, norm_mps_arr, alpha=0)
    # print("alpha regrets:", alpha_regrets)

    # lets test different values for alpha
    # min_no_regrets = alpha_fairness(norm_eval_sols, norm_mps_arr, alpha=0.0)  # should match to the utilitarian fairness
    utilitarian = alpha_fairness(norm_eval_sols, norm_mps_arr, alpha=0.0)  # some midway between nash and utilitarian
    print("utilitarian regrets no impro:", min(utilitarian), max(utilitarian))
    # avg_regrets = alpha_fairness(norm_eval_sols, norm_mps_arr, alpha=1)  # should match Nash (proportionally fair)
    nash = alpha_fairness(norm_eval_sols, norm_mps_arr, alpha=1)  # alpha goes to infinity matches maxmin fair.
    print("nash regrets no impro:", min(nash), max(nash))

    # Regret UFs, no achievement for aspiration. Take the DM that is best-off. Max()
    #
    # TODO: to get fairness values for each solution, this topfair should also return the index.
    min_r_no, min_no_i = get_top_n_fair_solutions(eval_sols_in_objs, min_no_regrets, n)
    util_r, util_i = get_top_n_fair_solutions(eval_sols_in_objs, utilitarian, n)
    min_r, min_i = get_top_n_fair_solutions(eval_sols_in_objs, min_regrets, n)
    avg_r, avg_i = get_top_n_fair_solutions(eval_sols_in_objs, avg_regrets, n)
    gini_r, gini_i = get_top_n_fair_solutions(eval_sols_in_objs, gini_regrets, n)
    nash_r, nash_i = get_top_n_fair_solutions(eval_sols_in_objs, nash, n)
    # TODO: find out what is the bug here. returns arrays
    # alpha_r = get_top_n_fair_solutions(eval_sols_in_objs, alpha_regrets, n)

    # SOLVE MAXMIN CONES
    # TODO: all this needs to be inside its own function after find_GRP has been rewritten for usability
    # for dtlz2
    cip = np.array(
        [np.max(norm_mps_arr[:, 0]) + 0.001, np.max(norm_mps_arr[:, 1]) + 0.001, np.max(norm_mps_arr[:, 2]) + 0.001]
    )
    ideal_arr = np.array([np.min(norm_mps_arr[:, 0]), np.min(norm_mps_arr[:, 1]), np.min(norm_mps_arr[:, 2])])

    # for river_pollution
    # cip = np.array([np.max(norm_mps_arr[:, 0]) + 0.001, np.max(norm_mps_arr[:, 1]) + 0.001,
    #               np.max(norm_mps_arr[:, 2]) + 0.001, np.max(norm_mps_arr[:, 3]) + 0.001])
    # ideal_arr = np.array([np.min(norm_mps_arr[:, 0]), np.min(norm_mps_arr[:, 1]), np.min(norm_mps_arr[:, 2]), np.min(norm_mps_arr[:, 3])])

    print(norm_mps_arr)
    print("starting")
    print(cip)
    print(ideal_arr)

    k = len(problem.objectives)
    q = len(norm_mps_arr)
    pa = "eq_maxmin_cones"
    # pa = "eq_maxmin"
    all_rps = norm_mps_arr
    GRP, s_values = find_GRP(all_rps, cip, k, q, ideal_arr, all_rps, pa)
    print("SVALUES", s_values)
    # GRP = GRP - cip
    print("MAXMIN cones GRP", GRP)
    # Find PO solution with conesGRP
    # GRP_dict = {"f_1": GRP[0], "f_2": GRP[1], "f_3": GRP[2], "f_4": GRP[3]} # for river pollution,
    GRP_dict = {"f_1": GRP[0], "f_2": GRP[1], "f_3": GRP[2]}

    p, target = add_asf_diff(
        problem,
        symbol=f"asf",
        reference_point=GRP_dict,
    )
    # scaled_problem, target = add_asf_diff(self.problem, "asf", refp)
    solver = PyomoIpoptSolver(p)
    res = solver.solve(target)
    fs = res.optimal_objectives
    GRP_po = objective_dict_to_numpy_array(problem, fs)

    # SOLVE MAXMIN
    pa = "eq_maxmin"
    GRPmm, mm_s_values = find_GRP(all_rps, cip, k, q, ideal_arr, all_rps, pa)
    print("SVALUES", mm_s_values)
    # GRP = GRP - cip
    print("MAXMIN GRP", GRPmm)
    # Find PO solution with conesGRP
    # GRPmm_dict = {"f_1": GRPmm[0], "f_2": GRPmm[1], "f_3": GRPmm[2], "f_4": GRPmm[3]}
    GRPmm_dict = {"f_1": GRPmm[0], "f_2": GRPmm[1], "f_3": GRPmm[2]}

    p, target = add_asf_diff(
        problem,
        symbol=f"asf",
        reference_point=GRPmm_dict,
    )
    # scaled_problem, target = add_asf_diff(self.problem, "asf", refp)
    solver = PyomoIpoptSolver(p)
    resmm = solver.solve(target)
    fsmm = resmm.optimal_objectives
    GRPmm_po = objective_dict_to_numpy_array(problem, fsmm)

    # top_fair = np.concatenate((min_r_no, min_r, avg_r, gini_r, util_r, nash_r, [GRP_po]))
    top_fair = np.concatenate((min_r_no, min_r, avg_r, gini_r, util_r, nash_r, [GRP_po], [GRPmm_po]))
    # TODO: smarter way instead of this monstrosity
    # Adds the top fair solution's fairness value to a np.array of lists, similar to top_fair, to be added to FairSolution objects.
    top_fair_values = np.concatenate(
        (
            [min_no_regrets[min_no_i[0]]],
            [min_regrets[min_i[0]]],
            [avg_regrets[avg_i[0]]],
            [gini_regrets[gini_i[0]]],
            [utilitarian[util_i[0]]],
            [nash[nash_i[0]]],
            [np.max(s_values)],
            [np.max(mm_s_values)],
        )
    )
    # top_fair = np.stack(top_fair)
    fairness_criteria = [
        "min_no",
        "min",
        "avg",
        "gini",
        "util",
        "nash",
        "cones",
        "maxmin",
    ]  # need to get this elsewhere
    FairSolutions_arr = []
    for i, fair_solution in enumerate(top_fair):
        fair_sol = FairSolution(
            objective_values=numpy_array_to_objective_dict(problem, fair_solution),
            fairness_criterion=fairness_criteria[i],
            fairness_value=top_fair_values[i],
        )
        FairSolutions_arr.append(fair_sol)

    regret_values = {
        "min_no": min_no_regrets,
        "min": min_regrets,
        "avg": avg_regrets,
        "gini": gini_regrets,
        "util": utilitarian,
        "nash": nash,
        "cones": GRP_po,
        "maxmin": GRPmm_po,
    }

    print(FairSolutions_arr)

    return FairSolutions_arr, regret_values


def shift_points(
    problem: Problem, most_preferred_solutions, group_preferred_solution: dict[str, float], steps_remaining
):
    """Calls calculate_navigation_point to shift individual most most_preferred_solutions. Then projects them to Pareto front to return as most preferred solutions"""

    shifted_mps = {}
    for dm in most_preferred_solutions:
        shifted_point = calculate_navigation_point(
            problem, most_preferred_solutions[dm], group_preferred_solution, steps_remaining
        )
        p, target = add_asf_diff(
            problem,
            symbol="asf",
            reference_point=shifted_point,
        )
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        shifted_mps.update({dm: res.optimal_objectives})

    return shifted_mps


# evaluated_points: list(_EvaluatedPoint) = Field("The evaluated points.")


def get_representative_set_IPR(problem: Problem, options: GPRMOptions, results_list: list[GPRMResults]) -> GPRMResults:
    """Get the representative set according to IPR_Options."""
    if not isinstance(options.method_options, IPR_Options):
        raise TypeError("Expected IPR_Options for IPR method.")

    evaluated_points = []

    # Normalize mps for fairness and IPR. Convert to array for now.
    mps = {}
    for dm in options.method_options.most_preferred_solutions:
        mps.update(
            {
                dm: scale_rp(
                    problem,
                    options.method_options.most_preferred_solutions[dm],
                    options.fake_ideal,
                    options.fake_nadir,
                    False,
                )
            }
        )

    # RPs as array for methods to come
    rp_arr = []
    for _, dm in enumerate(mps):
        rp_arr.append(objective_dict_to_numpy_array(problem, mps[dm]).tolist())

    dims = len(problem.get_nadir_point())

    # get the representative set
    # set n or the possibilities of n according to the num points to evaluate
    for n in [options.num_points_to_evaluate, options.num_points_to_evaluate / 2, 10]:
        try:
            # generate points in the convex hull of RPs or fake_ideal and fake_nadir
            if options.method_options.version == "convex_hull":
                _, refp = generate_points(
                    num_points=options.method_options.num_initial_reference_points,
                    num_dims=dims,
                    reference_points=rp_arr,
                )
            else:
                _, refp = generate_points(
                    num_points=options.method_options.num_initial_reference_points, num_dims=dims, reference_points=None
                )

            num_runs = n
            wrapped_problem = ProblemWrapper(problem, fake_ideal=options.fake_ideal, fake_nadir=options.fake_nadir)
            for i in range(num_runs):
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"Run {i + 1}/{num_runs}")
                reference_point, _ = choose_reference_point(refp, evaluated_points)
                evaluated_points = wrapped_problem.solve(reference_point)
            break
        except Exception:
            break

    ipr_res = IPR_Results(evaluated_points=evaluated_points)

    results = GPRMResults(
        raw_results=ipr_res,
        solutions=None,
        outputs=pl.DataFrame([point.objectives for point in evaluated_points]),
    )

    return results


def get_representative_set_EMO(problem: Problem, options: GPRMOptions, results_list: list[GPRMResults]) -> GPRMResults:
    """Get the representative set according to EMOOptions.

    Args:
        problem: DESDEO Problem object
        options: EMOOptions
        results_list: list of previous EMOResult objects

    Returns:

    """
    opts = nsga3_options()
    dr_opts = DesirableRangesOptions(
        aspiration_levels=options.fake_ideal,
        reservation_levels=options.fake_nadir,
        method="DF transformation",
        desirability_levels=(0.999, 0.001),
    )
    opts.preference = dr_opts
    opts.template.generator.n_points = options.num_points_to_evaluate
    opts.template.selection.reference_vector_options.number_of_vectors = options.num_points_to_evaluate
    opts.template.selection.invert_reference_vectors = True
    if results_list:
        opts.template.generator = ArchiveGeneratorOptions(
            solutions=results_list[-1].solutions.select([var.symbol for var in problem.get_flattened_variables()]),
            outputs=results_list[-1].outputs.select([obj.name for obj in problem.objectives]),
        )
    solver, extras = emo_constructor(problem=problem, emo_options=opts)
    res = solver()
    archive_results = extras.archive.results
    var_cols = archive_results.optimal_variables.columns
    obj_cols = archive_results.optimal_outputs.columns
    solutions = pl.concat([archive_results.optimal_variables, archive_results.optimal_outputs], how="horizontal")

    for obj in problem.objectives:
        if obj.maximize:
            solutions = solutions.filter(
                (
                    pl.col(obj.symbol) >= options.fake_nadir[obj.symbol]
                    # & (pl.col(obj.symbol) <= options.fake_ideal[obj.symbol])
                    # uncomment for stricter filtering
                )
            )
        else:
            solutions = solutions.filter(
                (
                    pl.col(obj.symbol) <= options.fake_nadir[obj.symbol]
                    # & (pl.col(obj.symbol) >= options.fake_ideal[obj.symbol])
                    # uncomment for stricter filtering
                )
            )
    return GPRMResults(
        raw_results=res,
        solutions=solutions.select(var_cols),
        outputs=solutions.select(obj_cols),
    )


def get_representative_set(problem: Problem, options: GPRMOptions, results_list: list[GPRMResults]) -> GPRMResults:
    """Get the representative set according to the given MethodOptions.

    Switches between IPR and EMO based on the type of options given.

    Args:
        problem: DESDEO Problem object
        options: MethodOptions, either IPR_Options or EMOOptions
        results_list: list of previous MethodResults objects

    Returns:
        tuple: (DataFrame of evaluated points, MethodResults)

    Raises:
        TypeError: If the provided MethodOptions type is invalid.
    """
    if isinstance(options.method_options, IPR_Options):
        return get_representative_set_IPR(problem, options, results_list)
        # return get_representative_set_IPR(problem, options, None)
    if options.method_options is None:
        return get_representative_set_EMO(problem, options, results_list)
        # return get_representative_set_EMO(problem, options, None)
    raise TypeError("Invalid MethodOptions type provided.")


class ZoomOptions(pydantic.BaseModel):
    """Pydantic model to contain options for zooming strategy."""

    method: Literal["nautilus"] = "nautilus"
    """Zooming method to use."""
    num_steps_remaining: int = Field(default=5, ge=1)
    """Number of remaining zooming steps. Determines step size. Must be positive integer."""


class FavOptions(pydantic.BaseModel):
    """Pydantic model to contain options for the favorite method."""

    GPRMoptions: GPRMOptions
    """Options for the representative set method. EMO and IPR supported."""
    candidate_generation_options: Literal["Not implemented yet"] = "Not implemented yet"
    """Options for generating candidate fair solutions. Support more options later."""
    zoom_options: ZoomOptions = Field(default_factory=ZoomOptions)
    """Options for the zooming strategy. Support more options later."""
    original_most_preferred_solutions: dict[str, dict[str, float]]
    """Dictionary of the original most preferred solutions for each decision maker."""
    votes: dict[str, int] | None = None
    (
        """The votes for each decision maker's most preferred solution."""
        """ The candidates are from `fair_solutions` in FavResults of the previous iteration."""
        """Not required for the first iteration."""
    )


class FavResults(pydantic.BaseModel):
    """Pydantic model to contain results from one iteration of the favorite method."""

    FavOptions: FavOptions
    """Options used in this iteration of the favorite method."""
    GPRMResults: GPRMResults
    """Results from the representative set method."""
    fair_solutions: list[FairSolution]
    """List of candidate fair solutions found in this iteration."""


def setup(problem: Problem, options: FavOptions, results_list: list[FavResults]) -> FavOptions:
    """Setup function for favorite method.

    Args:
        problem: DESDEO Problem object
        options: FavOptions for the favorite method.
        results_list: List of previous FavResults.

    Returns:
        FavOptions: Updated options for the favorite method.
    """
    options = options.model_copy()
    winner = None

    orig_mps = options.original_most_preferred_solutions
    orig_mps_list = dict_of_rps_to_list_of_rps(orig_mps)
    fake_ideal, fake_nadir = agg_aspbounds(orig_mps_list, problem)
    # first iteration
    if not results_list:  # noqa:SIM102
        if isinstance(options.GPRMoptions.method_options, IPR_Options):
            options.GPRMoptions.method_options.most_preferred_solutions = orig_mps
    if results_list:  # not the first iteration
        if options.votes is None:
            raise ValueError("Votes must be provided for iterations after the first.")
        # handle voting
        old_candidates = results_list[-1].fair_solutions
        print(results_list)
        print(old_candidates)
        votes = options.votes
        winner = majority_rule(votes=votes)
        print("WINNER", winner)
        if winner is None:
            raise ValueError("No winner could be determined from the votes provided.")
        winner = old_candidates[winner]
        fake_nadir = results_list[-1].FavOptions.GPRMoptions.fake_nadir
        if fake_nadir is None:
            raise ValueError("Previous fake_nadir is None, cannot proceed with zooming.")
        fake_nadir = calculate_navigation_point(
            problem=problem,
            previous_navigation_point=fake_nadir,
            reachable_objective_vector=winner.objective_values,
            number_of_steps_remaining=options.zoom_options.num_steps_remaining,
        )  # TODO: Is this still needed? -> redefine fake_nadir based on winner. Note that fake_ideal stays the same.
        shifted_mps = shift_points(
            problem,
            most_preferred_solutions=orig_mps,
            group_preferred_solution=winner.objective_values,
            steps_remaining=options.zoom_options.num_steps_remaining,
        )
        if isinstance(options.GPRMoptions.method_options, IPR_Options):
            options.GPRMoptions.method_options.most_preferred_solutions = shifted_mps
    # Common to both IPR and EMO
    options.GPRMoptions.fake_ideal = fake_ideal
    options.GPRMoptions.fake_nadir = fake_nadir
    return options


def favorite_method(problem: Problem, options: FavOptions, results_list: list[FavResults]) -> FavResults:
    """Run one complete iteration of the favorite method.

    For multiple iterations, call this function multiple times, passing the previous results in results_list.
    Make note to change the votes in options for each iteration after the first.
    Also change options.zoom_options.num_steps_remaining accordingly.

    Args:
        problem: DESDEO Problem object
        options: FavOptions for the favorite method.
        results_list: List of previous FavResults. Can be None in the first iteration.

    Returns:
        FavResults: Results from this iteration of the favorite method. It also contains a filled up version of
        FavOptions (which includes, e.g., updated most preferred solutions and fake_nadir after zooming)
    """
    # Getting representative set

    # calculating candidate fair solutions
    options = setup(problem, options, results_list)

    # first iteration
    # gprm_results = get_representative_set(problem, options.GPRMoptions, [])
    gprm_results = get_representative_set(problem, options.GPRMoptions, [result.GPRMResults for result in results_list])

    fair_solutions = []
    # TODO (@jpajasmaa): many complicated things happening here. Rewrirte this function to be simpler.
    # Know that you can find obj values in gprm_results.outputs (not normalized)

    targets = pl.DataFrame([point.targets for point in gprm_results.raw_results.evaluated_points])
    # print(targets)
    fair_solutions, _ = find_group_solutions(problem, solutions=gprm_results.outputs, targets=targets,
                                             most_preferred_solutions=options.original_most_preferred_solutions)

    return FavResults(
        FavOptions=options,
        GPRMResults=gprm_results,
        fair_solutions=fair_solutions,
    )


if __name__ == "__main__":
    # MethodOptions = IPR_Options | EMOOptions
    # MethodResults = IPR_Results | EMOResult

    from desdeo.problem.testproblems.dtlz2_problem import dtlz2

    dtlz2_problem = dtlz2(8, 3)
    ideal = dtlz2_problem.get_ideal_point()
    nadir = dtlz2_problem.get_nadir_point()

    n_of_dms = 4

    evaluated_points = []
    most_preferred_solutions = {
        "DM1": {"f_1": 0.17049589013991726, "f_2": 0.17049589002331159, "f_3": 0.9704959056742878},
        "DM2": {"f_1": 0.17049589008489896, "f_2": 0.9704959056849697, "f_3": 0.17049589001752685},
        "DM3": {"f_1": 0.17049589008489896, "f_2": 0.9704959056849697, "f_3": 0.17049589001752685},
        "DM4": {"f_1": 0.9704959057874635, "f_2": 0.17049588971897997, "f_3": 0.1704958898000307},
    }
    """
# random rps
    reference_points = {}
    for i in range(n_of_dms):
        reference_points[f"DM{i+1}"] = {"f_1": np.random.random(), "f_2": np.random.random(), "f_3": np.random.random()}

    print(reference_points)
    from desdeo.tools.scalarization import add_asf_nondiff, add_asf_diff
    from desdeo.tools import ProximalSolver, GurobipySolver, PyomoIpoptSolver

    most_preferred_solutions = {}
    DMs = reference_points.keys()
    for dm in DMs:
        p, target = add_asf_diff(
            dtlz2_problem,
            symbol=f"asf",
            reference_point=reference_points[dm],
        )
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        fs = res.optimal_objectives
        most_preferred_solutions[f"{dm}"] = fs

    """
    most_preferred_solutions_list = dict_of_rps_to_list_of_rps(most_preferred_solutions)

    # TODO: get fake_ideal and fake_nadir to get started!
    fake_ideal, fake_nadir = agg_aspbounds(most_preferred_solutions_list, dtlz2_problem)
    print("Fake ideal", fake_ideal)
    print("Fake nadir", fake_nadir)

    fairness_metrics = ["regret_sum", "regret_mm", "cones"]

    options = IPR_Options(
        fake_ideal=fake_ideal,
        fake_nadir=fake_nadir,
        most_preferred_solutions=most_preferred_solutions,
        num_initial_reference_points=10000,
        num_points_to_evaluate=num_of_runs,
        EvaluatedPoints=evaluated_points,
        version="fakenadir",
        # fairness_metrics=fairness_metrics,
        # version="fake"
    )
    """
     GET REPRESENTATIVE SET
    """
    df, method_res = get_representative_set(dtlz2_problem, options)
    # print(df["objectives"])
    # print(method_res.evaluated_points)

    mps = most_preferred_solutions
    rp_arr = []
    for i, dm in enumerate(mps):
        rp_arr.append(objective_dict_to_numpy_array(dtlz2_problem, mps[dm]).tolist())

    """
    Get fair solutions and visualization
    """
    fair_sols = []
    fairness_metrics = ["regret"]
    n = 1

    # TODO: fix pseudocode. This is the idea, but now we would need to call the polars expression versions of these.
    # fair_sols.append(find_group_solutions(dtlz2_problem, method_res.evaluated_points, most_preferred_solutions_list, "regret", "sum"))
    fair_sols, regret_values_dict = find_group_solutions(
        dtlz2_problem, method_res.evaluated_points, rp_arr, selectors=fairness_metrics, n=n
    )

    print("fair sols:", fair_sols)

    """
    fairmm = regret_values_dict["nash"]
    # fairmm = regret_values_dict["min"]
    # print(fairmm)

    y = np.linspace(min(fairmm), max(fairmm), 100)
    x = fairmm
    print(min(fairmm), max(fairmm))
    fig = ex.scatter(x, y)
    # fig.write_image(f"/home/jp/tyot/mop/desdeo/DESDEO/experiment/code/generic_method/fairness_tests/fairlinmm.png")
    fig.show("browser")
    """
    visualize_3d(options, method_res.evaluated_points, fair_sols, n)

    """
         Voting
    """
    # TODO: Implement majority judgemnet for voting to select the group preferred solution. Now uses majority rule, fails otherwise.
    from desdeo.gdm.voting_rules import majority_rule

    votes_idx = {"DM1": 6, "DM2": 6, "DM3": 2, "DM4": 6}
    winner_idx = majority_rule(votes_idx)
    print(winner_idx)

    # TODO: either convert or return fair solutions in dictionary format
    group_preferred_solution = fair_sols[winner_idx].objective_values
    # group_preferred_solution = {"f_1": fair_sols[winner_idx][0], "f_2": fair_sols[winner_idx][1], "f_3": fair_sols[winner_idx][2]}
    print(group_preferred_solution)

    """
        Zooom in
    """
    # TODO: Zoom in with the ITP (either opt. more solutions or just remove the ones outside)
    # TODO: note that when zooming to steps remaining 1, all mpses are at the same point and fair sol stuff of course breaks.
    steps_remaining = 2
    new_iter_options = handle_zooming(dtlz2_problem, method_res, group_preferred_solution, steps_remaining)

    # TODO: update rp arr to the shifter MPSES
    mps = new_iter_options.most_preferred_solutions
    shifted_rp_arr = []
    for i, dm in enumerate(mps):
        shifted_rp_arr.append(objective_dict_to_numpy_array(dtlz2_problem, mps[dm]).tolist())

    """
     GET REPRESENTATIVE SET
    """
    # TODO: update the UFs, show new fair solutions and the loop continues
    df, method_res2 = get_representative_set(dtlz2_problem, new_iter_options)
    # print(new_iter_options)
    # print("new iter")
    # print(df["objectives"])
    # print(method_res)

    """
    Get fair solutions and visualization
    """
    # print(method_res.fair_solutions)
    # TODO: fix pseudocode. This is the idea, but now we would need to call the polars expression versions of these.
    # fair_sols.append(find_group_solutions(dtlz2_problem, method_res.evaluated_points, most_preferred_solutions_list, "regret", "sum"))
    fair_sols, regret_values_dict = find_group_solutions(
        dtlz2_problem, method_res2.evaluated_points, shifted_rp_arr, selectors=fairness_metrics, n=n
    )

    print("fair sols after shrinking:", fair_sols)

    visualize_3d(new_iter_options, method_res2.evaluated_points, fair_sols, n)

    """
    fairmm = regret_values_dict["nash"]
    # fairmm = regret_values_dict["min"]
    print(fairmm)

    y = np.linspace(min(fairmm), max(fairmm), 100)
    x = fairmm
    print(min(fairmm), max(fairmm))
    fig = ex.scatter(x, y)
    # fig.write_image(f"/home/jp/tyot/mop/desdeo/DESDEO/experiment/code/generic_method/fairness_tests/fairlinmm.png")
    fig.show("browser")
    """
    print("done")
