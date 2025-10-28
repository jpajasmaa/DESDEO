""" The Favorite method, a general method for group decision making in multiobjective optimization """


from sys import version
import pydantic
from pydantic import Field, ConfigDict
from desdeo.gdm.gdmtools import *  # TODO: fix start import
from desdeo.tools.generics import EMOResult, SolverResults
from desdeo.emo.options.templates import EMOOptions
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint, choose_reference_point
from desdeo.tools.GenerateReferencePoints import generate_points
from desdeo.tools.scalarization import add_asf_diff
from desdeo.tools import IpoptOptions, PyomoIpoptSolver
from desdeo.problem import (
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)
from desdeo.mcdm.nautilus_navigator import calculate_navigation_point
import polars as pl

import numpy as np
from desdeo.problem.schema import Problem


# get a problem
class ProblemWrapper():
    def __init__(self, problem: Problem, fake_ideal, fake_nadir):
        """Initialize the problem wrapper for a DESDEO problem.

        Args:
        problem = a DESDEO problem
        """
        # problem = dtlz2(8, 3)  # TODO: set this as a parameter someway
        self.problem = problem  # TODO: set this as a parameter someway
        self.ideal, self.nadir = fake_ideal, fake_nadir
        self.problem = problem.update_ideal_and_nadir(new_ideal=self.ideal, new_nadir=self.nadir)
        self.evaluated_points: list[_EvaluatedPoint] = []

    # TODO: set solver
    def solve(self, scaled_refp: np.ndarray) -> list[_EvaluatedPoint]:
        refp = {obj: val * (self.nadir[obj] - self.ideal[obj]) + self.ideal[obj] for obj, val in zip(self.ideal.keys(), scaled_refp)}
        scaled_problem, target = add_asf_diff(self.problem, "target", refp)
        solver = PyomoIpoptSolver(scaled_problem)
        # init_solver = guess_best_solver(scaled_problem)
        results = solver.solve(target)
        objs = results.optimal_objectives
        scaled_objs = {obj: (objs[obj] - self.ideal[obj]) / (self.nadir[obj] - self.ideal[obj]) for obj in objs.keys()}
        self.evaluated_points.append(_EvaluatedPoint(
            reference_point=dict(zip(self.ideal.keys(), scaled_refp)),
            targets=scaled_objs,
            objectives=objs)
        )
        return self.evaluated_points

    """
     TODO: would need some sort of setter or other tool to set a desired solver for ProblemWrapper
    def set_solver():
        pass
    """


# TODO: this to be re-written after UFs are implemented as polars expressions
def find_group_solutions(problem: Problem, evaluated_points, mps, solution_selector: str, aggregator: str, n: int = 3):
    """Find n fair group solutions according to different fairness criteria.
        TODO: extend to return any amount of fair solutions with some order such as:
        1. Maxmin-cones solution
        2. regret sum
        3. regret maxmin

        Maxmin-cones is a singular, for regret sum and regret maxmin can return multiple solutions.

        mps must be np array for the below code !

    """

    # TODO: normalize the MPSses
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

    # TODO: would be convenient to get all the parameter to some parameter dictionary etc that I can apply.
    # Look for inspiration in Bhupinder's EMO stuff?
    rw = 0
    maximize = False
    UF_vals, UF_agg = solve_UFs(norm_eval_sols, norm_mps_arr, rw, None, aggregator, maximize)
    print(len(UF_vals))
    print(len(UF_agg))

    top_fair = np.stack(get_top_n_fair_solutions(eval_sols_in_objs, UF_agg, n))
    # top5fair = np.stack(top5fair)

    return top_fair


def shift_points(problem: Problem, most_preferred_solutions, group_preferred_solution: dict[str, float], steps_remaining):
    """Calls calculate_navigation_point to shift fake_nadir and individual most most_preferred_solutions. """

    shifted_mps = {}
    for dm in most_preferred_solutions:
        shifted_mps.update({dm: calculate_navigation_point(dtlz2_problem, most_preferred_solutions[dm], group_preferred_solution, steps_remaining)})

    return shifted_mps


# TODO: create two versions with version1 options and version2 options

class IPR_Options(pydantic.BaseModel):
    """Options for iterative_pareto_representer applied with the favorite method."""
    fake_ideal: dict[str, float | None]
    """fake ideal"""
    fake_nadir: dict[str, float | None]
    most_preferred_solutions: dict[str, dict[str, float]]
    """What about DMs' RPs? To be normalized!"""
    total_points: int
    """Big number"""
    num_points_to_evaluate: int
    """How many points to evaluate. Not so large number"""
    EvaluatedPoints: list[_EvaluatedPoint]
    """List of EvaluatedPoints from IPR."""
    # version: str
    """Version 1: evaluate in the convex hull of MPS. Version 2: evaluate in the box of fake_ideal and fake_nadir."""
    # fairness_metrics: list[str]

class IPR_OptionsV1(pydantic.BaseModel):
    """Options for iterative_pareto_representer applied with the favorite method."""
    most_preferred_solutions: dict[str, dict[str, float]]
    """What about DMs' RPs? To be normalized!"""
    total_points: int
    """Big number"""
    num_points_to_evaluate: int
    """How many points to evaluate. Not so large number"""
    EvaluatedPoints: list[_EvaluatedPoint]


class IPR_OptionsV2(pydantic.BaseModel):
    """Options for iterative_pareto_representer applied with the favorite method."""
    fake_ideal: dict[str, float | None]
    """fake ideal"""
    fake_nadir: dict[str, float | None]
    """What about DMs' RPs? To be normalized!"""
    total_points: int
    """Big number"""
    num_points_to_evaluate: int
    """How many points to evaluate. Not so large number"""
    EvaluatedPoints: list[_EvaluatedPoint]


# class IPR_Results(pydantic.BaseModel):
#    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)
#    evaluated_points: str = Field("The evaluated points.")  # I dont know how to use this properly.

class IPR_Results(pydantic.BaseModel):
    #    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)
    evaluated_points: list[_EvaluatedPoint]  # I dont know how to use this properly.
    most_preferred_solutions: dict[str, dict[str, float]]
    # fair_solutions: list[dict[str, float]]  # TODO: decide the type
    # fair_solutions: list  # TODO: decide the type
    # fairness_metrics: list[str]  # string to indicate the fair solution type


# TODO: move to some template file etc. Handle EMOOptions for IOPIS
MethodOptions = IPR_Options  # | EMOOptions
MethodResults = IPR_Results  # | EMOResult

# evaluated_points: list(_EvaluatedPoint) = Field("The evaluated points.")

# I dont know how to use this properly.
def get_representative_set(problem: Problem, options: MethodOptions) -> tuple[pl.DataFrame, MethodResults]:
    evaluated_points = None

    # TODO: switch case, for emo side and IPR side
    # Normalize mps for fairness and IPR. Convert to array for now.
    mps = {}
    for dm in options.most_preferred_solutions:
        mps.update({dm: scale_rp(problem, options.most_preferred_solutions[dm], options.fake_ideal, options.fake_nadir, False)})

    # RPs as array for methods to come
    rp_arr = []
    for i, dm in enumerate(mps):
        rp_arr.append(objective_dict_to_numpy_array(problem, mps[dm]).tolist())
    norm_mps = rp_arr

    # get the representative set
    # set n or the possibilities of n according to the num points to evaluate
    for n in ([options.num_points_to_evaluate, options.num_points_to_evaluate / 2, 10]):
        # for n in ([100, 50, 20, 10]): # This runs !
        try:
            # generate points in the convex hull of RPs or fake_ideal and fake_nadir
            if version == "convex_hull":
                _, refp = generate_points(num_points=options.total_points, num_dims=3, reference_points=norm_mps)
            else:
                _, refp = generate_points(num_points=options.total_points, num_dims=3, reference_points=None)

            num_runs = n
            wrapped_problem = ProblemWrapper(problem, fake_ideal=options.fake_ideal, fake_nadir=options.fake_nadir)
            for i in range(num_runs):
                if (i+1) % 10 == 0 or i == 0:
                    print(f"Run {i+1}/{num_runs}")
                reference_point, _ = choose_reference_point(refp, evaluated_points)
                evaluated_points = wrapped_problem.solve(reference_point)
            break
        except Exception:
            break

    df = pl.DataFrame(evaluated_points)  # TODO: only the output to be visualized to the DMs
    fav_res = IPR_Results(
        evaluated_points=evaluated_points,
        most_preferred_solutions=options.most_preferred_solutions,
        # fair_solutions=fair_sols,
        # fairness_metrics=options.fairness_metrics,
    )

    return (df, fav_res)

def handle_zooming(problem: Problem, res: MethodResults, group_mps: dict[str, float], steps_remaining: int) -> MethodOptions:
    """Should handle zooming and return MethodOptions for the next iteration. """

    # TODO: switch cases for different versions.
    # Currently, gets get current ideal and current nadir from DMs' MPSses
    most_preferred_solutions = res.most_preferred_solutions  # most preferred solutions of current iteration
    print(most_preferred_solutions)
    most_preferred_solutions_list = dict_of_rps_to_list_of_rps(most_preferred_solutions)
    fake_ideal, current_fake_nadir = agg_aspbounds(most_preferred_solutions_list, problem)  # we can get fake_nadir with calculate_navigation_point
    print(fake_ideal, current_fake_nadir)
    fake_nadir = calculate_navigation_point(problem, current_fake_nadir, group_mps, steps_remaining)

    # TODO: now need to shift the MPSes
    shifted_mps = shift_points(problem, most_preferred_solutions, group_mps, steps_remaining)
    print(shifted_mps)
    print("Fake ideal", fake_ideal)
    print("Fake nadir", fake_nadir)

    # steps_remaining = res.steps_remaining - 1
    # TODO: somewhere add check for steps_remaining need to be > 0

    options = IPR_Options(
        fake_ideal=fake_ideal,
        fake_nadir=fake_nadir,
        most_preferred_solutions=shifted_mps,
        total_points=10000,  # TODO: set params for these someway
        num_points_to_evaluate=10,
        EvaluatedPoints=res.evaluated_points,
        # version=next_iter_version,
        # fairness_metrics=fairness_metrics,
        # steps_remaining=steps_remaining,
    )
    return options


if __name__ == "__main__":

    # MethodOptions = IPR_Options | EMOOptions
    # MethodResults = IPR_Results | EMOResult
    from desdeo.problem.testproblems.dtlz2_problem import dtlz2
    dtlz2_problem = dtlz2(8, 3)
    ideal = dtlz2_problem.get_ideal_point()
    nadir = dtlz2_problem.get_nadir_point()

    evaluated_points = []
    most_preferred_solutions = {'DM1': {'f_1': 0.17049589013991726, 'f_2': 0.17049589002331159, 'f_3': 0.9704959056742878},
                                'DM2': {'f_1': 0.17049589008489896, 'f_2': 0.9704959056849697, 'f_3': 0.17049589001752685},
                                'DM3': {'f_1': 0.9704959057874635, 'f_2': 0.17049588971897997, 'f_3': 0.1704958898000307}}

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
        total_points=10000,
        num_points_to_evaluate=10,
        EvaluatedPoints=evaluated_points,
        # version="fakenadir",
        # fairness_metrics=fairness_metrics,
        # version="fake"
    )
    # df, res = get_representative_set(dtlz2_problem, options)
    df, method_res = get_representative_set(dtlz2_problem, options)
    print(df["objectives"])
    print(method_res.evaluated_points)

    # mps going to find_group_solutions() need to be 2d ndarray
    mps = most_preferred_solutions
    rp_arr = []
    for i, dm in enumerate(mps):
        rp_arr.append(objective_dict_to_numpy_array(dtlz2_problem, mps[dm]).tolist())

    # print(method_res.fair_solutions)
    """ FAIRNESS STUFF WORKED BUT NOW IMPROVING ON IT
    """
    fair_sols = []
    fairness_metrics = ["regret"]

    for criterion in fairness_metrics:
        # TODO: fix pseudocode. This is the idea, but now we would need to call the polars expression versions of these.
        # fair_sols.append(find_group_solutions(dtlz2_problem, method_res.evaluated_points, most_preferred_solutions_list, "regret", "sum"))
        fair_sols.append(find_group_solutions(dtlz2_problem, method_res.evaluated_points, rp_arr, "regret", "sum"))

    fair_sols = fair_sols[0]  # remove the outer list, only needed for this scuffed implementation
    print("fair sols:", fair_sols)

    # TODO: Implement majority judgemnet for voting to select the group preferred solution. Now uses majority rule, fails otherwise.
    from desdeo.gdm.voting_rules import majority_rule

    votes_idx = {
        "DM1": 1,
        "DM2": 1,
        "DM3": 2
    }
    winner_idx = majority_rule(votes_idx)
    print(winner_idx)

    # TODO: either convert or return fair solutions in dictionary format
    group_preferred_solution = {"f_1": fair_sols[winner_idx][0], "f_2": fair_sols[winner_idx][1], "f_3": fair_sols[winner_idx][2]}
    print(group_preferred_solution)

    # TODO: Zoom in with the ITP (either opt. more solutions or just remove the ones outside)
    steps_remaining = 3
    new_iter_options = handle_zooming(dtlz2_problem, method_res, group_preferred_solution, steps_remaining)

    # TODO: update the UFs, show new fair solutions and the loop continues
    df, method_res = get_representative_set(dtlz2_problem, new_iter_options)
    print(new_iter_options)
    print("new iter")
    print(df["objectives"])
    # print(method_res)

    # print(method_res.fair_solutions)
