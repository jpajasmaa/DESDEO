""" The Favorite method, a general method for group decision making in multiobjective optimization """


import plotly.express as ex
from desdeo.problem.schema import Problem
import numpy as np
import polars as pl
from desdeo.mcdm.nautilus_navigator import calculate_navigation_point
from desdeo.problem import (
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)
from desdeo.tools import IpoptOptions, PyomoIpoptSolver
from desdeo.tools.scalarization import add_asf_diff
from desdeo.tools.GenerateReferencePoints import generate_points
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint, choose_reference_point
from desdeo.emo.options.templates import EMOOptions
from desdeo.tools.generics import EMOResult, SolverResults
from sys import version
import pydantic
from pydantic import Field, ConfigDict
from desdeo.gdm.gdmtools import (dict_of_rps_to_list_of_rps, agg_aspbounds, alpha_fairness, min_max_regret, min_max_regret_no_impro, scale_rp, get_top_n_fair_solutions,
                                 max_min_regret, min_max_regret_no_impro, average_pareto_regret, inequality_in_pareto_regret)

# from desdeo.gdm.preference_aggregation import find_GRP


# TODO: now for easier testing. REMOVE THIS GLOBAL VAR
num_of_runs = 100


def visualize_3d(options, evaluated_points, fair_sols, n):
    fig = ex.scatter_3d()

    # Add reference points
    chosen_refps = pl.DataFrame([point.reference_point for point in evaluated_points])
    # rescale reference points
    chosen_refps = chosen_refps.with_columns(
        [(pl.col(obj) * (options.fake_nadir[obj] - options.fake_ideal[obj]) + options.fake_ideal[obj]).alias(obj) for obj in options.fake_ideal.keys()]
    )

    # rescale reference points # TODO: need to fix this
    # chosen_refps = chosen_refps.with_columns(
    #    [(pl.col(obj) * (problem.nadir[obj] - problem.ideal[obj]) + problem.ideal[obj]).alias(obj) for obj in problem.ideal.keys()]
    # )
    fig = fig.add_scatter3d(
        x=chosen_refps["f_1"].to_numpy(),
        y=chosen_refps["f_2"].to_numpy(),
        z=chosen_refps["f_3"].to_numpy(),
        name="Reference Points", mode="markers", marker_symbol="circle", opacity=0.8)
    # TODO: add color for the agg. sum fairness value for each solution. Find in some bhupinder notebook how it was done exactly.
    # Add front
    front = pl.DataFrame([point.objectives for point in evaluated_points])
    fig = fig.add_scatter3d(
        x=front["f_1"].to_numpy(),
        y=front["f_2"].to_numpy(),
        z=front["f_3"].to_numpy(),
        mode="markers", name="Front", marker_symbol="circle", opacity=0.9)
    fig = fig.add_scatter3d(
        x=[options.fake_ideal["f_1"]],
        y=[options.fake_ideal["f_2"]],
        z=[options.fake_ideal["f_3"]],
        mode="markers", name="fake_ideal", marker_symbol="diamond", opacity=0.9)
    fig = fig.add_scatter3d(
        x=[options.fake_nadir["f_1"]],
        y=[options.fake_nadir["f_2"]],
        z=[options.fake_nadir["f_3"]],
        mode="markers", name="fake_nadir", marker_symbol="diamond", opacity=0.9)
    DMs = options.most_preferred_solutions.keys()
    for dm in DMs:
        fig = fig.add_scatter3d(
            x=[options.most_preferred_solutions[dm]["f_1"]],
            y=[options.most_preferred_solutions[dm]["f_2"]],
            z=[options.most_preferred_solutions[dm]["f_3"]],
            mode="markers", name=dm, marker_symbol="square", opacity=0.9)

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

# get a problem
class ProblemWrapper():
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
        refp = {obj: val * (self.nadir[obj] - self.ideal[obj]) + self.ideal[obj] for obj, val in zip(self.ideal.keys(), scaled_refp)}
        scaled_problem, target = add_asf_diff(self.problem, "target", refp)
        solver = PyomoIpoptSolver(scaled_problem)
        # solver = guess_best_solver(scaled_problem)
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
def find_group_solutions(problem: Problem, evaluated_points: list[_EvaluatedPoint], mps: list[np.ndarray],
                         selectors: list[str], n: int = 3):
    """Find n fair group solutions according to different fairness criteria.
        Assumes everything has been properly converted to minimization already.
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

    # TODO: apply the selectors string to determine which of the fairness_metrics to apply
    print(selectors)
    # add maxmin-cones, maybe maxmin also?

    # TODO: would be convenient to get all the parameter to some parameter dictionary etc that I can apply.
    # Look for inspiration in Bhupinder's EMO stuff?
    # Regret UFs, no achievement for aspiration. Sum over DMs.

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
    min_r_no = get_top_n_fair_solutions(eval_sols_in_objs, min_no_regrets, n)
    util_r = get_top_n_fair_solutions(eval_sols_in_objs, utilitarian, n)
    min_r = get_top_n_fair_solutions(eval_sols_in_objs, min_regrets, n)
    avg_r = get_top_n_fair_solutions(eval_sols_in_objs, avg_regrets, n)
    gini_r = get_top_n_fair_solutions(eval_sols_in_objs, gini_regrets, n)
    nash_r = get_top_n_fair_solutions(eval_sols_in_objs, nash, n)
    # TODO: find out what is the bug here. returns arrays
    # alpha_r = get_top_n_fair_solutions(eval_sols_in_objs, alpha_regrets, n)

    """
    Maxmin-cones
    cip = np.array([np.max(norm_mps_arr[:, 0]) + 0.001, np.max(norm_mps_arr[:, 1]) + 0.001, np.max(norm_mps_arr[:, 2]) + 0.001])
    ideal_arr = np.array([np.min(norm_mps_arr[:, 0]), np.min(norm_mps_arr[:, 1]), np.min(norm_mps_arr[:, 2])])

    k = 3
    q = 3
    pa = "eq_maxmin_cones"
    # pa = "eq_maxmin"
    all_rps = norm_mps_arr
    GRP, _ = find_GRP(all_rps, cip, k, q, ideal_arr, all_rps, pa)
    # GRP = GRP - cip
    print("MAXMIN cones GRP", GRP)
    # Find PO solution with conesGRP
    GRP_dict = {"f_1": GRP[0], "f_2": GRP[1], "f_3": GRP[2]}

    p, target = add_asf_diff(
        dtlz2_problem,
        symbol=f"asf",
        reference_point=GRP_dict,
    )
    # scaled_problem, target = add_asf_diff(self.problem, "asf", refp)
    solver = PyomoIpoptSolver(p)
    res = solver.solve(target)
    fs = res.optimal_objectives
    GRP_po = objective_dict_to_numpy_array(problem, fs)

    """
    # top_fair = np.concatenate((min_r_no, min_r, avg_r, gini_r, util_r, nash_r, [GRP_po]))
    top_fair = np.concatenate((min_r_no, min_r, avg_r, gini_r, util_r, nash_r))
    # top_fair = np.stack(top_fair)

    regret_values = {"min_no": min_no_regrets, "min": min_regrets, "avg": avg_regrets, "gini": gini_regrets,
                     "util": utilitarian, "nash": nash, }  # "cones": GRP_po}

    return top_fair, regret_values


def shift_points(problem: Problem, most_preferred_solutions, group_preferred_solution: dict[str, float], steps_remaining):
    """Calls calculate_navigation_point to shift individual most most_preferred_solutions. Then projects them to Pareto front to return as most preferred solutions"""

    shifted_mps = {}
    for dm in most_preferred_solutions:
        shifted_point = calculate_navigation_point(problem, most_preferred_solutions[dm], group_preferred_solution, steps_remaining)
        p, target = add_asf_diff(
            problem,
            symbol="asf",
            reference_point=shifted_point,
        )
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        shifted_mps.update({dm: res.optimal_objectives})

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
    version: str
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
    evaluated_points = []

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

    dims = len(problem.get_nadir_point())

    # get the representative set
    # set n or the possibilities of n according to the num points to evaluate
    for n in ([options.num_points_to_evaluate, options.num_points_to_evaluate / 2, 10]):
        try:
            # generate points in the convex hull of RPs or fake_ideal and fake_nadir
            if options.version == "convex_hull":
                _, refp = generate_points(num_points=options.total_points, num_dims=dims, reference_points=norm_mps)
            else:
                _, refp = generate_points(num_points=options.total_points, num_dims=dims, reference_points=None)

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
    # print(most_preferred_solutions)
    most_preferred_solutions_list = dict_of_rps_to_list_of_rps(most_preferred_solutions)
    fake_ideal, current_fake_nadir = agg_aspbounds(most_preferred_solutions_list, problem)  # we can get fake_nadir with calculate_navigation_point
    # print(fake_ideal, current_fake_nadir)
    fake_nadir = calculate_navigation_point(problem, current_fake_nadir, group_mps, steps_remaining)

    shifted_mps = shift_points(problem, most_preferred_solutions, group_mps, steps_remaining)
    # print(shifted_mps)

    # steps_remaining = res.steps_remaining - 1
    # TODO: somewhere add check for steps_remaining need to be > 0

    options = IPR_Options(
        fake_ideal=fake_ideal,
        fake_nadir=fake_nadir,
        most_preferred_solutions=shifted_mps,
        total_points=10000,  # TODO: set params for these someway
        num_points_to_evaluate=num_of_runs,  # TODO: give as a parameter
        EvaluatedPoints=res.evaluated_points,
        version="convex_hull",
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

    n_of_dms = 3

    evaluated_points = []
    most_preferred_solutions = {'DM1': {'f_1': 0.17049589013991726, 'f_2': 0.17049589002331159, 'f_3': 0.9704959056742878},
                                'DM2': {'f_1': 0.17049589008489896, 'f_2': 0.9704959056849697, 'f_3': 0.17049589001752685},
                                'DM3': {'f_1': 0.9704959057874635, 'f_2': 0.17049588971897997, 'f_3': 0.1704958898000307}}
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
        total_points=10000,
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
    fair_sols, regret_values_dict = find_group_solutions(dtlz2_problem, method_res.evaluated_points,
                                                         rp_arr, selectors=fairness_metrics, n=n)

    print("fair sols:", fair_sols)

    fairmm = regret_values_dict["nash"]
    # fairmm = regret_values_dict["min"]
    # print(fairmm)

    y = np.linspace(min(fairmm), max(fairmm), 100)
    x = fairmm
    print(min(fairmm), max(fairmm))
    fig = ex.scatter(x, y)
    # fig.write_image(f"/home/jp/tyot/mop/desdeo/DESDEO/experiment/code/generic_method/fairness_tests/fairlinmm.png")
    fig.show("browser")

    visualize_3d(options, method_res.evaluated_points, fair_sols, n)

    """
         Voting
    """
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
    fair_sols, regret_values_dict = find_group_solutions(dtlz2_problem, method_res2.evaluated_points,
                                                         shifted_rp_arr, selectors=fairness_metrics, n=n)

    print("fair sols after shrinking:", fair_sols)

    visualize_3d(new_iter_options, method_res2.evaluated_points, fair_sols, n)

    fairmm = regret_values_dict["nash"]
    # fairmm = regret_values_dict["min"]
    print(fairmm)

    y = np.linspace(min(fairmm), max(fairmm), 100)
    x = fairmm
    print(min(fairmm), max(fairmm))
    fig = ex.scatter(x, y)
    # fig.write_image(f"/home/jp/tyot/mop/desdeo/DESDEO/experiment/code/generic_method/fairness_tests/fairlinmm.png")
    fig.show("browser")
