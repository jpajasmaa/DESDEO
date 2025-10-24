""" The Favorite method, a general method for group decision making in multiobjective optimization """


from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint, choose_reference_point
from desdeo.tools.GenerateReferencePoints import generate_points
from desdeo.tools.scalarization import add_asf_diff
from desdeo.tools import IpoptOptions, PyomoIpoptSolver
from desdeo.problem import (
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)


import numpy as np
from desdeo.problem.schema import Problem

# Juergen UFS for optimization
# Looks to be correctly working
"""
And yes, I am assuming we are minimising the objectives and the utilities.

Did you mean maximisation of objectives, or utilities, or both?
I think if you just want to maximise objectives (and still minimise utilities), then
I think you would swap around terms:
>
> PUtility(i, p, k)= pdweight(p,k)*max(rw*(P(p,k)-X(i,k)),P(p,k)-X(i,k))

If you want to maximise utility, probably you would multiply by -1.

We are then searching for solutions than are non-dominated with respect to Utility, so IOPIS with this utility function should work ok.
"""
def PUtility(i, p, k, X, P, rw, pdw, maximizing=False):
    """
    i: index over solutions ==> a solution
    p: index over DMs ==> specific DM
    k: index over objectives ==> speficic obj
    """
    # weights, rows DMS, columns objs. TODO: if providing different weights, change this. Now 1 for all.

    if maximizing:
        f_term = rw * (P[p, k] - X[i, k])
        s_term = P[p, k] - X[i, k]
    else:
        f_term = rw * (X[i, k] - P[p, k])
        s_term = X[i, k] - P[p, k]
    maxterm = np.max((f_term, s_term))  # if wanting to maximize utility, multiply by -1.
    return pdw[p, k] * maxterm

def UF_total_sum(i, p, X, P, rw, pdw, maximize=False):
    summa = 0
    K = X.shape[1]
    for k in range(K):
        summa += PUtility(i, p, k, X, P, rw, pdw, maximize)

    return summa

def UF_mm(i, p, X, P, rw, pdw, maximize=False):
    Putilities = []
    K = X.shape[1]
    for k in range(K):
        Putilities.append(PUtility(i, p, k, X, P, rw, pdw, maximize))

    return max(Putilities)


# TODO: addstuff params and so on. rewrite maybe
# X : solutions
# P : MPSses
# maximize: Whether to minimize or maximize the objective functions
def solve_UFs(X: np.ndarray, P: np.ndarray, rw: float, pdw: np.ndarray | None, agg: str, maximize=False):
    UF_vals = []
    UF_ws = []

    Q, K = X.shape[0], X.shape[1]
    # weights, rows DMS, columns objs. TODO: if providing different weights, change this. Now 1 for all.
    rw = rw  # factor to multiple rewards
    if pdw is None:
        pdw = np.ones((Q, K))
    # for each solution
    for i in range(len(X)):
        UFs_for_each = []
        # for each DM. So this UF for each DM as an objective function for IOPIS should work?
        for p in range(len(P)):  # 4 DMs right now
            if agg == "sum":
                uf_val = UF_total_sum(i, p, X, P, rw, pdw, maximize)
            else:  # agg == mm
                uf_val = UF_mm(i, p, X, P, rw, pdw, maximize)
            UFs_for_each.append(uf_val)
            UF_vals.append(uf_val)
        if agg == "sum":
            UF_ws.append(sum(UFs_for_each))
        else:
            UF_ws.append(min(UFs_for_each))
    print(len(UF_vals))
    print(len(UF_ws))

    return UF_vals, UF_ws

def get_top_n_fair_solutions(solutions, UF_ws, n):
    idxs = np.argpartition(UF_ws, -n)[-n:]
    fair_sols = []
    for i in range(n):
        fair_sols.append(solutions[idxs[i]])
    return fair_sols

# TODO: comments
def scale_rp(problem: Problem, reference_point, ideal, nadir, maximize):
    """Scales a reference point to [0,1]"""
    rp = {}
    # ideal = problem.get_ideal_point()
    # nadir = problem.get_nadir_point()

    # scaling to [0,1], when maximizing objective functions
    for obj in problem.objectives:
        if maximize:
            rp.update({obj.symbol: (reference_point[obj.symbol] - nadir[obj.symbol]) / (ideal[obj.symbol] - nadir[obj.symbol])})
        else:
            rp.update({obj.symbol: (reference_point[obj.symbol] - ideal[obj.symbol]) / (nadir[obj.symbol] - ideal[obj.symbol])})  # when minimizing
    return rp


# get a problem
class ProblemWrapper():
    def __init__(self, problem: Problem):
        """Initialize the problem wrapper for a DESDEO problem.

        Args:
        problem = a DESDEO problem
        """
        # problem = dtlz2(8, 3)  # TODO: set this as a parameter someway
        self.problem = problem  # TODO: set this as a parameter someway
        self.ideal, self.nadir = problem.get_ideal_point(), problem.get_nadir_point()
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

def itp_mps(problem: Problem, mps):
    """
    Problem needs to have ideal and nadir set. right?
    mps : most preferred solution for each DM
    """

    evaluated_points = None
    # TODO: this is one of the optimizers only
    # todo update the iterations as a list parameter
    # evaluated_points = None
    for n in ([100, 50, 20, 10]):
        try:
            # generate points in the convex hull of RPs or maxmin RPs
            _, refp = generate_points(num_points=10000, num_dims=3, reference_points=mps)

            num_runs = n
            wrapped_problem = ProblemWrapper(problem)
            for i in range(num_runs):
                if (i+1) % 10 == 0 or i == 0:
                    print(f"Run {i+1}/{num_runs}")
                reference_point, _ = choose_reference_point(refp, evaluated_points)
                evaluated_points = wrapped_problem.solve(reference_point)

            print(wrapped_problem.ideal)
            print(wrapped_problem.nadir)
            break
        except Exception:
            break

    return evaluated_points

def find_group_solutions(problem: Problem, evaluated_points, norm_mps, solution_selector: str, aggregator: str, n: int = 3):
    """Find n fair group solutions according to different fairness criteria.
        TODO: extend to return any amount of fair solutions with some order such as:
        1. Maxmin-cones solution
        2. regret sum
        3. regret maxmin

        Maxmin-cones is a singular, for regret sum and regret maxmin can return multiple solutions.

    """

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


def shrink_group_ROI(wrapped_problem: ProblemWrapper, group_preferred_solution: dict[str, float]):
    """Version with problemWrapper that gets updated."""
    pass

def shift_points(problem: Problem, most_preferred_solutions, group_preferred_solution: dict[str, float], steps_remaining):
    """Calls calculate_navigation_point to shift fake_nadir and individual most most_preferred_solutions. """

    shifted_mps = {}
    for dm in most_preferred_solutions:
        shifted_mps.update({dm: calculate_navigation_point(dtlz2_problem, most_preferred_solutions[dm], group_preferred_solution, steps_remaining)})

    return shifted_mps


"""
THINK how to do smarter, this needs many parameters which I feel are unnecessary if done different way.
def normalize_mps_and_to_array(most_preferred_solutions):
    mps = {}
    for dm in most_preferred_solutions:
        mps.update({dm: scale_rp(dtlz2_problem, most_preferred_solutions[dm], ideal, nadir, False)})

    # RPs as array for methods to come
    normalized_most_preferred_solutions = []
    for _, dm in enumerate(mps):
        rp_arr.append(objective_dict_to_numpy_array(dtlz2_problem, mps[dm]).tolist())

    return normalized_most_preferred_solutions
"""

if __name__ == "__main__":

    from desdeo.problem.testproblems.dtlz2_problem import dtlz2
    from desdeo.problem import (
        numpy_array_to_objective_dict,
        objective_dict_to_numpy_array,
    )

    dtlz2_problem = dtlz2(8, 3)
    saved_solutions = []
    ideal = dtlz2_problem.get_ideal_point()
    nadir = dtlz2_problem.get_nadir_point()
    dtlz2_problem = dtlz2_problem.update_ideal_and_nadir(new_ideal=ideal, new_nadir=nadir)
    print(ideal)
    print(nadir)

    most_preferred_solutions = {'DM1': {'f_1': 0.17049589013991726, 'f_2': 0.17049589002331159, 'f_3': 0.9704959056742878},
                                'DM2': {'f_1': 0.17049589008489896, 'f_2': 0.9704959056849697, 'f_3': 0.17049589001752685},
                                'DM3': {'f_1': 0.9704959057874635, 'f_2': 0.17049588971897997, 'f_3': 0.1704958898000307}}

    # wrapped_problem = ProblemWrapper(dtlz2_problem)

    # do programmatically
    current_fake_nadir = {"f_1": 0.9704959057874635, "f_2": 0.9704959056849697, "f_3": 0.9704959056742878}

    # need to scale the mpses for fairness
    mps = {}
    for dm in most_preferred_solutions:
        mps.update({dm: scale_rp(dtlz2_problem, most_preferred_solutions[dm], ideal, nadir, False)})

    # RPs as array for methods to come
    rp_arr = []
    for i, dm in enumerate(mps):
        rp_arr.append(objective_dict_to_numpy_array(dtlz2_problem, mps[dm]).tolist())
    normalized_most_preferred_solutions = rp_arr

    # TODO: implement other optimizers
    optimizer = "itp"
    eval_points = itp_mps(dtlz2_problem, normalized_most_preferred_solutions)
    # eval_points = itp_mm(dtlz2_problem, maxmin_bounds)
    print(eval_points)

    # TODO: get other stuff related to the fairness
    solution_selector = "regret"
    aggregator = "sum"
    fair_sols = find_group_solutions(dtlz2_problem, eval_points, normalized_most_preferred_solutions, solution_selector, aggregator)
    print(fair_sols)

    """
    to aggregate with maxmin
    solution_selector = "regret"
    aggregator = "mm"
    fair_sols = find_group_solutions(dtlz2_problem, eval_points, normalized_most_preferred_solutions, solution_selector, aggregator)
    print(fair_sols)'
    """

    # TODO: Implement majority judgemnet for voting to select the group preferred solution. Now uses majority rule, fails otherwise.
    from desdeo.gdm.voting_rules import majority_rule

    votes_idx = {
        "DM1": 1,
        "DM2": 1,
        "DM3": 2
    }
    winner_idx = majority_rule(votes_idx)

    # TODO: either convert or return fair solutions in dictionary format
    group_preferred_solution = {"f_1": fair_sols[winner_idx][0], "f_2": fair_sols[winner_idx][1], "f_3": fair_sols[winner_idx][2]}
    print(group_preferred_solution)

    # Shift fake nadir is next. Can use from nautilus navigator
    from desdeo.mcdm.nautilus_navigator import calculate_navigation_point
    # TODO: need to think how to set this.
    steps_remaining = 3
    current_fake_nadir = calculate_navigation_point(dtlz2_problem, current_fake_nadir, group_preferred_solution, steps_remaining)
    print(current_fake_nadir)

    # TODO: now need to shift the MPSes
    shifted_mps = shift_points(dtlz2_problem, most_preferred_solutions, group_preferred_solution, steps_remaining)
    print(shifted_mps)

    # TODO: Zoom in with the ITP (either opt. more solutions or just remove the ones outside)

    # TODO: update the UFs, show new fair solutions and the loop continues
