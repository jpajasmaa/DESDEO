""" This module contains tools for group decision making such as manipulating set of preferences. """

from desdeo.problem import Problem
from numba import njit  # type: ignore

import numpy as np

# Below two are tools for GDM, have needed them in both projects
def dict_of_rps_to_list_of_rps(reference_points: dict[str, dict[str, float]]) -> list[dict[str, float]]:
    """
    Convert dict containing the DM key to an ordered list.
    """
    return list(reference_points.values())

def list_of_rps_to_dict_of_rps(reference_points: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Convert the ordered list to a dict contain the DM keys. TODO: Later maybe get these keys from somewhere.
    """
    return {f"DM{i+1}": rp for i, rp in enumerate(reference_points)}

# TODO:(@jpajasmaa) document
def agg_aspbounds(po_list: list[dict[str, float]], problem: Problem):
    agg_aspirations = {}
    agg_bounds = {}

    for obj in problem.objectives:
        if obj.maximize:
            agg_aspirations.update({obj.symbol: max(s[obj.symbol] for s in po_list)})
            agg_bounds.update({obj.symbol: min(s[obj.symbol] for s in po_list)})
        else:
            agg_aspirations.update({obj.symbol: min(s[obj.symbol] for s in po_list)})
            agg_bounds.update({obj.symbol: max(s[obj.symbol] for s in po_list)})

    return agg_aspirations, agg_bounds


# TODO:(@jpajasmaa) comments
def scale_delta(problem: Problem, d: float):
    delta = {}
    ideal = problem.get_ideal_point()
    nadir = problem.get_nadir_point()

    for obj in problem.objectives:
        if obj.maximize:
            delta.update({obj.symbol: d*(ideal[obj.symbol] - nadir[obj.symbol])})
        else:
            delta.update({obj.symbol: d*(nadir[obj.symbol] - ideal[obj.symbol])})
    return delta


def get_top_n_fair_solutions(solutions, fairness_values, n):
    idxs = np.argpartition(fairness_values, -n)[-n:]
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


"""
%Compute regret values
MDutility=zeros(n,3);
for i=1:n
        MDutility(i, 1)=sum(max(zeros(1,3),X(i,:)-A(1,:))); p = 0
        MDutility(i, 2)=sum(max(zeros(1,3),X(i,:)-A(2,:))); p = 1
        MDutility(i, 3)=sum(max(zeros(1,3),X(i,:)-A(3,:))); p = 2
end
"""
# for i in n solutions and for each p in P DMs. returns the regret values as a list of each P DMs.
# So for 3 DMs, returns [p1, p2, p3]
# The larger the value is, the more regret the DM has
# Numba goes BRRRRRT
@njit()
def MDutility_allDMs(sol, mpses):
    uf_arr = []  # convert to numpy later for numba

    print(sol)
    zeros = np.zeros(len(sol))

    for p in range(len(mpses)):
        uf_arr.append(np.sum(np.maximum(zeros, sol - mpses[p])))  # improvements do not count
        # uf_arr.append(np.sum(sol - mpses[p]))  # improvements do count

    return uf_arr

# X: all solutions
# P: MPSes
# all solutions, MPSes, everything have to be scaled and converted to minimization
def uf_total_sum(all_sols, mpses):

    sum_regrets = []
    for i in range(len(all_sols)):
        per_sol = MDutility_allDMs(all_sols[i], mpses)
        print(per_sol)

        # TODO: call fairness func to aggreate here. Now just taking the min-max.
        sum_regrets.append(min(per_sol))  # minmax
        # sum_regrets.append(sum(per_sol))  # sum max
        # sum_regrets.append(1/3*sum(per_sol))  # avg max
    return sum_regrets


"""
OLD VERSION
"""

# OK Juergen UFs just provide for each solution i in n, and for k in objs, a regret value of the DMs.
#
#
#
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
