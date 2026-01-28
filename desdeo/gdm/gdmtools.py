""" This module contains tools for group decision making such as manipulating set of preferences. """

from desdeo.problem import Problem
from numba import njit  # type: ignore

import polars as pl

from .preference_aggregation import eval_RP, maxmin_criterion
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
    """
    Numpy docs about argpartition: The returned indices are not guaranteed to be sorted according to the values. Furthermore, 
    the default selection algorithm introselect is unstable, and hence the returned indices are not guaranteed to be 
    the earliest/latest occurrence of the element.

    Basically this may frick up
    TODO: should be only used in getting all the solutions?
    """
    idxs = np.argpartition(fairness_values, -n)[-n:]
    fair_sols = []
    for i in range(n):
        fair_sols.append(solutions[idxs[i]])

    return fair_sols, idxs

# TODO: comments
def scale_rp(problem: Problem, reference_point, ideal, nadir, maximize):
    """Scales a reference point to [0,1]"""
    rp = {}
    # ideal = problem.get_ideal_point()
    # nadir = problem.get_nadir_point()

    # scaling to [0,1], when maximizing objective functions
    # TODO (@jpajasmaa) Why not use problem.objective[..].maximize?
    for obj in problem.objectives:
        if maximize:
            rp.update({obj.symbol: (reference_point[obj.symbol] - nadir[obj.symbol]) / (ideal[obj.symbol] - nadir[obj.symbol])})
        else:
            rp.update({obj.symbol: (reference_point[obj.symbol] - ideal[obj.symbol]) / (nadir[obj.symbol] - ideal[obj.symbol])})  # when minimizing
    return rp


"""
Utility funcs
"""
# for i in n solutions and for each p in P DMs. returns the regret values as a list of each P DMs.
# So for 3 DMs, returns [p1, p2, p3]
# The larger the value is, the more regret the DM has
# Numba goes BRRRRRT
# @njit()
def regret_allDMs_no_impro(sol, mpses):
    uf_arr = []  # convert to numpy later for numba

    zeros = np.zeros(len(sol)) # + 1e-6 # removed from now here.

    for p in range(len(mpses)):
        uf_arr.append(np.sum(np.maximum(zeros, sol - mpses[p])))  # improvements do not count
        # uf_arr.append(np.sum(sol - mpses[p]))  # improvements do count

    return uf_arr

@njit()
def regret_allDMs(sol, mpses):
    uf_arr = []  # convert to numpy later for numba

    for p in range(len(mpses)):
        uf_arr.append(np.sum(sol - mpses[p]))  # improvements do count

    return uf_arr

@njit()
def regret_allDMs_max(sol, mpses):
    uf_arr = []  # convert to numpy later for numba

    # ones = np.ones(len(sol))

    for p in range(len(mpses)):
        uf_val = np.max(sol - mpses[p])
        if uf_val < 1e-6:
            uf_val = 1e-6
        uf_arr.append(uf_val)  # improvements do not count
        # uf_arr.append(np.maximum(ones, sol - mpses[p]))  # improvements do not count
        # uf_arr.append(np.sum(np.maximum(ones, sol - mpses[p])))  # improvements do not count
        # uf_arr.append(np.max(sol - mpses[p]))  # improvements do count

    return uf_arr


"""
 Fainess funcs
"""


# X: all solutions
# P: MPSes
# all solutions, MPSes, everything have to be scaled and converted to minimization
def min_max_regret_no_impro(targets_df: pl.DataFrame, mpses: list[np.ndarray]):
    min_regrets = []
    # convert polars dataframes dictionaries to numpy arrays
    all_sols = targets_df.to_numpy()
    for i in range(len(all_sols)):
        per_sol = regret_allDMs_no_impro(all_sols[i], mpses)

        min_regrets.append(min(per_sol))  # minmax
    return min_regrets

# X: all solutions
# P: MPSes
# all solutions, MPSes, everything have to be scaled and converted to minimization
def min_max_regret(all_sols, mpses):
    min_regrets = []
    for i in range(len(all_sols)):
        per_sol = regret_allDMs_max(all_sols[i], mpses)
        # per_sol = regret_allDMs(all_sols[i], mpses)

        min_regrets.append(min(per_sol))  # minmax
    return min_regrets

# X: all solutions
# P: MPSes
# all solutions, MPSes, everything have to be scaled and converted to minimization
def max_min_regret(all_sols, mpses):
    max_min_regrets = []
    for i in range(len(all_sols)):
        # per_sol = regret_allDMs_max(all_sols[i], mpses)
        per_sol = regret_allDMs(all_sols[i], mpses)

        max_min_regrets.append(max(per_sol))  # minmax
    return max_min_regrets


def average_pareto_regret(all_sols, mpses):
    avg_regrets = []
    M = len(mpses)  # number of DMs
    for i in range(len(all_sols)):
        pr = regret_allDMs(all_sols[i], mpses)
        avg_pr = 1 / M * sum(pr)

        avg_regrets.append(avg_pr)  # avg pr
    return avg_regrets

def inequality_in_pareto_regret(all_sols, mpses):
    gini_regrets = []
    M = len(mpses)  # number of DMs
    for i in range(len(all_sols)):
        pr = regret_allDMs(all_sols[i], mpses)
        pr = np.array(pr)  # to make subtraction to work
        avg_pr = 1 / M * sum(pr)

        gini_sum = sum(abs(pr - avg_pr))

        gini_regrets.append(gini_sum)

    return gini_regrets

# See Bertsimas: On the Efficiency-Fairness Trade-off
# Utilities must be strictly positive.
def alpha_fairness(targets_df: pl.DataFrame, mpses: list[np.ndarray], alpha: float):
    alpha_fairs = []

    # convert polars dataframes dictionaries to numpy arrays
    all_sols = targets_df.to_numpy()
    print(all_sols)

    M = len(mpses)  # number of DMs
    for i in range(len(all_sols)):
        # TODO: utilities must be [0, R^n_+]
        utilities = regret_allDMs_no_impro(all_sols[i], mpses)
        if alpha == 1:
            alpha_fairs.append(np.sum(np.log(utilities)))
        else:
            tops = []
            bottom = 1 - alpha
            if bottom == 0:  # this should not happen but just in case
                bottom += 1e-6
            for m in range(M):
                tops.append(utilities[m]**(1 - alpha))
            result = np.array(tops) / bottom
            alpha_fairs.append((np.sum(result)))

    return alpha_fairs

# Vectorized versions of cones and additive local preference models
def cones_preference_model(r, mpses):
    cip = mpses.max(axis=0, keepdims=True) + 1e-6  # get fake nadir as cip
    cip = cip[0]  # drop unnessecary 2d list
    print("cip", cip)
    cones_utilities = []
    M = len(mpses)
    for m in range(M):
        cones_utilities.append(eval_RP(mpses[m], cip, r))
    return cones_utilities

def additive_preference_model(r, mpses):
    pass
