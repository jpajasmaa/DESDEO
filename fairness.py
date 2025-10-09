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
    maxterm = np.max((f_term, s_term))  # np.min for when maximizing the objectives, np.max when minimizing
    return pdw[p, k] * maxterm

def UF(i, p, X, P, rw, pdw, maximize=False):
    summa = 0
    K = X.shape[1]
    for k in range(K):
        summa += PUtility(i, p, k, X, P, rw, pdw, maximize)

    return summa


# TODO: addstuff params and so on. rewrite maybe
# X : solutions
# P : MPSses
# maximize: Whether to minimize or maximize the objective functions
def solve_UFs(X: np.ndarray, P: np.ndarray, rw: float, pdw: np.ndarray | None, maximize=False):
    UF_vals = []
    UF_ws = []

    Q, K = X.shape[0], X.shape[1]
    # weights, rows DMS, columns objs. TODO: if providing different weights, change this. Now 1 for all.
    rw = rw  # factor to multiple rewards
    if pdw is None:
        pdw = np.ones((Q, K))
    # for each solution
    for i in range(len(X)):
        # for each DM. So this UF for each DM as an objective function for IOPIS should work?
        for p in range(len(P)):  # 4 DMs right now
            uf_val = UF(i, p, X, P, rw, pdw, maximize)
            UF_vals.append(uf_val)
        UF_ws.append(sum(UF_vals[i:i+len(P)]))
    print(len(UF_vals))
    print(len(UF_ws))

    """
    max_fair = np.argmax(UF_ws) # actually should probably use argmin if we are minimizing utilities? or not? 
    min_fair = np.argmin(UF_ws)
    # I think yeah, because min max, min is the argmin and max is in the function.
    print(X[max_fair])


    print(X[min_fair])
    """
    return UF_vals, UF_ws

def get_top_n_fair_solutions(solutions, UF_ws, n):
    idxs = np.argpartition(UF_ws, -n)[-n:]
    fair_sols = []
    for i in range(n):
        fair_sols.append(solutions[idxs[i]])
    return fair_sols

# TODO: comments
def scale_rp(problem: Problem, reference_point, ideal, nadir, maximize):
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
