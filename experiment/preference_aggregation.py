import numpy as np
from scipy.optimize import minimize
from desdeo.problem import (
    Problem,
    get_ideal_dict,
    get_nadir_dict,
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)
from desdeo.problem.schema import (
    Constant,
    Constraint,
    ConstraintTypeEnum,
    DiscreteRepresentation,
    ExtraFunction,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    Simulator,
    TensorConstant,
    TensorVariable,
    Variable,
    VariableTypeEnum,
)

# TODO: THIS DOES NOT WORK PROPERLY
# def find_GRP_old(
def find_GRP_simple(
    rps: np.ndarray,
    cip: np.ndarray,
    k: int,
    q: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    pref_agg_method: str
) -> np.ndarray:
    """
    Finds a group reference point by forming a subproblem optimizing the given fairness criteria.

    Args:
        rps (np.ndarray): the reference points from the DMs
        cip (np.ndarray): current iteration point
        k (int): the number of objective functions
        q (int): the number of DMs
        ideal (np.ndarray): the ideal vector of the problem
        nadir (np.ndarray): the nadir vector of the problem

    Returns:
        np.ndarray : the group reference point
    """
    bnds = []
    # bounds for objective functions, bounded by ideal and current iteration point.
    for i in range(k):
        bnds.append((ideal[i], cip[i]))

    # weight bounds for DMs. Can be between 0 and 1.
    for i in range(k, k+q):
        bnds.append((0, 1))

    # create X of first iteration guess
    # X = [RPs.., DM weights.., ALPHA]
    X = np.zeros(k+q)
    # Fill first guess for GRP by taking the mean of the RPs
    for i in range(k):
        X[i] = np.mean(rps[:, i])

    # Calculate the weights for DMs
    for qk in range(q):
        X[k+qk] = 1/q

    # constraints for feasible space
    def feas_space_const(X, k, q, i, rps):
        return lambda X: sum([X[k+j]*rps[j, i] for j in range(q)]) - X[i]

    # Create constraints for each objective function
    Cons = []
    for i in range(k):
        Cons.append({'type': 'eq', 'fun': feas_space_const(X, k, q, i, rps)})

    # Convex constraint, sum should be 1.
    # sum_{r=1}^4(lambda_r) - 1 = 0
    def convex_constr(X):
        return sum(X[k:k+q]) - 1

    conv = {'type': 'eq', 'fun': convex_constr}
    Cons.append(conv)

    rho = 0.0001

    # The s_m(R) for all DMs
    # TODO: standardization for maxmin (especially if not solving zdt1 and zdt2 only)
    if pref_agg_method == "maxmin" or pref_agg_method == "eq_maxmin":
        def fx(X):
            # TODO: recheck, if -1* should be inside maxmin terms or not!
            # max S. For scipy we need to convert to minimization.
            maxmin_terms = [maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            worst_off = np.min(maxmin_terms)
            to_maximize = -1*worst_off
            return to_maximize
            # return -1*np.min(maxmin_terms)  # + rho * np.sum(maxmin_terms) # was min
    if pref_agg_method == "maxmin_ext" or pref_agg_method == "eq_maxmin_ext":
        def fx(X):
            maxmin_terms = [maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            return -1 * (np.min(maxmin_terms) + rho * np.sum(maxmin_terms))
    if pref_agg_method == "maxmin_cones" or pref_agg_method == "eq_maxmin_cones":
        def fx(X):
            # TODO: these are equivalent.. not sure if it matters which to use.. in otherwords, which is closest to the og formulation i guess
            all_s_m = [maxmin_cones_criterion(cip, rps[j, :], X[:k]) for j in range(q)]
            s_m = -1*np.min(all_s_m)  # was min
            return s_m
    if pref_agg_method == "maxmin_cones_ext" or pref_agg_method == "eq_maxmin_cones":
        def fx(X):
            # TODO: these are equivalent.. not sure if it matters which to use.. in otherwords, which is closest to the og formulation i guess
            all_s_m = [maxmin_cones_criterion(cip, rps[j, :], X[:k]) for j in range(q)]
            s_m = -1 * (np.min(all_s_m) + rho * np.sum(all_s_m))
            return s_m

    solution = minimize(fx,
                        X,
                        bounds=bnds,
                        constraints=Cons,
                        # method = 'trust-constr',
                        # options={'xtol': 1e-20, 'gtol': 1e-20, 'maxiter': 10000, 'disp': True}
                        method='SLSQP',
                        options={'ftol': 1e-10, 'maxiter': 20000, 'disp': True}
                        # options={'ftol': 1e-20, 'maxiter': 20000, 'disp': True} # change later to more iters and toler
                        )

    # Decision variables (solutions)
    print(solution.x)
    group_RP = solution.x[0:k]
    return group_RP

# THIS is the version with separate aplha parameter to optimize. To handle the discontinuity on min()
def find_GRP_old(
    rps: np.ndarray,
    cip: np.ndarray,
    k: int,
    q: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    pref_agg_method: str
) -> np.ndarray:
    """
    Finds a group reference point by forming a subproblem optimizing the given fairness criteria.

    Args:
        rps (np.ndarray): the reference points from the DMs
        cip (np.ndarray): current iteration point
        k (int): the number of objective functions
        q (int): the number of DMs
        ideal (np.ndarray): the ideal vector of the problem
        nadir (np.ndarray): the nadir vector of the problem

    Returns:
        np.ndarray : the group reference point
    """

    alpha = k+q

    bnds = []
    # bounds for objective functions, bounded by ideal and current iteration point.
    for i in range(k):
        bnds.append((ideal[i], cip[i]))

    # weight bounds for DMs. Can be between 0 and 1.
    for i in range(k, k+q):
        bnds.append((0, 1))
    # bnds.append((-100000, 100000))
    bnds.append((-100000, 100000))

    # create X of first iteration guess
    # X = [RPs.., DM weights.., ALPHA]
    X = np.zeros(k+q+1)
    # Fill first guess for GRP by taking the mean of the RPs
    for i in range(k):
        X[i] = np.mean(rps[:, i])
        # X[i] = 0

    # Calculate the weights for DMs
    for qk in range(q):
        X[k+qk] = 1/q

    # constraints for DMs S
    def DMconstr_mm(X, q, k, rps, cip):
        # alpha - S_4(R) <= 0
        # FOR MAXMIN
        return lambda X: X[alpha] - np.min([maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)])

    rho = 0.0001

    def DMconstr_ext(X, q, k, rps, cip):
        # alpha - S_4(R) <= 0
        # FOR maxmin
        maxmin_terms = [maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
        print(maxmin_terms)
        return lambda X: X[alpha] - (np.min(maxmin_terms) - rho * np.sum(maxmin_terms))

    def DMconstr_cones(X, q, k, rps, cip):
        all_s_m = [maxmin_cones_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
        s_m = -1*np.min(all_s_m)
        return lambda X: X[alpha] - np.min(all_s_m)

    def DMconstr_cones_ext(X, q, k, rps, cip):
        all_s_m = [maxmin_cones_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
        s_m = -1*np.min(all_s_m)
        return lambda X: X[alpha] - (np.min(all_s_m) - rho * np.sum(all_s_m))
    # constraints for feasible space

    def feas_space_const(X, k, q, i, rps):
        return lambda X: sum([X[k+j]*rps[j, i] for j in range(q)]) - X[i]

    # Create constraints for each objective function
    Cons = []
    for i in range(k):
        Cons.append({'type': 'eq', 'fun': feas_space_const(X, k, q, i, rps)})

    if pref_agg_method == "maxmin_cones" or pref_agg_method == "eq_maxmin_cones":
        Cons.append({'type': 'ineq', 'fun': DMconstr_cones(X, q, k, rps, cip)})
    if pref_agg_method == "maxmin_cones_ext" or pref_agg_method == "eq_maxmin_cones_ext":
        Cons.append({'type': 'ineq', 'fun': DMconstr_cones_ext(X, q, k, rps, cip)})
    if pref_agg_method == "maxmin_ext" or pref_agg_method == "eq_maxmin_ext":
        Cons.append({'type': 'ineq', 'fun': DMconstr_ext(X, q, k, rps, cip)})
    else:
        Cons.append({'type': 'ineq', 'fun': DMconstr_mm(X, q, k, rps, cip)})

    # Convex constraint, sum should be 1.
    # sum_{r=1}^4(lambda_r) - 1 = 0
    def convex_constr(X):
        return sum(X[k:k+q]) - 1

    conv = {'type': 'eq', 'fun': convex_constr}
    Cons.append(conv)

    # rho = 0.0001

    # The s_m(R) for all DMs
    # TODO: standardization for maxmin (especially if not solving zdt1 and zdt2 only)
    if pref_agg_method == "maxmin" or pref_agg_method == "eq_maxmin":
        """ Have to convert to minimization for scipy.optimize. Hence Maximize min S_m(x) =>
        def fx(X):
            return -1*X[alpha] -1*np.min(maxmin_terms)
        """
        def fx(X):
            maxmin_terms = [maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            return -1*X[alpha] - np.min(maxmin_terms)  # + rho * np.sum(maxmin_terms)
    if pref_agg_method == "maxmin_ext" or pref_agg_method == "eq_maxmin_ext":
        def fx(X):
            """
            return -1*X[alpha]
            """
        def fx(X):
            maxmin_terms = [maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            return -1*X[alpha] - (np.min(maxmin_terms) + rho * np.sum(maxmin_terms))
    if pref_agg_method == "maxmin_cones" or pref_agg_method == "eq_maxmin_cones":
        """
        def fx(X):
            return -1*X[alpha]
        """
        def fx(X):
            # TODO: these are equivalent.. not sure if it matters which to use.. in otherwords, which is closest to the og formulation i guess
            all_s_m = [maxmin_cones_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            s_m = -np.min(all_s_m)
            return -1*X[alpha] + s_m
    if pref_agg_method == "maxmin_cones_ext" or pref_agg_method == "eq_maxmin_cones_ext":
        """
        def fx(X):
            return -1*X[alpha]
        """
        def fx(X):
            # TODO: these are equivalent.. not sure if it matters which to use.. in otherwords, which is closest to the og formulation i guess
            all_s_m = [maxmin_cones_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            s_m = -np.min(all_s_m)
            return -1*X[alpha] + s_m - rho * np.sum(all_s_m)

    solution = minimize(fx,
                        X,
                        bounds=bnds,
                        constraints=Cons,
                        # method = 'trust-constr',
                        # options={'xtol': 1e-20, 'gtol': 1e-20, 'maxiter': 10000, 'disp': True}
                        method='SLSQP',
                        options={'ftol': 1e-20, 'maxiter': 20000, 'disp': True}
                        # options={'ftol': 1e-20, 'maxiter': 20000, 'disp': True} # change later to more iters and toler
                        )

    # Decision variables (solutions)
    print(solution.x)
    group_RP = solution.x[0:k]
    return group_RP


def aggregate(
    problem: Problem,
    pref_agg_method: str,
    reference_points: dict[str, dict[str, float]],
    nav_point_arr=np.ndarray,
) -> np.ndarray:
    """Function to aggregate the preferences. Connects to find_GRP and handles conversions
    from dicts to np.ndarrays and using max_multiplier to convert the GRP to the true objective space.

    Args:
        problem (Problem): the problem being solved.
        pref_agg_method (str): the string depicting what preference aggregation method to use
        reference_points: (dict[str, dict[str, float]]): the reference points from the decision makers.
            The key is the objective function's symbol and the value is the aspired objective value.
        nav_point_arr (np.ndarray): the current navigation point in true objective space.

    Returns:
        np.ndarray : the group reference point in true objective space
    """
    # get problem information ideal, nadir and number of objectives (k) and number of DMs (q)
    group_reference_point = None
    max_multiplier = [-1 if obj.maximize else 1 for obj in problem.objectives]
    id_dict = get_ideal_dict(problem)
    nad_dict = get_nadir_dict(problem)
    k = len(id_dict)  # n_objs
    q = len(reference_points.keys())  # n of DMs
    ideal = objective_dict_to_numpy_array(problem, id_dict)
    nadir = objective_dict_to_numpy_array(problem, nad_dict)

    rp_a = {}
    for key, p in reference_points.items():
        rp_a[key] = list(np.array(list(p.values())) * np.array(max_multiplier))
    rp_arr = np.array(list(rp_a.values()))

    """
    TODO: LOGIC to select the proper aggregation function S
    Does not work because scipy, or not knowing how to do it with it.
    """
    # group_reference_point = find_GRP(rp_arr, nav_point_arr, k, q, ideal, nadir, pref_agg_method)
    # print("group RP", group_reference_point)
    group_reference_point = find_GRP(rp_arr, nav_point_arr, k, q, ideal, nadir, pref_agg_method)
    print("group RP", group_reference_point)

    return group_reference_point * max_multiplier  # GRP is in the true objective space


# THIS is the version with separate aplha parameter to optimize. To handle the discontinuity on min()
def find_GRP(
    rps: np.ndarray,
    cip: np.ndarray,
    k: int,
    q: int,
    ideal: np.ndarray,
    original_rps: np.ndarray,
    pref_agg_method: str
) -> np.ndarray:
    """
    Finds a group reference point by forming a subproblem optimizing the given fairness criteria.

    Args:
        rps (np.ndarray): the reference points from the DMs
        cip (np.ndarray): current iteration point
        k (int): the number of objective functions
        q (int): the number of DMs
        ideal (np.ndarray): the ideal vector of the problem
        nadir (np.ndarray): the nadir vector of the problem

    Returns:
        np.ndarray : the group reference point
    """

    """
    Forming the subproblem S for scipy.optimize.
    """

    alpha = k+q

    bnds = []
    # bounds for objective functions, bounded by ideal and current iteration point.
    for i in range(k):
        # bnds.append((ideal[i], cip[i]))
        bnds.append((ideal[i], 5))

    # weight bounds for DMs. Can be between 0 and 1.
    for i in range(k, k+q):
        bnds.append((0, 1))
    # bnds.append((-100000, 100000))
    bnds.append((-100000, 100000))

    # create R of first iteration guess
    # X = [RPs.., DM weights.., ALPHA]
    X = np.zeros(k+q+1)
    # Fill first guess for GRP by taking the mean of the RPs
    for i in range(k):
        X[i] = np.mean(rps[:, i])
        # X[i] = 0

    # Calculate the weights for DMs
    for qk in range(q):
        X[k+qk] = 1/q

    # constraints for DMs S
    def DMconstr_mm(X, q, k, rps, cip):
        # alpha - S_4(R) <= 0
        # FOR MAXMIN
        return lambda X: X[alpha] - np.min([maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)])

    rho = 0.0001
    # constraints for DMs S

    def DMconstr_ext(X, q, k, rps, cip):
        # alpha - S_4(R) <= 0
        # FOR maxmin
        maxmin_terms = [maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
        print(maxmin_terms)
        return lambda X: X[alpha] - (np.min(maxmin_terms) - rho * np.sum(maxmin_terms))

    def DMconstr_cones(X, q, k, rps, cip):
        all_s_m = [maxmin_cones_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
        # s_m = -1*np.min(all_s_m)
        return lambda X: X[alpha] - np.min(all_s_m)

    def DMconstr_cones_ext(X, q, k, rps, cip):
        all_s_m = [maxmin_cones_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
        # s_m = -1*np.min(all_s_m)
        return lambda X: X[alpha] - (np.min(all_s_m) - rho * np.sum(all_s_m))
    # constraints for feasible space

    def feas_space_const(X, k, q, i, rps):
        return lambda X: sum([X[k+j]*rps[j, i] for j in range(q)]) - X[i]

    # Create constraints for each objective function
    Cons = []
    for i in range(k):
        if pref_agg_method == "eq_maxmin_cones" or pref_agg_method == "eq_maxmin" or pref_agg_method == "eq_maxmin_ext" or pref_agg_method == "eq_maxmin_cones_ext":
            Cons.append({'type': 'eq', 'fun': feas_space_const(X, k, q, i, original_rps)})
        else:
            Cons.append({'type': 'eq', 'fun': feas_space_const(X, k, q, i, rps)})

    # rps = original_rps
    if pref_agg_method == "maxmin_cones" or pref_agg_method == "eq_maxmin_cones":
        Cons.append({'type': 'ineq', 'fun': DMconstr_cones(X, q, k, rps, cip)})
    if pref_agg_method == "maxmin_cones_ext" or pref_agg_method == "eq_maxmin_cones_ext":
        Cons.append({'type': 'ineq', 'fun': DMconstr_cones_ext(X, q, k, rps, cip)})
    if pref_agg_method == "maxmin_ext" or pref_agg_method == "eq_maxmin_ext":
        Cons.append({'type': 'ineq', 'fun': DMconstr_ext(X, q, k, rps, cip)})
    else:
        Cons.append({'type': 'ineq', 'fun': DMconstr_mm(X, q, k, rps, cip)})
    # Convex constraint, sum should be 1.
    # sum_{r=1}^4(lambda_r) - 1 = 0

    def convex_constr(X):
        return sum(X[k:k+q]) - 1

    conv = {'type': 'eq', 'fun': convex_constr}
    Cons.append(conv)

    # rho = 0.0001
    # rps = original_rps
    # The s_m(R) for all DMs
    # TODO: standardization for maxmin (especially if not solving zdt1 and zdt2 only)
    if pref_agg_method == "maxmin" or pref_agg_method == "eq_maxmin":
        """ Have to convert to minimization for scipy.optimize. Hence Maximize min S_m(x) =>
        def fx(X):
            return -1*X[alpha] -1*np.min(maxmin_terms)
        """
        def Sm(X):
            maxmin_terms = [maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            return -1*X[alpha] - np.min(maxmin_terms)  # + rho * np.sum(maxmin_terms)
            # return - np.min(maxmin_terms)  # + rho * np.sum(maxmin_terms)
    if pref_agg_method == "maxmin_ext" or pref_agg_method == "eq_maxmin_ext":
        def Sm(X):
            maxmin_terms = [maxmin_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            return -1*X[alpha] - (np.min(maxmin_terms) + rho * np.sum(maxmin_terms))
            # return - (np.min(maxmin_terms) + rho * np.sum(maxmin_terms))
    if pref_agg_method == "maxmin_cones" or pref_agg_method == "eq_maxmin_cones":
        """
        def fx(X):
            return -1*X[alpha]
        """
        def Sm(X):
            # TODO: these are equivalent.. not sure if it matters which to use.. in otherwords, which is closest to the og formulation i guess
            all_s_m = [maxmin_cones_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            s_m = np.min(all_s_m)
            return -1*X[alpha] - s_m
            # return - s_m
    if pref_agg_method == "maxmin_cones_ext" or pref_agg_method == "eq_maxmin_cones_ext":
        """
        def fx(X):
            return -1*X[alpha]
        """
        def Sm(X):
            # TODO: these are equivalent.. not sure if it matters which to use.. in otherwords, which is closest to the og formulation i guess
            all_s_m = [maxmin_cones_criterion(rps[j, :], cip, X[:k]) for j in range(q)]
            s_m = np.min(all_s_m)
            return -1*X[alpha] - s_m + rho * np.sum(all_s_m)
            # return - (s_m + rho * np.sum(all_s_m))

    solution = minimize(fun=Sm,
                        x0=X,
                        bounds=bnds,
                        constraints=Cons,
                        method='SLSQP',
                        options={'ftol': 1e-20, 'maxiter': 20000,
                                 'eps': 1e-10,
                                 'disp': True}
                        )

    # Decision variables (solutions)
    print(solution.x)
    group_RP = solution.x[0:k]
    return group_RP


# The old not normalized maxmin criterion.
# p = r1 DM's suggested point,
# c = r0, current iteration point
# r = R, suggested group point.
def not_normalized_maxmin_criterion(p, c, r):
    return np.sum((p - c)*r)


# p = r0, current iteration point
# c = r1 DM's suggested point,
# r = R, suggested group point.
def maxmin_cones_criterion(c, p, r):
    return eval_RP(c, p, r)


# given a search direction from old CIP RO
# to new suggested point R1,
# evaluate point P using a cone model
def eval_RP(R1, R0, P, a=0.5):
    # calc dir vector.
    D = R1 - R0
    # normalize the direction vector D
    # D = D_og/np.sqrt(np.sum(D_og**2))
    # constant of hyperplane going through P
    cv = np.matmul(D, P)
    # express as linear combination between R0 and R1
    tv = (cv - np.matmul(D, R1)) / (np.matmul(D, R0)-np.matmul(D, R1))
    # point B
    B = (tv*R0+(1-tv)*R1)
    # calculate direction vector V
    V = P - B
    # normalize vectros
    D1 = D/np.sqrt(np.sum(D**2))
    # V1 = V/np.sqrt(np.sum(V**2))

    # via tan beta, length of XB
    lXB = np.sqrt(np.sum(V**2)) * a/(1-a)
    # location of point X
    X = B - lXB * D1
    # print(X)
    # all components of the eval_value should be the same
    eval_value = (X-R0)/(R1-R0)
    # print("eval_list",eval_value)
    # return the first finite component.
    # print("eval_values\r\n", eval_value)
    if any(abs(eval_value - eval_value[0] > 1e-5)):
        print("this should not happen")
    eval_value2 = eval_value[np.isfinite(eval_value)][0]
    # print(eval_value2)
    return eval_value2


"""
 Normalized version of maxmin_criterion, the new normal
"""
# ! NOTE different symbols for q, p and R
# q = r1 DM's suggested point,
# p = r0, current iteration point
# r = R, suggested group point.
def maxmin_criterion(q, p, r):
    left_top = np.sum((q - p)*r)
    right_eq = np.sum((q - p)*p)
    left_b = np.sum((q - p)*q)
    up = (left_top - right_eq)
    denom = (left_b - right_eq)
    if denom == 0:
        return 0
    else:
        return up / denom
    # value = (left_top - right_eq) / ((left_b - right_eq) + 0.000001)  # prevent division by zero
    #    return value


def simple_linear_test_problem() -> Problem:
    """Defines a simple single objective linear problem suitable for testing purposes."""
    variables = [
        Variable(name="x_1", symbol="x_1", variable_type="real", lowerbound=-10, upperbound=10, initial_value=5),
        Variable(name="x_2", symbol="x_2", variable_type="real", lowerbound=-10, upperbound=10, initial_value=5),
    ]

    constants = [Constant(name="c", symbol="c", value=4.2)]

    f_1 = "x_1 + x_2"

    objectives = [
        Objective(name="f_1", symbol="f_1", func=f_1, maximize=False),  # min!
    ]

    con_1 = Constraint(name="g_1", symbol="g_1", cons_type=ConstraintTypeEnum.LTE, func="c - x_1")
    con_2 = Constraint(name="g_2", symbol="g_2", cons_type=ConstraintTypeEnum.LTE, func="0.5*x_1 - x_2")

    return Problem(
        name="Simple linear test problem.",
        description="A simple problem for testing purposes.",
        constants=constants,
        variables=variables,
        constraints=[con_1, con_2],
        objectives=objectives,
    )

def subproblem(rps, cip, k, q, ideal) -> Problem:
    # variables for the amount of objectives (k) in the original problem
    # bounds come from the ideal and nadir of the original problem
    # initial value is the mean.

    variables = []
    variables.append(
        Variable(name="alpha", symbol="a", variable_type="real", lowerbound=-100, upperbound=100, initial_value=1),
    )
    for i in range(k):
        variables.append(
            Variable(name=f"x_{i}", symbol=f"x_{i}", variable_type="real", lowerbound=ideal[i], upperbound=cip[i], initial_value=np.mean(rps[:, i])),
        )

    # need variables for DMs' weights
    for i in range(q):
        variables.append(
            Variable(name=f"w_{i}", symbol=f"w_{i}", variable_type="real", lowerbound=0, upperbound=1, initial_value=1/q),
        )
    # the DMs' rps are constants and cip
    constants = []
    for i in range(k):
        constants.append(
            Constant(name=f"cip_{i}", symbol=f"p_{i}", value=cip[i])
        )
    for m in range(q):
        for i in range(k):
            constants.append(
                Constant(name=f"dm_{m}_rp_{i}", symbol=f"dm_{m}_q_{i}", value=rps[m, i])
            )

    # DMconst = [] # not needed?
    # feasible_const = []
    constraints = []
    """
    for i in range(k):
        for m in range(q):
            con = (
                Constraint(
                    name=f"Feasible space constraint for objective {i}",
                    symbol=f"fs_{i}",
                    cons_type=ConstraintTypeEnum.EQ,
                    func=f"Sum(w_{i}*dm_{m}_q_{i}) - x_{i}",
                    is_linear=True,
                    is_convex=True,
                    is_twice_differentiable=True,
                )
            )
        constraints.append(con)
    """

    for i in range(k):
        con = (
            Constraint(
                name=f"Feasible space constraint for objective {i}",
                symbol=f"fs_{i}",
                cons_type=ConstraintTypeEnum.EQ,  # should be EQ, does not work
                func=f"( w_0*dm_0_q_{i} + w_1*dm_1_q_{i} +  w_2*dm_2_q_{i} )  - x_{i}",
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )
        constraints.append(con)

    constraints.append(
        Constraint(
            name="Convexity constraint",
            symbol="c_w",
            cons_type=ConstraintTypeEnum.EQ,
            # func=f"(w_{m}) - 1",
            func="(w_0 + w_1 + w_2) - 1",
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )
    )

    dm1_expr = "" + " + ".join([f"(dm_{0}_q_{i} - p_{i}) * x_{i} " for i in range(k)])
    dm2_expr = "" + " + ".join([f" (dm_{1}_q_{i} - p_{i}) * x_{i} " for i in range(k)])
    dm3_expr = "" + " + ".join([f" (dm_{2}_q_{i} - p_{i}) * x_{i} " for i in range(k)])

    min_term = f"Min(({dm1_expr}),({dm2_expr}), ({dm3_expr}) )"

    constraints.append(
        Constraint(
            name="DM1 const",
            symbol="dm1",
            cons_type=ConstraintTypeEnum.LTE,
            # func=f"(w_{m}) - 1",
            # func=f"a - {dm1_expr}",
            func=f"a- {dm1_expr} ",
            is_linear=True,
            # is_convex=True,
            is_twice_differentiable=True,
        )
    )
    constraints.append(
        Constraint(
            name="DM1 const",
            symbol="dm2",
            cons_type=ConstraintTypeEnum.LTE,
            # func=f"(w_{m}) - 1",
            func=f" a- {dm2_expr}",
            is_linear=True,
            # is_convex=True,
            is_twice_differentiable=True,
        )
    )
    constraints.append(
        Constraint(
            name="DM1 const",
            symbol="dm3",
            cons_type=ConstraintTypeEnum.LTE,
            # func=f"(w_{m}) - 1",
            func=f"a- {dm3_expr} ",
            is_linear=True,
            # is_convex=True,
            is_twice_differentiable=True,
        )
    )
    # value_for_dm_i = f" (dm_{m}_q_{i} - p_{i}) * x_{i} ) "
    # dm_expr = "(" + " + ".join([f"dm_{m}_q_{i} - p_{i}) * x_{i} " for i in range(k) for m in range(q)]) + ")"
    # maxmin_term = f" (dm_{m}_q_{i} - p_{i}) * x_{i} ) "
    # dm_expr += "+".join(value_for_dm_i)
    # maxmin_terms.append(value_for_dm)
    # dm_expr = "(" + " + ".join([f"( (dm_{m}_q_{i} - p_{i}) * x_{i} )" for i in range(k) for m in range(q)]) + ")"
    # maxmin_str = "".join(dm_expr1)
    # maxmin_str += ",".join(dm_expr2)  # + ",".join(dm_expr3)
    # maxmin_terms = "+".join(maxmin_terms)
    # sum_term = f"Sum{maxmin_terms}"
    # min_term = f"Min({maxmin_terms}"
    # maxmin_str = "".join(dm_expr1) + ",".join(dm_expr2) + ",".join(dm_expr3)# expr.append(dm1_expr)
    # expr.append(dm2_expr)
    # expr.append(dm3_expr)

    # maxmin_fairness = f"Min ({dm1_expr}, {dm2_expr}, {dm3_expr} )"
    # maxmin_fairness = f" 1 + Min ({dm1_expr}, 0) + 1"
    # maxmin_fairness = f"Min ({expr})"

    # maxmin_fairness = "1 + Min( 0, 1, 2 )"
    # maxmin_fairness = f" - a + Max(({dm1_expr}),({dm2_expr}), ({dm3_expr}) )"
    # maxmin_fairness = f" - a + {min_term}"
    # maxmin_fairness = f"a - {min_term}"
    # maxmin_fairness = "a"
    maxmin_fairness = "Min((dm_0_q_0 - p_0) * x_0 + (dm_0_q_1 - p_1) * x_1, (dm_1_q_0 - p_0) * x_0 + (dm_1_q_1 - p_1) * x_1, (dm_2_q_0 - p_0) * x_0 + (dm_2_q_1 - p_1) * x_1)"

    # func=f"Max(({x_1_eprs}) * x_3 - 7.735 * (({x_1_eprs})**2 / x_2) - 180, 0) + Max(4 - x_3 / x_2, 0)",

    objective = [
        Objective(
            name="f_1",
            symbol="f_1",
            func=maxmin_fairness,
            objective_type=ObjectiveTypeEnum.analytical,
            # is_linear=True,
            # is_convex=True,
            is_twice_differentiable=True,
            maximize=True,
        )
    ]

    return Problem(
        name="maxmin subproblem",
        description="subproblem for aggregation of reference points according to maxmin fairness",
        constants=constants,
        variables=variables,
        # constraints=[feasible_const, convex_const],
        constraints=constraints,
        objectives=objective,
    )


def subproblem2(rps, cip, k, q, ideal) -> Problem:
    # variables for the amount of objectives (k) in the original problem
    # bounds come from the ideal and nadir of the original problem
    # initial value is the mean.

    variables = []
    for i in range(k):
        variables.append(
            Variable(name=f"x_{i}", symbol=f"x_{i}", variable_type="real", lowerbound=ideal[i], upperbound=cip[i], initial_value=np.mean(rps[:, i])),
        )

    # need variables for DMs' weights
    for i in range(q):
        variables.append(
            Variable(name=f"w_{i}", symbol=f"w_{i}", variable_type="real", lowerbound=0, upperbound=1, initial_value=1/q),
        )
    # the DMs' rps are constants and cip
    constants = []
    for i in range(k):
        constants.append(
            Constant(name=f"cip_{i}", symbol=f"p_{i}", value=cip[i])
        )
    for m in range(q):
        for i in range(k):
            constants.append(
                Constant(name=f"dm_{m}_rp_{i}", symbol=f"dm_{m}_q_{i}", value=rps[m, i])
            )

    # feasible_const = []
    constraints = []
    """
    for i in range(k):
        for m in range(q):
            con = (
                Constraint(
                    name=f"Feasible space constraint for objective {i}",
                    symbol=f"fs_{i}",
                    cons_type=ConstraintTypeEnum.EQ,
                    func=f"Sum(w_{i}*dm_{m}_q_{i}) - x_{i}",
                    is_linear=True,
                    is_convex=True,
                    is_twice_differentiable=True,
                )
            )
        constraints.append(con)
    """

    for i in range(k):
        con = (
            Constraint(
                name=f"Feasible space constraint for objective {i}",
                symbol=f"fs_{i}",
                cons_type=ConstraintTypeEnum.EQ,  # should be EQ, does not work
                func=f"( w_0*dm_0_q_{i} + w_1*dm_1_q_{i} +  w_2*dm_2_q_{i} )  - x_{i}",
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )
        constraints.append(con)

    constraints.append(
        Constraint(
            name="Convexity constraint",
            symbol="c_w",
            cons_type=ConstraintTypeEnum.EQ,
            # func=f"(w_{m}) - 1",
            func="(w_0 + w_1 + w_2) - 1",
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )
    )

    dm1_expr = "" + " + ".join([f"(dm_{0}_q_{i} - p_{i}) * x_{i} " for i in range(k)])
    dm2_expr = "" + " + ".join([f" (dm_{1}_q_{i} - p_{i}) * x_{i} " for i in range(k)])
    dm3_expr = "" + " + ".join([f" (dm_{2}_q_{i} - p_{i}) * x_{i} " for i in range(k)])

    min_term = f"Min(({dm1_expr}),({dm2_expr}), ({dm3_expr}) )"
    # 'Min(((dm_0_q_0 - p_0) * x_0  + (dm_0_q_1 - p_1) * x_1 ),  ( (dm_1_q_0 - p_0) * x_0  + (dm_1_q_1 - p_1) * x_1 ), ( (dm_2_q_0 - p_0) * x_0  +  (dm_2_q_1 - p_1) * x_1 ) )'
    # min_term = f"(({dm1_expr}))"
    # maxmin_fairness = "Min( ((dm_0_q_0 - p_0) * x_0 + (dm_0_q_1 - p_1) * x_1 ), ((dm_1_q_0 - p_0) * x_0 + (dm_1_q_1 - p_1) * x_1), ((dm_2_q_0 - p_0) * x_0 + (dm_2_q_1 - p_1) * x_1))"

    maxmin_fairness = f" {min_term}"

    # func=f"Max(({x_1_eprs}) * x_3 - 7.735 * (({x_1_eprs})**2 / x_2) - 180, 0) + Max(4 - x_3 / x_2, 0)",

    objective = [
        Objective(
            name="f_1",
            symbol="f_1",
            func=maxmin_fairness,
            objective_type=ObjectiveTypeEnum.analytical,
            # is_linear=True,
            # is_convex=True,
            is_twice_differentiable=True,
            maximize=True,
        )
    ]

    return Problem(
        name="maxmin subproblem",
        description="subproblem for aggregation of reference points according to maxmin fairness",
        constants=constants,
        variables=variables,
        # constraints=[feasible_const, convex_const],
        constraints=constraints,
        objectives=objective,
    )

def simple_test_problem2() -> Problem:
    """Defines a simple problem suitable for testing purposes."""
    variables = [
        Variable(name="x_1", symbol="x_1", variable_type="real", lowerbound=0, upperbound=10, initial_value=5),
        Variable(name="x_2", symbol="x_2", variable_type="real", lowerbound=0, upperbound=10, initial_value=5),
    ]

    constants = [Constant(name="c", symbol="c", value=4.2)]

    ff = f"x_1 + x_2"
    ff2 = f"x_1 + x_2"

    f_1 = "x_1 + x_2"
    f_2 = "x_2**3"
    f_3 = "x_1 + x_2"
    f_4 = f"Max(Abs(x_1 - x_2), c, {ff}, {ff2})"  # c = 4.2
    f_5 = "(-x_1) * (-x_2)"

    objectives = [
        Objective(name="f_1", symbol="f_1", func=f_1, maximize=False,
                  is_twice_differentiable=True,
                  ),  # min!
        Objective(name="f_2", symbol="f_2", func=f_2, maximize=True,

                  is_twice_differentiable=True,

                  ),  # max!
        Objective(name="f_3", symbol="f_3", func=f_3, maximize=True,

                  is_twice_differentiable=True,
                  ),  # max!
        Objective(name="f_4", symbol="f_4", func=f_4, maximize=False,

                  is_twice_differentiable=True,
                  ),  # min!
        Objective(name="f_5", symbol="f_5", func=f_5, maximize=True,
                  is_twice_differentiable=True,
                  ),  # max!
    ]

    return Problem(
        name="Simple test problem.",
        description="A simple problem for testing purposes.",
        constants=constants,
        variables=variables,
        objectives=objectives,
    )
