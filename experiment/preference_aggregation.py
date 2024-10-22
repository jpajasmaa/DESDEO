import numpy as np
from scipy.optimize import minimize
from desdeo.problem import (
    Problem,
    get_ideal_dict,
    get_nadir_dict,
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)

def find_GRP(
    rps: np.ndarray, 
    cip: np.ndarray, 
    k: int, 
    q: int, 
    ideal: np.ndarray, 
    nadir: np.ndarray, 
    pref_agg_method:str
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
        bnds.append((ideal[i],cip[i]))

    # weight bounds for DMs. Can be between 0 and 1.
    for i in range(k, k+q):
        bnds.append((0,1))

    # create X of first iteration guess
    # X = [RPs.., DM weights.., ALPHA]
    X = np.zeros(k+q) 
    # Fill first guess for GRP by taking the mean of the RPs
    for i in range(k):
        X[i] = np.mean(rps[:,i]) 

    # Calculate the weights for DMs
    for qk in range(q):
        X[k+qk] = 1/q

    # constraints for feasible space
    def feas_space_const(X, k, q, i, rps):
        return lambda X: sum([X[k+j]*rps[j,i] for j in range(q)]) - X[i]

    # Create constraints for each objective function
    Cons = []
    for i in range(k):
        Cons.append({'type':'eq', 'fun' : feas_space_const(X, k, q, i, rps)})

    # Convex constraint, sum should be 1.
    # sum_{r=1}^4(lambda_r) - 1 = 0
    def convex_constr(X):
        return sum(X[k:k+q]) - 1

    conv ={'type':'eq', 'fun':convex_constr}
    Cons.append(conv)

    # The s_m(R) for all DMs
    if pref_agg_method == "maxmin" or pref_agg_method == "eq_maxmin":
        def fx(X):
            return -1*np.min([maxmin_criterion(rps[j,:], cip, X[:k]) for j in range(q)])
    if pref_agg_method == "maxmin_cones" or pref_agg_method == "eq_maxmin_cones":
        def fx(X):
            #all_s_m = [-maxmin_cones_criterion(cip, rps[j,:], X[:k]) for j in range(q)]
            #print(all_s_m)
            #s_m = np.max(all_s_m)
            # TODO: these are equivalent.. not sure if it matters which to use.. in otherwords, which is closest to the og formulation i guess
            all_s_m = [maxmin_cones_criterion(cip, rps[j,:], X[:k]) for j in range(q)]
            s_m = -1*np.min(all_s_m)
            return s_m
      
    solution = minimize(fx,
        X,
        bounds = bnds,
        constraints = Cons,   
        method = 'trust-constr',
        options={'xtol': 1e-20, 'gtol': 1e-20, 'maxiter': 10000, 'disp': True}
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
    """Function to aggregate the preferecnes. Connects to find_GRP and handles conversions 
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

    group_reference_point = find_GRP(rp_arr, nav_point_arr, k, q, ideal, nadir, pref_agg_method)
    print("group RP", group_reference_point) 

    return group_reference_point * max_multiplier  # GRP is in the true objective space


# p = r1 DM's suggested point,
# c = r0, current iteration point
# r = R, suggested group point. 
def maxmin_criterion(p, c, r):
    return np.sum((p - c)*r)


# c = r0, current iteration point
# p = r1 DM's suggested point,
# r = R, suggested group point. 
def maxmin_cones_criterion(c, p, r):
    return eval_RP(c, p, r)


# given a search direction from old CIP RO to new suggested point R1, evaluate point P using a cone model
def eval_RP(R0, R1, P, a=0.5):
    # calc dir vector.
    D = R1 - R0
    # normalize the direction vector D
    #D = D_og/np.sqrt(np.sum(D_og**2))
    # constant of hyperplane going through P
    cv = np.matmul(D, P)
    # express as linear combination between R0 and R1
    tv = (cv - np.matmul(D, R1)) / (np.matmul(D, R0)-np.matmul(D, R1))
    # point B
    B = (tv*R0+(1-tv)*R1)
    # calculate direction vector V
    V = P - B
    #normalize vectros
    D1 = D/np.sqrt(np.sum(D**2))
    #V1 = V/np.sqrt(np.sum(V**2))

    #via tan beta, length of XB
    lXB = np.sqrt(np.sum(V**2)) * a/(1-a)
    # location of point X
    X = B - lXB * D1
    #print(X)
    # all components of the eval_value should be the same
    eval_value = (X-R0)/(R1-R0)
    #print("eval_list",eval_value)
    # return the first finite component.
    #print("eval_values\r\n", eval_value)
    eval_value2 = eval_value[np.isfinite(eval_value)][0]
    #print(eval_value2)
    return eval_value2
