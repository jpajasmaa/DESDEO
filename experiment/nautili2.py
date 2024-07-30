"""Methods for the NAUTILI (a group decision making variant for NAUTILUS) method."""

import numpy as np
from pydantic import BaseModel, Field

import pandas as pd

from desdeo.mcdm.nautilus_navigator import (
    calculate_distance_to_front,
    calculate_navigation_point,
)
from desdeo.problem import (
    Problem,
    get_ideal_dict,
    get_nadir_dict,
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)

from desdeo.problem import (
    binh_and_korn,
    momip_ti7,
    objective_dict_to_numpy_array,
    river_pollution_problem,
    get_nadir_dict,
    get_ideal_dict,
    dtlz2,
    #variable_dict_to_numpy_array,
    objective_dict_to_numpy_array
    )
from desdeo.tools import (
        BonminOptions,
        NevergradGenericOptions,
        NevergradGenericSolver,
        PyomoBonminSolver,
        ScipyMinimizeSolver,
    )

from desdeo.tools.generics import CreateSolverType, SolverResults
from desdeo.tools.scalarization import (
    add_asf_generic_nondiff,
    add_asf_generic_diff,
    add_asf_nondiff,
    add_epsilon_constraints,
    add_lte_constraints,
    add_scalarization_function,
)
from desdeo.tools.utils import guess_best_solver

from testprobs import *

class NAUTILI_Response(BaseModel):
    """The response of the NAUTILI method."""

    step_number: int = Field(description="The step number associted with this response.")
    distance_to_front: float = Field(
        description=(
            "The distance travelled to the Pareto front. "
            "The distance is a ratio of the distances between the nadir and navigation point, and "
            "the nadir and the reachable objective vector. The distance is given in percentage."
        )
    )
    reference_points: dict | None = Field(description="The reference point used in the step.")
    improvement_directions: dict | None = Field(description="The improvement directions for each DM.")
    group_improvement_direction: dict | None = Field(description="The group improvement direction.")
    group_reference_point: dict | None = Field(description="The group reference point.")
    navigation_point: dict = Field(description="The navigation point used in the step.")
    reachable_solution: dict | None = Field(description="The reachable solution found in the step.")
    reachable_bounds: dict = Field(description="The reachable bounds found in the step.")


class NautiliError(Exception):
    """Exception raised for errors in the NAUTILI method."""


def solve_reachable_bounds(
    problem: Problem, navigation_point: dict[str, float], create_solver: CreateSolverType | None = None
) -> tuple[dict[str, float], dict[str, float]]:
    """Computes the current reachable (upper and lower) bounds of the solutions in the objective space.

    The reachable bound are computed based on the current navigation point. The bounds are computed by
    solving an epsilon constraint problem.

    Args:
        problem (Problem): the problem being solved.
        navigation_point (dict[str, float]): the navigation point limiting the
            reachable area. The key is the objective function's symbol and the value
            the navigation point.
        create_solver (CreateSolverType | None, optional): a function of type CreateSolverType that returns a solver.
            If None, then a solver is utilized bases on the problem's properties. Defaults to None.

    Raises:
        NautiliError: when optimization of an epsilon constraint problem is not successful.

    Returns:
        tuple[dict[str, float], dict[str, float]]: a tuple of dicts, where the first dict are the lower bounds and the
            second element the upper bounds, the key is the symbol of each objective.
    """
    # If an objective is to be maximized, then the navigation point component of that objective should be
    # multiplied by -1.
    const_bounds = {
        objective.symbol: -1 * navigation_point[objective.symbol]
        if objective.maximize
        else navigation_point[objective.symbol]
        for objective in problem.objectives
    }

    # if a solver creator was provided, use that, else, guess the best one
    solver_init = guess_best_solver(problem) if create_solver is None else create_solver

    lower_bounds = {}
    upper_bounds = {}
    for objective in problem.objectives:
        eps_problem, target, _ = add_epsilon_constraints(
            problem,
            "target",
            {f"{obj.symbol}": f"{obj.symbol}_eps" for obj in problem.objectives},
            objective.symbol,
            const_bounds,
        )

        # solve
        solver = solver_init(eps_problem)
        res = solver.solve(target)

        if not res.success:
            # could not optimize eps problem
            msg = (
                f"Optimizing the epsilon constrait problem for the objective "
                f"{objective.symbol} was not successful. Reason: {res.message}"
            )
            raise NautiliError(msg)

        lower_bound = res.optimal_objectives[objective.symbol]

        if isinstance(lower_bound, list):
            lower_bound = lower_bound[0]

        # solver upper bounds
        # the lower bounds is set as in the NAUTILUS method, e.g., taken from
        # the current iteration/navigation point
        if isinstance(navigation_point[objective.symbol], list):
            # It should never be a list accordint to the type hints
            upper_bound = navigation_point[objective.symbol][0]
        else:
            upper_bound = navigation_point[objective.symbol]

        # add the lower and upper bounds logically depending whether an objective is to be maximized or minimized
        lower_bounds[objective.symbol] = lower_bound if not objective.maximize else upper_bound
        upper_bounds[objective.symbol] = upper_bound if not objective.maximize else lower_bound

    return lower_bounds, upper_bounds


def solve_reachable_solution(
    problem: Problem,
    group_improvement_direction: dict[str, float],
    previous_nav_point: dict[str, float],
    create_solver: CreateSolverType | None = None,
) -> SolverResults:
    """Calculates the reachable solution on the Pareto optimal front.

    For the calculation to make sense in the context of NAUTILI, the reference point
    should be bounded by the reachable bounds present at the navigation step the
    reference point has been given.

    In practice, the reachable solution is calculated by solving an achievement
    scalarizing function.

    Args:
        problem (Problem): the problem being solved.
        group_improvement_direction (dict[str, float]): the improvement direction for the group.
        previous_nav_point (dict[str, float]): the previous navigation point. The reachable solution found
            is always better than the previous navigation point.
        create_solver (CreateSolverType | None, optional): a function of type CreateSolverType that returns a solver.
            If None, then a solver is utilized bases on the problem's properties. Defaults to None.

    Returns:
        SolverResults: the results of the projection.
    """
    # check solver
    init_solver = guess_best_solver(problem) if create_solver is None else create_solver

    # create and add scalarization function
    # previous_nav_point = objective_dict_to_numpy_array(problem, previous_nav_point).tolist()
    # weights = objective_dict_to_numpy_array(problem, group_improvement_direction).tolist()
    
    problem_w_asf, target = add_asf_generic_diff(
        problem,
        symbol="asf",
        reference_point=previous_nav_point,
        weights=group_improvement_direction,
        reference_point_aug=group_improvement_direction,
    )
    
    # makes maxmincones better in some cases
    """
    problem_w_asf, target = add_asf_nondiff(
        problem,
        symbol="asf",
        reference_point=group_improvement_direction,
        #weights=group_improvement_direction,
        reference_in_aug=True,
    )
    """

    # Note: We do not solve the global problem. Instead, we solve this constrained problem:
    const_exprs = [
        f"{obj.symbol}_min - {previous_nav_point[obj.symbol] * (-1 if obj.maximize else 1)}"
        for obj in problem.objectives
    ]
    problem_w_asf = add_lte_constraints(
        problem_w_asf, const_exprs, [f"const_{i}" for i in range(1, len(const_exprs) + 1)]
    )

    # solve the problem
    solver = init_solver(problem_w_asf)
    return solver.solve(target)


def nautili_init(problem: Problem, create_solver: CreateSolverType | None = None) -> NAUTILI_Response:
    """Initializes the NAUTILI method.

    Creates the initial response of the method, which sets the navigation point to the nadir point
    and the reachable bounds to the ideal and nadir points.

    Args:
        problem (Problem): The problem to be solved.
        create_solver (CreateSolverType | None, optional): The solver to use. Defaults to ???.

    Returns:
        NAUTILUS_Response: The initial response of the method.
    """
    nav_point = get_nadir_dict(problem)
    lower_bounds, upper_bounds = solve_reachable_bounds(problem, nav_point, create_solver=create_solver)
    return NAUTILI_Response(
        distance_to_front=0,
        navigation_point=nav_point,
        reachable_bounds={"lower_bounds": lower_bounds, "upper_bounds": upper_bounds},
        reachable_solution=None,
        reference_points=None,
        improvement_directions=None,
        group_improvement_direction=None,
        group_reference_point=None,
        step_number=0,
    )


def nautili_step(  # NOQA: PLR0913
    problem: Problem,
    steps_remaining: int,
    step_number: int,
    nav_point: dict,
    create_solver: CreateSolverType | None = None,
    group_improvement_direction: dict | None = None,
    group_reference_point: dict | None = None,
    reachable_solution: dict | None = None,
) -> NAUTILI_Response:
    if group_improvement_direction is None and reachable_solution is None:
        raise NautiliError("Either group_improvement_direction or reachable_solution must be provided.")

    if group_improvement_direction is not None and reachable_solution is not None:
        raise NautiliError("Only one of group_improvement_direction or reachable_solution should be provided.")

    if group_improvement_direction is not None:
        opt_result = solve_reachable_solution(problem, group_improvement_direction, nav_point, create_solver)
        reachable_solution = opt_result.optimal_objectives

    # update nav point
    new_nav_point = calculate_navigation_point(problem, nav_point, reachable_solution, steps_remaining)

    # update_bounds

    lower_bounds, upper_bounds = solve_reachable_bounds(problem, new_nav_point, create_solver=create_solver)

    distance = calculate_distance_to_front(problem, new_nav_point, reachable_solution)

    return NAUTILI_Response(
        step_number=step_number,
        distance_to_front=distance,
        reference_points=None,
        improvement_directions=None,
        group_improvement_direction=group_improvement_direction,
        group_reference_point = group_reference_point,
        navigation_point=new_nav_point,
        reachable_solution=reachable_solution,
        reachable_bounds={"lower_bounds": lower_bounds, "upper_bounds": upper_bounds},
    )


# TODO: changed this for the experiments. Will aggregate here.
# Always computes the direction, only to be used when testing this.
def nautili_step2(  # NOQA: PLR0913
    problem: Problem,
    steps_remaining: int,
    step_number: int,
    nav_point: dict,
    reference_points: dict[str, dict[str, float]],
    create_solver: CreateSolverType | None = None,
    group_improvement_direction: dict | None = None,
    group_reference_point: dict | None = None,
    reachable_solution: dict | None = None,
    pref_agg_method: str | None = None,
) -> NAUTILI_Response:

    #if group_improvement_direction is None and reachable_solution is None:
    #    raise NautiliError("Either group_improvement_direction or reachable_solution must be provided.")

    #if group_improvement_direction is not None and reachable_solution is not None:
    #    raise NautiliError("Only one of group_improvement_direction or reachable_solution should be provided.")

    # Calculate the improvement directions for each DM
    improvement_directions = {}
    group_reference_point = None
    for dm in reference_points:
        if reference_points[dm] is None:
            # If no reference point is provided, use the previous improvement direction
            if previous_responses[-1].reference_points is None:
                raise NautiliError("A reference point must be provided for the first iteration.")
            if previous_responses[-1].improvement_directions is None:
                raise NautiliError("An improvement direction must be provided for the first iteration.")
            reference_points[dm] = previous_responses[-1].reference_points[dm]
            improvement_directions[dm] = previous_responses[-1].improvement_directions[dm]
        else:
            # If a reference point is provided, calculate the improvement direction
            # First, check if the reference point is better than the navigation point
            max_multiplier = [-1 if obj.maximize else 1 for obj in problem.objectives]
            reference_point = (
                np.array([reference_points[dm][obj.symbol] for obj in problem.objectives]) * max_multiplier
            )
            nav_point_arr = np.array([nav_point[obj.symbol] for obj in problem.objectives]) * max_multiplier
            improvement = nav_point_arr - reference_point
            if np.any(improvement < 0):
                msg = (
                    f"If a reference point is provided, it must be better than the navigation point.\n"
                    f" The reference point for {dm} is not better than the navigation point.\n"
                    f" Reference point: {reference_point}, Navigation point: {nav_point}\n"
                    f"Check objectives {np.where(improvement < 0)}"
                )
                raise NautiliError(msg)
            # The improvement direction is in the true objective space
            improvement_directions[dm] = improvement * max_multiplier
    # can just aggregate mean and median here
    if pref_agg_method == "mean":
        #print(pref_agg_method)
        g_improvement_direction = np.mean(list(improvement_directions.values()), axis=0)
        group_reference_point = None # we dont have group RP.. #np.mean(list(reference_points.values()), axis=0) 
    if pref_agg_method == "maxmin" or pref_agg_method == "maxmin_cones":
        #print(pref_agg_method)
        nadir = get_nadir_dict(problem)
        ideal = get_ideal_dict(problem)
        g_reference_point = aggregate(pref_agg_method, reference_points, nav_point_arr,
            len(ideal),
            len(reference_points.keys()),
            ideal, nadir
            )
        group_reference_point = {
            obj.symbol: g_reference_point[i] for i, obj in enumerate(problem.objectives)
        }
        g_improvement_direction = nav_point_arr - g_reference_point

    group_improvement_direction = {
        obj.symbol: g_improvement_direction[i] for i, obj in enumerate(problem.objectives)
    }

    if group_improvement_direction is not None:
        opt_result = solve_reachable_solution(problem, group_improvement_direction, nav_point, create_solver)
        reachable_solution = opt_result.optimal_objectives

    # update nav point
    new_nav_point = calculate_navigation_point(problem, nav_point, reachable_solution, steps_remaining)
    # update_bounds
    lower_bounds, upper_bounds = solve_reachable_bounds(problem, new_nav_point, create_solver=create_solver)

    distance = calculate_distance_to_front(problem, new_nav_point, reachable_solution)

    return NAUTILI_Response(
        step_number=step_number,
        distance_to_front=distance,
        reference_points=None,
        improvement_directions=None,
        group_improvement_direction=group_improvement_direction,
        group_reference_point=group_reference_point,
        navigation_point=new_nav_point,
        reachable_solution=reachable_solution,
        reachable_bounds={"lower_bounds": lower_bounds, "upper_bounds": upper_bounds},
    )


def nautili_all_steps(
    problem: Problem,
    steps_remaining: int,
    reference_points: dict[str, dict[str, float]],
    previous_responses: list[NAUTILI_Response],
    pref_agg_method: str | None,
    create_solver: CreateSolverType | None = None,
):
    responses = []
    nav_point = previous_responses[-1].navigation_point
    step_number = previous_responses[-1].step_number + 1
    first_iteration = True
    reachable_solution = dict
    pref_agg_method: str | None

    # Calculate the improvement directions for each DM
    improvement_directions = {}
    for dm in reference_points:
        if reference_points[dm] is None:
            # If no reference point is provided, use the previous improvement direction
            if previous_responses[-1].reference_points is None:
                raise NautiliError("A reference point must be provided for the first iteration.")
            if previous_responses[-1].improvement_directions is None:
                raise NautiliError("An improvement direction must be provided for the first iteration.")
            reference_points[dm] = previous_responses[-1].reference_points[dm]
            improvement_directions[dm] = previous_responses[-1].improvement_directions[dm]
        else:
            # If a reference point is provided, calculate the improvement direction
            # First, check if the reference point is better than the navigation point
            max_multiplier = [-1 if obj.maximize else 1 for obj in problem.objectives]
            reference_point = (
                np.array([reference_points[dm][obj.symbol] for obj in problem.objectives]) * max_multiplier
            )
            nav_point_arr = np.array([nav_point[obj.symbol] for obj in problem.objectives]) * max_multiplier
            improvement = nav_point_arr - reference_point
            if np.any(improvement < 0):
                msg = (
                    f"If a reference point is provided, it must be better than the navigation point.\n"
                    f" The reference point for {dm} is not better than the navigation point.\n"
                    f" Reference point: {reference_point}, Navigation point: {nav_point}\n"
                    f"Check objectives {np.where(improvement < 0)}"
                )
                raise NautiliError(msg)
            # The improvement direction is in the true objective space
            improvement_directions[dm] = improvement * max_multiplier
    if pref_agg_method == "mean":
        print(pref_agg_method)
        g_improvement_direction = np.mean(list(improvement_directions.values()), axis=0)
        g_reference_point = nav_point_arr - g_improvement_direction
        group_reference_point = {
            obj.symbol: g_reference_point[i] for i, obj in enumerate(problem.objectives)
        } 
            # TODO: TEST this. It looks ok, and seems to work.
    if pref_agg_method == "eq_mean":
        for dm in improvement_directions:
            dict_impr = numpy_array_to_objective_dict(problem, improvement_directions[dm])
            opt_result = solve_reachable_solution(problem, dict_impr, nav_point)
            reachable_solution = opt_result.optimal_objectives
            improvement_directions[dm] = objective_dict_to_numpy_array(problem, reachable_solution)
        #print(improvement_directions)
        #print(pref_agg_method)
        g_reference_point = np.mean(list(improvement_directions.values()), axis=0)
        g_improvement_direction = nav_point_arr - g_reference_point
        #nav_point_arr - g_improvement_direction
        group_reference_point = {
            obj.symbol: g_reference_point[i] for i, obj in enumerate(problem.objectives)
        } 
    # TODO: TEST this. It looks ok, and seems to work.
    if pref_agg_method == "eq_maxmin" or pref_agg_method == "eq_maxmin_cones":
        print(pref_agg_method)
        nadir = get_nadir_dict(problem)
        ideal = get_ideal_dict(problem)
        # use improvmenet directions because we are solving the problem for each DM
        for dm in improvement_directions:
            dict_impr = numpy_array_to_objective_dict(problem, improvement_directions[dm])
            opt_result = solve_reachable_solution(problem, dict_impr, nav_point)
            reachable_solution = opt_result.optimal_objectives
            # reachable solution is set to improvement directions object as A RP for that DM.
            improvement_directions[dm] = reachable_solution #- nav_point
            #improvement_directions[dm] = objective_dict_to_numpy_array(problem, reachable_solution) - nav_point_arr
        # here they are now converted back to RPS
        reference_points = improvement_directions

        # solve maxmin normally, againg improvmenet directions contain the reference points
        g_reference_point = aggregate(pref_agg_method, reference_points, nav_point_arr,
            len(ideal),
            len(reference_points.keys()), # get number of DMs
            ideal, nadir
            )
        group_reference_point = {
            obj.symbol: g_reference_point[i] for i, obj in enumerate(problem.objectives)
        }
        g_improvement_direction = nav_point_arr - g_reference_point

        # TODO: TEST this. It looks ok, and seems to work.
    if pref_agg_method == "maxmin" or pref_agg_method == "maxmin_cones":
        print(pref_agg_method)
        nadir = get_nadir_dict(problem)
        ideal = get_ideal_dict(problem)
        g_reference_point = aggregate(pref_agg_method, reference_points, nav_point_arr,
            len(ideal),
            len(reference_points.keys()),
            ideal, nadir
            )
        group_reference_point = {
            obj.symbol: g_reference_point[i] for i, obj in enumerate(problem.objectives)
        }
        g_improvement_direction = nav_point_arr - g_reference_point

    group_improvement_direction = {
        obj.symbol: g_improvement_direction[i] for i, obj in enumerate(problem.objectives)
    }

    while steps_remaining > 0:
        if first_iteration:
            response = nautili_step(
                problem,
                steps_remaining=steps_remaining,
                step_number=step_number,
                nav_point=nav_point,
                group_improvement_direction=group_improvement_direction,
                group_reference_point=group_reference_point,
                create_solver=create_solver,
            )
            first_iteration = False
        else:
            response = nautili_step(
                problem,
                steps_remaining=steps_remaining,
                step_number=step_number,
                nav_point=nav_point,
                reachable_solution=reachable_solution,
                create_solver=create_solver,
            )
        response.reference_points = reference_points
        response.improvement_directions = improvement_directions
        response.group_improvement_direction = group_improvement_direction
        response.group_reference_point = group_reference_point
        responses.append(response)
        reachable_solution = response.reachable_solution
        nav_point = response.navigation_point
        steps_remaining -= 1
        step_number += 1
    return responses

"""
BELOW related to pref agg tests
"""
def aggregate(pref_agg_method: str,
    reference_points: dict[str, dict[str, float]],
    nav_point_arr = list[float],
    k = int,
    q = int,
    ideal = list[float],
    nadir = list[float]):

    """Aggregates maxmin.

    Args:
        pref_agg_method: the string depicting what preference aggregation method to use

    Returns:
        group improvemnet direction
    """

    # TODO: HERE is aggre. TODO: make smarter. NOTE the different logic on the lists or dictionaries. Make own function.

    print(f"number of Objs:{k}, number of DMs:{q}")
    group_reference_point = None
    rp_a = {}
    for key, p in reference_points.items():
        rp_a[key] = list(p.values())
    rp_arr = np.array(list(rp_a.values()))
    ideal = np.array(list(ideal.values()))
    nadir = np.array(list(nadir.values())) 

    if pref_agg_method == "maxmin" or pref_agg_method == "eq_maxmin":
        # TODO: get from the problem?
        #rp_arr = np.array([pref[1] for pref in reference_points.items()])
        group_pref = agg_maxmin(rp_arr, nav_point_arr, k, q, ideal, nadir)
        print("group RP", group_pref) # nav_point_arr - group_pref
        group_reference_point = group_pref

    if pref_agg_method == "maxmin_cones" or pref_agg_method == "eq_maxmin_cones":
        # TODO:
        #rp_arr = np.array([pref[1] for pref in reference_points.items()])
        group_pref = agg_maxmin_cones(rp_arr, nav_point_arr, k, q, ideal, nadir)
        print(group_pref) # nav_point_arr - group_pref
        group_reference_point = group_pref

    return group_reference_point

# TODO: 
# - FIX solver; it brings issues not being stable.
#   - trust-constr(paras?) ja slsqp (tosi sama kuin mean aina). TRust const välillä feilaa, 
#       mutta selkeästi slsqp lopettaa haun aina todella lähelle vaan aloituspistettä eli mean(objs)
# - FIX bounds for subprob opt. variables.. 
def agg_maxmin(agg, cip, k, q, ideal, nadir):
    agg_pref = []     
    # X = [R1, R2, R3, W1, W2, W3, W4, ALPHA]
    bnds = []
    # how many rows for X, n of DMs, n of objs + 1 for alpha
    #ra = k+q+1
    alpha = k+q

    # bounds for objectives and DMs
    # TODO: Need to make smarter for bounds to consider if obj is maximized. now assumes minimization. Also prob wrong place for the bounds.
    #i = 0
    #for _ in range(k+q+1):
        #b = (0,1)
    """
        if i < k:
            # dumb way of converting max to min for scipy
            if (ideal[i] > nadir[i]):
                 b = (-1*ideal[i], -1*nadir[i])
            else:
                b = (ideal[i], nadir[i])
            i += 1
        else:
            i = 0
            b = (ideal[i], nadir[i])
            i += 1
        print("boiunds",b)
    """
        #bnds.append(b)
    
    for _ in range(k):
        #bnds.append((-np.inf,np.inf))
        bnds.append((0,1)) # only for dtlz2

    for _ in range(k, k+q):
        bnds.append((0,1))
    bnds.append((0,10000))

    # bnds = [(0,1), (0,1), (0,1), (0,1)]
    # create X of first iteration guess
    X = np.zeros(k+q+1) # last number (alpha, the param to maximize) can stay as zero change the rest
    # Fill first guess for RP taking max from the DMs
    # TODO: weight bounds must be allways positve!
    for i in range(k):
        X[i] = np.mean(agg[:,i]) #TODO remember this change

    # Calculate the weights for DMs
    for qk in range(q):
        X[k+qk] = 1/q

    X[alpha] = 1
    # constraints for feasible space
    def feas_space_const(X, k, q, i, agg): #X!!
        return lambda X: sum([X[k+j]*agg[j,i] for j in range(q)]) - X[i]

    # p = r1 DM's suggested point,
    # c = r0, current iteration point
    # r = R, suggested group point. 
    def maxmin_crit(p, c, r):
        return np.sum((p - c)*r)

    # constraints for DMs S
    def DMconstr(X, q, k, agg, cip):
        # alpha - S_4(R) <= 0
        # R x[:alpha]*agg[j, :] for j in range(q) 
        # equ 4
        #print("val", X[j], sum((agg[j,i] - cip[i])*X[i] for i in range(k)))
        print("vals",[maxmin_crit(agg[j,:], cip, X[:k]) for j in range(q)] )
        # i am taking the wrong min, I need to take the min of the UTItILIES of DMS not fro each point individually
        
        # anna pelkät vektorit !!!
        return lambda X: X[alpha] - np.min([maxmin_crit(agg[j,:], cip, X[:k]) for j in range(q)] )

        #return lambda X: X[alpha] - np.min([(agg[j,i] - cip[i])*X[i] for i in range(k) for j in range(q)]) 
        
        #return lambda X: X[alpha] - sum((agg[j,i] - cip[i])*X[i] for i in range(k)) # TODO: this is the only different thing, so should be able to just combine the functions of maxmin and maxmmin cones
        #sum(X[i]*agg[j, :] for i in range(k) ) # Make to do exactly what equ4 says, mka new func as constra. that can take this function,
        #return lambda X: X[alpha] - sum(X[i]*agg[j, :] for i in range(nDMs))
        #return lambda X: X[alpha] - (sum([X[j]*agg[j,i] for j in range(q)]) - X[i]) #- sum((agg[j,i] - cip[i])*X[i] for i in range(k))

    # Create constraints
    Cons = []
    for i in range(k):
        Cons.append({'type':'eq', 'fun' : feas_space_const(X, k, q, i, agg)})

    # inequality mean in scipy that has to be NONNEGATIVE
    Cons.append({'type':'ineq', 'fun' : DMconstr(X, q, k, agg, cip)})
    #for j in range(q):
        #Cons.append({'type':'ineq', 'fun' : DMconstr(X, j, k, agg, cip)})

    # Convex constraint
    def constraint5(X):
        # sum_{r=1}^4(lambda_r) - 1 = 0
        return sum(X[k:k+q]) - 1

    con5 ={'type':'eq', 'fun':constraint5}
    Cons.append(con5)

    # The s_m(R) for all DMs
    def fx(X):
        #return -1*X[alpha]  
        return -1*X[alpha]-np.min([maxmin_crit(agg[j,:], cip, X[:k]) for j in range(q)] )
        #return -1* sum((agg[j,i] - cip[i])*X[i] for i in range(k) for j in range(q))
        
    from scipy.optimize import minimize
    solution = minimize(fx,
                  X,
                  #method = 'SLSQP',
                  bounds = bnds,
                  constraints = Cons,
                  #options={'ftol': 1e-20, 'eps':1e-10,'maxiter': 10000, 'disp': True}
                  method = 'trust-constr',
                  #bounds = bnds,
                  #constraints = Cons,
                  #hess = lambda x: np.zeros_like(X),
                  options={'xtol': 1e-20, 'gtol': 1e-20, 'maxiter': 10000, 'disp': True}
                  )

    # Decision variables (solutions)
    print(solution.x)
    agg_pref.append(solution.x[0:k])

    return agg_pref[0]


#   
# TODO: 
# - FIX solver; it brings issues not being stable.
#   - trust-constr(paras?) ja slsqp (tosi sama kuin mean aina). TRust const välillä feilaa, 
#       mutta selkeästi slsqp lopettaa haun aina todella lähelle vaan aloituspistettä eli mean(objs)
# - FIX bounds for subprob opt. variables.. 
def agg_maxmin_cones(agg, cip, k, q, ideal, nadir):
    agg_pref = []     
    # X = [R1, R2, R3, W1, W2, W3, W4, ALPHA]
    bnds = []
    # how many rows for X, n of DMs, n of objs + 1 for alpha
    #ra = k+q+1
    alpha = k+q

    # bounds for objectives and DMs
    # TODO: fix
    """
    i=0
    for _ in range(k+1):
        if i < k:
            # dumb way of converting max to min for scipy
            if (ideal[i] > nadir[i]):
                 b = (-1*ideal[i], -1*nadir[i])
            else:
                b = (ideal[i], nadir[i])
            i += 1
        else:
            i = 0
            b = (ideal[i], nadir[i])
            i += 1
        print("boiunds",b)
        bnds.append(b)
    """
    for _ in range(k):
        #bnds.append((-np.inf,np.inf))
        bnds.append((0,1)) # only for dtlz2

    for _ in range(k, k+q):
        bnds.append((0,1))
    bnds.append((0,10000))
    #bnds.append((-np.inf,np.inf)) # aplha can be anyhing
    print(bnds)
    
    # create X of first iteration guess
    X = np.zeros(k+q+1) # last number (alpha, the param to maximize) can stay as zero change the rest
    X[alpha] = 1 # set alpha to 1 for some reason at begining
    # Fill first guess for RP taking mean from all the RPs
    for i in range(k):
        X[i] = np.mean(agg[:,i])

    # Set the starting weights for DMs
    for qk in range(q):
        X[k+qk] = 1/q

    # constraints for feasible space
    def feas_space_const(X, k, q, i):
        return lambda X: sum([X[k+j]*agg[j,i] for j in range(q)]) - X[i]

    # constraints for DMs S.. TODO: DO I need these anmyore CHECK
    def DMconstr(X, q, k, cip, agg):
        # w - S_1(R) <= 0
        # alpha - (-max(eval_RP))
        print("vals:", [eval_RP(cip, agg[j,:], X[:k]) for j in range(q)])
        return lambda X: X[alpha] + np.max([eval_RP(cip, agg[j,:], X[:k]) for j in range(q)])
        #return lambda X: X[alpha] + eval_RP(cip, agg[q,:], X[:k]) # same result whether we get the max of all DMs or individually.


    # Create constraints
    Cons = []
    for i in range(k):
        Cons.append({'type':'eq', 'fun' : feas_space_const(X,k, q, i)})
    
    Cons.append({'type':'ineq', 'fun' : DMconstr(X,q,k, cip, agg)})
    #for j in range(q):
    #    Cons.append({'type':'ineq', 'fun' : DMconstr(X,j,k, cip, agg)})
        #Cons.append({'type':'ineq', 'fun' : weight_pos(X,j,k)})
        #Cons.append({'type':'ineq', 'fun': eval_rp_constr(X,j,q)})

    # Convex constraint
    def constraint5(X):
        # sum_{r=1}^4(lambda_r) - 1 = 0
        return sum(X[k:k+q]) - 1

    con5 ={'type':'eq', 'fun':constraint5}
    Cons.append(con5)

    # The s_m(R) for all DMs
    def fx(X): # , cip, agg, k, q DO I need these?
        # if I change this sign to -, it works for case 1. But then in the second part of case 1, it seems not to work properly
        return -1*X[alpha] + np.max([eval_RP(cip, agg[j,:], X[:k]) for j in range(q)])

    print(X.shape)
    from scipy.optimize import minimize
    solution = minimize(fx,
                  X, 
                  method = 'trust-constr', # which is better, ofcourse they bring different solutions.
                  #method = 'SLSQP',
                  bounds = bnds,
                  constraints = Cons,
                  #hess = lambda x: np.zeros_like(X),
                  #options={'ftol': 1e-20, 'eps':1e-10,'maxiter': 10000, 'disp': True}
                  options={'xtol': 1e-20,'gtol': 1e-20, 'maxiter': 10000, 'disp': True}
                  )

    print(solution.x)
    # Decision variables (solutions)
    agg_pref.append(solution.x[0:k])

    return agg_pref[0]

# given a search direction from old CIP RO to new suggested point R1, evaluate point P using a cone model
def eval_RP(R0, R1, P, a=0.5):
    # calc dir vector.
    D = R1 - R0
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
    V1 = V/np.sqrt(np.sum(V**2))

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
    return eval_value2 #* (-1) # TODO: note this !! check if it workds but WHY?

# TODO: update to consider all objs and DMs.
def df_from_responses(all_resp, ndms):
    # TODO: make the f_1's come from the k
    dd = {"step_n":[], "Type":[], "f_1":[], "f_2":[], "f_3":[]}
    for resp in all_resp:
        first_iter_gid = all_resp[0].group_improvement_direction
        first_iter_grp = all_resp[0].group_reference_point
        for q in range(1,ndms+1): # DM määrä on q
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"DM{q}")
            dd["f_1"].append(resp.reference_points[f"DM{q}"]["f_1"])
            dd["f_2"].append(resp.reference_points[f"DM{q}"]["f_2"])
            dd["f_3"].append(resp.reference_points[f"DM{q}"]["f_3"])
        # add navigation point
        dd["step_n"].append(resp.step_number)
        dd["Type"].append(f"NAVPOINT")
        dd["f_1"].append(resp.navigation_point["f_1"])
        dd["f_2"].append(resp.navigation_point["f_2"])
        dd["f_3"].append(resp.navigation_point["f_3"])
        # add reachable solution
        dd["step_n"].append(resp.step_number)
        dd["Type"].append(f"REACHSOL")
        dd["f_1"].append(resp.reachable_solution["f_1"])
        dd["f_2"].append(resp.reachable_solution["f_2"])
        dd["f_3"].append(resp.reachable_solution["f_3"])
        # add group improvement direction
        if resp.group_improvement_direction == None:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GID")
            dd["f_1"].append(first_iter_gid["f_1"])
            dd["f_2"].append(first_iter_gid["f_2"])
            dd["f_3"].append(first_iter_gid["f_3"])
        else:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GID")
            dd["f_1"].append(resp.group_improvement_direction["f_1"])
            dd["f_2"].append(resp.group_improvement_direction["f_2"])
            dd["f_3"].append(resp.group_improvement_direction["f_3"])
        if resp.group_reference_point == None:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GRP")
            dd["f_1"].append(first_iter_grp["f_1"])
            dd["f_2"].append(first_iter_grp["f_2"])
            dd["f_3"].append(first_iter_grp["f_3"])
        else:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GRP")
            dd["f_1"].append(resp.group_reference_point["f_1"])
            dd["f_2"].append(resp.group_reference_point["f_2"])
            dd["f_3"].append(resp.group_reference_point["f_3"])

    ttt = pd.DataFrame(dd)
    return ttt


def df_from_responses2d(all_resp, ndms):
    # TODO: make the f_1's come from the k
    dd = {"step_n":[], "Type":[], "f_1":[], "f_2":[]}
    for resp in all_resp:
        first_iter_gid = all_resp[0].group_improvement_direction
        first_iter_grp = all_resp[0].group_reference_point
        for q in range(1,ndms+1): # DM määrä on q
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"DM{q}")
            dd["f_1"].append(resp.reference_points[f"DM{q}"]["f_1"])
            dd["f_2"].append(resp.reference_points[f"DM{q}"]["f_2"])
            #dd["f_3"].append(resp.reference_points[f"DM{q}"]["f_3"])
        # add navigation point
        dd["step_n"].append(resp.step_number)
        dd["Type"].append(f"NAVPOINT")
        dd["f_1"].append(resp.navigation_point["f_1"])
        dd["f_2"].append(resp.navigation_point["f_2"])
        #dd["f_3"].append(resp.navigation_point["f_3"])
        # add reachable solution
        dd["step_n"].append(resp.step_number)
        dd["Type"].append(f"REACHSOL")
        dd["f_1"].append(resp.reachable_solution["f_1"])
        dd["f_2"].append(resp.reachable_solution["f_2"])
        #dd["f_3"].append(resp.reachable_solution["f_3"])
        # add group improvement direction
        if resp.group_improvement_direction == None:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GID")
            dd["f_1"].append(first_iter_gid["f_1"])
            dd["f_2"].append(first_iter_gid["f_2"])
            #dd["f_3"].append(first_iter_gid["f_3"])
        else:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GID")
            dd["f_1"].append(resp.group_improvement_direction["f_1"])
            dd["f_2"].append(resp.group_improvement_direction["f_2"])
            #dd["f_3"].append(resp.group_improvement_direction["f_3"])
        if resp.group_reference_point == None:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GRP")
            dd["f_1"].append(first_iter_grp["f_1"])
            dd["f_2"].append(first_iter_grp["f_2"])
            #dd["f_3"].append(first_iter_grp["f_3"])
        else:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GRP")
            dd["f_1"].append(resp.group_reference_point["f_1"])
            dd["f_2"].append(resp.group_reference_point["f_2"])
            #dd["f_3"].append(resp.group_reference_point["f_3"])

    ttt = pd.DataFrame(dd)
    return ttt

def visualize_2d(filename, all_resp, ndms):
    import plotly.express as ex

    print(all_resp)
    df = df_from_responses2d(all_resp, ndms)
    #fig3.write_html(f"results/{filename}.html")
    df.to_csv(f"{filename}.csv")
    #ideal=np.array([-20, -12]), nadir=np.array([-14, 0.5]
    visu_ideal = [0, 0]
    visu_nadir = [140,50]

    # näin helppoa se on kun sen vaan saa oikein !!!
    fig2d = ex.scatter(df.sort_values("step_n"),
    x="f_1", y="f_2", #z="f_3",
    color="Type",
    #range_x=[visu_ideal[0] - 0.5, visu_nadir[0] + 0.5],
    #range_y=[visu_ideal[1] - 0.5, visu_nadir[1] + 0.5],
    #range_z=[visu_ideal[2] - 0.5, visu_nadir[2] + 0.5],
    range_x=[visu_ideal[0] - 0, visu_nadir[0]],
    range_y=[visu_ideal[1] - 0, visu_nadir[1]],
    #range_z=[visu_ideal[2] - 0, visu_nadir[2]],
    width= 1000,
    height= 1000,
    animation_frame=f"step_n",
    )

    fig2d.show()
    fig2d.write_html(f"results/{filename}.html")


# use only with 3 objs, TODO: get nadir and ideal from the problem.
def visualize_3d(filename, all_resp, ndms):
    import plotly.express as ex

    df = df_from_responses(all_resp, ndms)
    # TODO: change this when changing the task
    #df.to_csv(f"results/dtlz2/{filename}.csv")
    visu_ideal = [0,0,0]
    visu_nadir = [1,1,1]

    # näin helppoa se on kun sen vaan saa oikein !!!
    fig3 = ex.scatter_3d(df.sort_values("step_n"),
    x="f_1", y="f_2", z="f_3",
    color="Type",
    #range_x=[visu_ideal[0] - 0.5, visu_nadir[0] + 0.5],
    #range_y=[visu_ideal[1] - 0.5, visu_nadir[1] + 0.5],
    #range_z=[visu_ideal[2] - 0.5, visu_nadir[2] + 0.5],
    range_x=[visu_ideal[0] - 0, visu_nadir[0]],
    range_y=[visu_ideal[1] - 0, visu_nadir[1]],
    range_z=[visu_ideal[2] - 0, visu_nadir[2]],
    width= 1000,
    height= 1000,
    animation_frame=f"step_n",
    )
    fig3.show()
    #fig3.write_html(f"results/dtlz2/{filename}.html")

def test_binhkorn():
    problem = binh_and_korn()
    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)
    print(nadir)
    print(ideal)

    total_steps = 5
    #steps_remaining = 1
    ## take a step back
    pref_agg_method="maxmin"
    DMs = 5

    initial_response = nautili_init(problem)
    rps= {
    "DM1": {"f_1": 100, "f_2": 30 }, # dm1 and dm2 prefer f1
    "DM2": {"f_1": 100, "f_2": 30 },
    "DM3": {"f_1": 100, "f_2": 30 },
    "DM4": {"f_1": 30, "f_2": 50, },
    "DM5": {"f_1": 30, "f_2": 50 },
    }
    all_resp = nautili_all_steps(
        problem,
        total_steps,
        rps,
        [initial_response], # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )
    #print(all_resp)
    print(all_resp[-1].reference_points)
    print(all_resp[-1].group_improvement_direction)
    print(all_resp[-1].navigation_point)
    print("reachable solution:", all_resp[-1].reachable_solution)

    visualize_2d("binhtestmaxmin_cones", all_resp, DMs)

# Testaa dtlz2 step 3 mennen takaisin. Täsäs on uusin "versio" jolla voi ajaa useamman prefs keissin.
def test_dltz2_stepat3(prefs1, prefs2, pref_agg_methods, M, case):
    problem = dtlz2(10,3)
    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)
    total_steps = 5
    testnames= [f"{M}_DTLZ2_{case}_{pa}" for pa in pref_agg_methods]

    for idx, pref_agg_method in enumerate(pref_agg_methods):
        initial_response = nautili_init(problem)

        all_resp = nautili_all_steps(
            problem,
            total_steps,
            prefs1,
            [initial_response], 
            pref_agg_method=pref_agg_method, 
        )
        # to go to back to step 3
        stepback_resp = all_resp[:3]

        all_resp2 = nautili_all_steps(
            problem,
            2,
            prefs2,
            stepback_resp, 
            pref_agg_method=pref_agg_method, 
        )
        # append to stepback_resp
        for item in all_resp2:
            #print(item)
            stepback_resp.append(item)

        # visualize and save results
        visualize_3d(testnames[idx], stepback_resp, DMs)
        s = objective_dict_to_numpy_array(problem, stepback_resp[-1].reachable_solution)
        print(check_PO(s))

## works now
def test_binhkorn_stepbystep():
    problem = binh_and_korn()
    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)
    total_steps = 5
    #steps_remaining = 1
    ## take a step back
    pref_agg_method="maxmin_cones"
    testname="3C4BKC"
    DMs = 3

    initial_response = nautili_init(problem)
    rps= {
    "DM1": {"f_1": 1, "f_2": 3 },
    "DM2": {"f_1": 5, "f_2": 1 },
    "DM3": {"f_1": 50, "f_2": 30},
    #"DM4": {"f_1": 10, "f_2": 25 }, # dm1 and dm2 prefer f1
    #"DM5": {"f_1": 50, "f_2": 3},
    #"DM4": {"f_1": 0.3, "f_2": 0.3, },
    #"DM5": {"f_1": 23, "f_2": 35, },
    }
    all_resp = nautili_all_steps(
        problem,
        total_steps,
        rps,
        [initial_response], # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )

    ## DMs change prefs at step 3
    rps2= {
    "DM1": {"f_1": 1, "f_2": 3 },
    "DM2": {"f_1": 5, "f_2": 1 },
    "DM3": {"f_1": 30, "f_2": 20},

    #"DM4": {"f_1": 10, "f_2": 25 }, # dm1 and dm2 prefer f1
    #"DM5": {"f_1": 50, "f_2": 13},
    #"DM4": {"f_1": 3, "f_2": 3, },
    #"DM5": {"f_1": 23, "f_2": 25, },
    }
    stepback_resp = all_resp[:3]
    all_resp2 = nautili_all_steps(
        problem,
        2,
        rps2,
        stepback_resp, # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )
    for item in all_resp2:
        print(item)
        stepback_resp.append(item)

    """
    stepback_resp2 = stepback_resp[:4]
    rps3= {
    "DM1": {"f_1": 2, "f_2": 18 }, 
    "DM2": {"f_1": 5, "f_2": 15 },
    "DM3": {"f_1": 20, "f_2": 20},
    }
    all_resp3 = nautili_all_steps(
        problem,
        1,
        rps3,
        stepback_resp2, # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )
    for item in all_resp3:
        print(item)
        stepback_resp2.append(item)
    """

    visualize_2d(testname, stepback_resp, DMs)

def test_dltz2():
    problem = dtlz2(10,3)
    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)
    total_steps = 5
    pref_agg_method="maxmin"
    testname="3C1BKmean"
    DMs = 4

    initial_response = nautili_init(problem)
    rps= {
    "DM1": {"f_1": 0.8, "f_2": 0.4, "f_3": 0.7}, 
    "DM2": {"f_1": 0.62, "f_2": 0.62, "f_3": 0.8},
    "DM3": {"f_1": 0.5, "f_2": 0.37, "f_3": 0.53},
    "DM4": {"f_1": 0.3, "f_2": 0.3, "f_3": 0.15},
    #"DM5": {"f_1": 0.2, "f_2": 0.5, "f_3": 0.25},
    }
    all_resp = nautili_all_steps(
        problem,
        total_steps,
        rps,
        [initial_response], # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )
    #print(all_resp)
    rps2= {
    "DM1": {"f_1": 0.8, "f_2": 0.4, "f_3": 0.7}, 
    "DM2": {"f_1": 0.62, "f_2": 0.62, "f_3": 0.8},
    "DM3": {"f_1": 0.5, "f_2": 0.37, "f_3": 0.53},
    "DM4": {"f_1": 0.3, "f_2": 0.3, "f_3": 0.15},
    #"DM4": {"f_1": 3, "f_2": 3, },
    #"DM5": {"f_1": 23, "f_2": 25, },
    }

    stepback_resp = all_resp[:3]

    #print("HEllo", stepback_resp[-1].step_number)
    all_resp2 = nautili_all_steps(
        problem,
        2,
        rps2,
        stepback_resp, # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )
    #print(all_resp)
    #print(all_resp2[-1].reference_points)
    #print(all_resp2[-1].group_improvement_direction)
    #print(all_resp2[-1].navigation_point)
    #print("reachable solution:", all_resp2[-1].reachable_solution)

    for item in all_resp2:
        print(item)
        stepback_resp.append(item)

    visualize_3d(testname, all_resp, DMs)
    

def test_kuro():
    problem = kurosawe()
    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)
    total_steps = 5
    pref_agg_method="maxmin"
    DMs = 3

    initial_response = nautili_init(problem)
    rps= {
    "DM1": {"f_1": -18, "f_2": -7}, # dm1 and dm2 prefer f1
    "DM2": {"f_1": -16, "f_2": -2},
    "DM3": {"f_1": -15, "f_2": -3.7 },
    #"DM4": {"f_1": 0.3, "f_2": 0.3, "f_3": 0.15},
    #"DM5": {"f_1": 0.2, "f_2": 0.5, "f_3": 0.25},
    }
    all_resp = nautili_all_steps(
        problem,
        total_steps,
        rps,
        [initial_response], # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )
    #print(all_resp)
    print(all_resp[-1].reference_points)
    print(all_resp[-1].group_improvement_direction)
    print(all_resp[-1].navigation_point)
    print("reachable solution:", all_resp[-1].reachable_solution)

    visualize_2d("dtlz4test", all_resp)

def check_PO(sol):
    return (sol[0]**2 + sol[1]**2 + sol[2]**2)**(0.5) 

if __name__=="__main__":
    #test_binhkorn_stepbystep()
    #test_binhkorn()

    p1 = {
    "DM1": {"f_1": 0.1, "f_2": 0.1, "f_3": 0.1}, 
    "DM2": {"f_1": 0.6, "f_2": 0.6, "f_3": 0.6},
    "DM3": {"f_1": 0.6, "f_2": 0.6, "f_3": 0.6},
    "DM4": {"f_1": 0.6, "f_2": 0.7, "f_3": 0.7},
    }
    p2 = {
    "DM1": {"f_1": 0.5, "f_2": 0.6, "f_3": 0.7}, 
    "DM2": {"f_1": 0.15, "f_2": 0.7, "f_3": 0.6},
    "DM3": {"f_1": 0.6, "f_2": 0.7, "f_3": 0.7},
    "DM4": {"f_1": 0.15, "f_2": 0.65, "f_3": 0.65},
    }

    
    #pref_agg_methods=["mean", "maxmin","maxmin_cones", "eq_mean", "eq_maxmin", "eq_maxmin_cones"]
    pref_agg_methods=[ "maxmin","maxmin_cones"]
    DMs = 4
    test_dltz2_stepat3(p1, p2, pref_agg_methods, DMs, "c1test")
    """
    
    problem = dtlz2(10,3)
    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)
    total_steps = 3
    pref_agg_method="maxmin"
    DMs = 3

    initial_response = nautili_init(problem)
    rps= {
    "DM1": {"f_1": 0.8, "f_2": 0.84, "f_3": 0.7}, # dm1 and dm2 prefer f1
    "DM2": {"f_1": 0.62, "f_2": 0.72, "f_3": 0.8},
    "DM3": {"f_1": 0.85, "f_2": 0.7, "f_3": 0.53},
    #"DM4": {"f_1": 0.3, "f_2": 0.3, "f_3": 0.15},
    #"DM5": {"f_1": 0.48, "f_2": 0.1584, "f_3": 0.37}, # dm1 and dm2 prefer f1
    #"DM6": {"f_1": 0.62, "f_2": 0.5362, "f_3": 0.38},
    #"DM7": {"f_1": 0.25, "f_2": 0.2937, "f_3": 0.53},
    #"DM8": {"f_1": 0.3, "f_2": 0.13, "f_3": 0.15},
    #"DM5": {"f_1": 0.2, "f_2": 0.5, "f_3": 0.25},
    }
    all_resp = nautili_all_steps(
        problem,
        total_steps,
        rps,
        [initial_response], # Note that this is a list of NAUTILUS_Response objects
        pref_agg_method=pref_agg_method, # used pref agg method
    )
    #print(all_resp)
    print(all_resp[-1].reference_points)
    print(all_resp[-1].group_improvement_direction)
    print(all_resp[-1].navigation_point)
    print("reachable solution:", all_resp[-1].reachable_solution)

    s = objective_dict_to_numpy_array(problem, all_resp[-1].reachable_solution)
    print(check_PO(s))
 

    visualize_3d("dtlz2test", all_resp, DMs)
    """