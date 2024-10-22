"""Methods for the NAUTILI (a group decision making variant for NAUTILUS) method."""

import numpy as np
from pydantic import BaseModel, Field

import pandas as pd

from preference_aggregation import aggregate

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
#    zdt1,
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

from desdeo.tools.generics import BaseSolver, SolverResults
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

# Let's say X_train is your input dataframe
from sklearn.preprocessing import MinMaxScaler, Normalizer

# TODO: remove this 
#import warnings
#warnings.filterwarnings("ignore")


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
    problem: Problem, navigation_point: dict[str, float], solver: BaseSolver | None = None
) -> tuple[dict[str, float], dict[str, float]]:
    """Computes the current reachable (upper and lower) bounds of the solutions in the objective space.

    The reachable bound are computed based on the current navigation point. The bounds are computed by
    solving an epsilon constraint problem.

    Args:
        problem (Problem): the problem being solved.
        navigation_point (dict[str, float]): the navigation point limiting the
            reachable area. The key is the objective function's symbol and the value
            the navigation point.
        solver (BaseSolver | None, optional): a function of type BaseSolver that returns a solver.
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
    solver_init = guess_best_solver(problem) if solver is None else solver
    #opts = NevergradGenericOptions(optimizer="CMA")
    #solver_init = NevergradGenericSolver(problem= problem, options=opts)
    #solver_init = NevergradGenericSolver

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
    solver: BaseSolver | None = None,
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
        solver (BaseSolver | None, optional): a function of type BaseSolver that returns a solver.
            If None, then a solver is utilized bases on the problem's properties. Defaults to None.

    Returns:
        SolverResults: the results of the projection.
    """
    # check solver
    init_solver = guess_best_solver(problem) if solver is None else solver

    # create and add scalarization function
    # previous_nav_point = objective_dict_to_numpy_array(problem, previous_nav_point).tolist()
    # weights = objective_dict_to_numpy_array(problem, group_improvement_direction).tolist()
    
    # or nondiff asf depending on the problem
    problem_w_asf, target = add_asf_generic_diff( # use nondiff as default like in nautili
        problem,
        symbol="asf",
        reference_point=previous_nav_point,
        weights=group_improvement_direction,
        reference_point_aug=previous_nav_point,
    )

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


def nautili_init(problem: Problem, solver: BaseSolver | None = None) -> NAUTILI_Response:
    """Initializes the NAUTILI method.

    Creates the initial response of the method, which sets the navigation point to the nadir point
    and the reachable bounds to the ideal and nadir points.

    Args:
        problem (Problem): The problem to be solved.
        solver (BaseSolver | None, optional): The solver to use. Defaults to ???.

    Returns:
        NAUTILUS_Response: The initial response of the method.
    """
    nav_point = get_nadir_dict(problem)
    lower_bounds, upper_bounds = solve_reachable_bounds(problem, nav_point, solver=solver)
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
    solver: BaseSolver | None = None,
    group_improvement_direction: dict | None = None,
    group_reference_point: dict | None = None,
    reachable_solution: dict | None = None,
) -> NAUTILI_Response:
    if group_improvement_direction is None and reachable_solution is None:
        raise NautiliError("Either group_improvement_direction or reachable_solution must be provided.")

    if group_improvement_direction is not None and reachable_solution is not None:
        raise NautiliError("Only one of group_improvement_direction or reachable_solution should be provided.")

    if group_improvement_direction is not None:
        opt_result = solve_reachable_solution(problem, group_improvement_direction, nav_point, solver)
        reachable_solution = opt_result.optimal_objectives

    # update nav point
    new_nav_point = calculate_navigation_point(problem, nav_point, reachable_solution, steps_remaining)

    # update_bounds

    lower_bounds, upper_bounds = solve_reachable_bounds(problem, new_nav_point, solver=solver)

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

# NOT CURRENTLY USED !
# TODO: changed this for the experiments. Will aggregate here.
# Always computes the direction, only to be used when testing this.
def nautili_step2(  # NOQA: PLR0913
    problem: Problem,
    steps_remaining: int,
    step_number: int,
    nav_point: dict,
    reference_points: dict[str, dict[str, float]],
    solver: BaseSolver | None = None,
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
        g_reference_point = aggregate(
            problem,
            pref_agg_method, reference_points, nav_point_arr,
            len(ideal),
            len(reference_points.keys()),
            )
        group_reference_point = {
            obj.symbol: g_reference_point[i] for i, obj in enumerate(problem.objectives)
        }
        g_improvement_direction = nav_point_arr - g_reference_point

    group_improvement_direction = {
        obj.symbol: g_improvement_direction[i] for i, obj in enumerate(problem.objectives)
    }

    if group_improvement_direction is not None:
        opt_result = solve_reachable_solution(problem, group_improvement_direction, nav_point, solver)
        reachable_solution = opt_result.optimal_objectives

    # update nav point
    new_nav_point = calculate_navigation_point(problem, nav_point, reachable_solution, steps_remaining)
    # update_bounds
    lower_bounds, upper_bounds = solve_reachable_bounds(problem, new_nav_point, solver=solver)

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

# TODO:
# reference_points dict is not multiplied by max_multiplier. Only individual RPs and then used as improvement directions.
def nautili_all_steps(
    problem: Problem,
    steps_remaining: int,
    reference_points: dict[str, dict[str, float]],
    previous_responses: list[NAUTILI_Response],
    pref_agg_method: str | None,
    solver: BaseSolver | None = None,
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
    # MEAN STUFF
    if pref_agg_method == "mean":
        print(pref_agg_method)
        g_improvement_direction = np.mean(list(improvement_directions.values()), axis=0)
        g_reference_point = nav_point_arr - g_improvement_direction
        group_reference_point = {
            obj.symbol: g_reference_point[i] for i, obj in enumerate(problem.objectives)
        } 
    if pref_agg_method == "eq_mean":
        # Find the PO solution for each improvement direction from each DM.
        for dm in improvement_directions:
            dict_impr = numpy_array_to_objective_dict(problem, improvement_directions[dm])
            opt_result = solve_reachable_solution(problem, dict_impr, nav_point)
            reachable_solution = opt_result.optimal_objectives
            # Set the PO solution as the improvement direction for that DM.
            improvement_directions[dm] = objective_dict_to_numpy_array(problem, reachable_solution)
        # now group reference point is mean of the PO solutions, which we later seek the nearest PO solution.
        g_reference_point = np.mean(list(improvement_directions.values()), axis=0)
        g_improvement_direction = nav_point_arr - g_reference_point
        group_reference_point = {
            obj.symbol: g_reference_point[i] for i, obj in enumerate(problem.objectives)
        } 

    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)
    # EQ MAXMIN STUFF
    if pref_agg_method == "eq_maxmin" or pref_agg_method == "eq_maxmin_cones":
        # use improvmenet directions because we are solving the problem for each DM
        for dm in improvement_directions:
            dict_impr = numpy_array_to_objective_dict(problem, improvement_directions[dm])
            opt_result = solve_reachable_solution(problem, dict_impr, nav_point)
            reachable_solution = opt_result.optimal_objectives
            # reachable solution is set to improvement directions object as A RP for that DM.
            improvement_directions[dm] = reachable_solution #- nav_point
        # here they are now converted back to RPS
        reference_points = improvement_directions

        # solve maxmin normally, againg improvmenet directions contain the reference points
        g_reference_point = aggregate(problem, pref_agg_method, reference_points, nav_point_arr)
        group_reference_point = {
            obj.symbol: g_reference_point[i] for i, obj in enumerate(problem.objectives)
        }
        g_improvement_direction = nav_point_arr - g_reference_point

    # MAXMIN STUFF
    if pref_agg_method == "maxmin" or pref_agg_method == "maxmin_cones":
        g_reference_point = aggregate(problem, pref_agg_method, reference_points, nav_point_arr)
        group_reference_point = {
            obj.symbol: g_reference_point[i] for i, obj in enumerate(problem.objectives)
        }
        g_improvement_direction = nav_point_arr - g_reference_point

    # Make the group improvement direction for taking a step
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
                solver=solver,
            )
            first_iteration = False
        else:
            response = nautili_step(
                problem,
                steps_remaining=steps_remaining,
                step_number=step_number,
                nav_point=nav_point,
                reachable_solution=reachable_solution,
                solver=solver,
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


# TODO: check, should work now with any number of ob
def df_from_responses(all_resp, ndms, nobjs):
    dd = {"step_n":[], "Type":[]}
    for k in range(1, nobjs + 1):
        dd[f"f_{k}"] = []

    for resp in all_resp:
        first_iter_gid = all_resp[0].group_improvement_direction
        first_iter_grp = all_resp[0].group_reference_point
        for q in range(1,ndms+1): # DM määrä on q
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"DM{q}")
            for k in range(1, nobjs + 1):
                dd[f"f_{k}"].append(resp.reference_points[f"DM{q}"][f"f_{k}"])
        dd["step_n"].append(resp.step_number)
        dd["Type"].append(f"NP")
        for k in range(1, nobjs + 1):
            dd[f"f_{k}"].append(resp.navigation_point[f"f_{k}"])
        dd["step_n"].append(resp.step_number)
        dd["Type"].append(f"PO")
        for k in range(1, nobjs + 1):
            dd[f"f_{k}"].append(resp.reachable_solution[f"f_{k}"])
        if resp.group_improvement_direction == None:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GID")
            for k in range(1, nobjs + 1):
                dd[f"f_{k}"].append(first_iter_gid[f"f_{k}"])
        else:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GID")
            for k in range(1, nobjs + 1):
                dd[f"f_{k}"].append(resp.group_improvement_direction[f"f_{k}"])
        if resp.group_reference_point == None:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GRP")
            for k in range(1, nobjs + 1):
                dd[f"f_{k}"].append(first_iter_grp[f"f_{k}"])
        else:
            dd["step_n"].append(resp.step_number)
            dd["Type"].append(f"GRP")
            for k in range(1, nobjs+ 1):
                dd[f"f_{k}"].append(resp.group_reference_point[f"f_{k}"])
    ttt = pd.DataFrame(dd)
    return ttt

def visualize_2d(filename, all_resp, ndms, problem):
    import plotly.express as ex
    import plotly.graph_objects as go

    folder = problem.name
    print(all_resp)

    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)
    nobjs = len(ideal.keys())
    print(nobjs)

    df = df_from_responses(all_resp, ndms, nobjs)
    #df.to_csv(f"../results/{folder}/{filename}.csv")
    nadir = objective_dict_to_numpy_array(problem, get_nadir_dict(problem))
    ideal = objective_dict_to_numpy_array(problem, get_ideal_dict(problem))
    
    # näin helppoa se on kun sen vaan saa oikein !!!
    fig2d = ex.scatter(df.sort_values("step_n"),
        x="f_1", y="f_2", #z="f_3",
        color="Type",
        text="Type",
        #mode="markers+text",
        #text="f_1","f_2",
        #texttemplate = "%{text}<br>(%{f_1:.2f}, %{f_2:.2f})",
        range_x=[ideal[0], nadir[0]],
        range_y=[ideal[1], nadir[1]],
        width= 1000,
        height= 1000,
        animation_frame=f"step_n",
        title=f"{filename}"
    )
    
    # Add problems pareto front
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot
    problem_pymoo = get_problem(problem.name)
    pf = problem_pymoo.pareto_front()
    # Convert NumPy array to a list of dictionaries
    keys = ["f_1", "f_2"]
    PF = [dict(zip(keys, row)) for row in pf]   
    # plot PF
    PF_xes = [point['f_1'] for point in PF]
    PF_yes = [point['f_2'] for point in PF]
    fig2d.add_trace(
        go.Scatter(x=PF_xes, y=PF_yes, mode='lines', line=dict(color="blue"), name='Pareto Front') # TODO: maybe should change lines to markers.. for zdt3 it looks bad as it fills the gaps.
    )

    fig2d.update_traces(marker=dict(size=30))
    fig2d.show()
    fig2d.write_html(f"../results/{folder}/{filename}.html")
    fig2d.write_image(f"../results/{folder}/{filename}.pdf")

# use only with 3 objs, TODO: get nadir and ideal from the problem.
def visualize_3d(filename, all_resp, ndms, problem):
    import plotly.express as ex
    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)
    nobjs = len(ideal.keys())
    print(nobjs)

    df = df_from_responses(all_resp, ndms, nobjs)
    # TODO: change this when changing the task
    df.to_csv(f"results/dtlz2/{filename}.csv")

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
        title=f"{filename}"
    )
    fig3.show()
    fig3.write_html(f"results/dtlz2/{filename}.html")

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
def test_zdts_stepat3(problem_n, prefs1, prefs2, pref_agg_methods, M, case):
    if problem_n == 1: 
        problem = zdt1(30)
    elif problem_n == 2:
        problem = zdt2(30)
    else:
        problem = zdt3(30)
    
    nadir = get_nadir_dict(problem)
    ideal = get_ideal_dict(problem)
    total_steps = 1
    testnames= [f"{M}_{problem.name}_{case}_{pa}" for pa in pref_agg_methods]

    for idx, pref_agg_method in enumerate(pref_agg_methods):
        initial_response = nautili_init(problem)

        all_resp = nautili_all_steps(
            problem,
            total_steps,
            prefs1,
            [initial_response],
            pref_agg_method=pref_agg_method,
        )
        stepback_resp = all_resp
        # to go to back to step 3
        """
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
        """
        
        # visualize and save results
        visualize_2d(testnames[idx], stepback_resp, DMs, problem)
        #s = objective_dict_to_numpy_array(problem, stepback_resp[-1].reachable_solution)
        #print(check_PO(s))


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
        stepback_resp = all_resp
        # to go to back to step 3
        """
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
        """
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

    for item in all_resp2:
        print(item)
        stepback_resp.append(item)

    visualize_3d(testname, all_resp, DMs)
    


def matplotlib2d(filename:str, all_resp:NAUTILI_Response, ndms:int, problem:Problem):
    import matplotlib.pyplot as plt
    from pymoo.problems import get_problem
    from pymoo.util.plotting import plot#not used?

    rp_arr = np.array([[col["f_1"], col["f_2"]] for col in rps])

    problem_pymoo = get_problem("zdt1")
    pf = problem_pymoo.pareto_front()

    # Convert NumPy array to a list of dictionaries
    PF = [dict(zip(keys, row)) for row in pf]   
    # plot PF
    fig = px.line(PF, x="f_1", y="f_2")

    # TODO: figure out the marker styles better
    # plot RPs
    for i in range(len(rp_arr)):
        fig.add_scatter(x=[rp_arr[i][0]], y=[rp_arr[i][1]], mode="markers", name=f"DM{i+1}_RP",showlegend=True)
    #fig.update_traces(marker=dict(size=15, symbol="x"))

    # PLOT GRP
    fig.add_scatter(x=[grp2[0]], y=[grp2[1]], mode="markers", name="GRP",showlegend=True) 
    #fig.update_traces(marker=dict(size=15, symbol="star"))
    fig.update_traces(marker=dict(size=15, symbol="x"))
    # PLOT results
    #fig.add_scatter(all_solutions,  x=all_solutions.f_1, y=all_solutions.f_2, mode="markers", name=all_solutions.names ,showlegend=True)
    for i in range(len(namelist)):
        fig.add_scatter(x=[all_solutions["f_1"][i]], y=[all_solutions["f_2"][i]], mode="markers", name=all_solutions.names[i] ,showlegend=True)
    #fig.add_scatter(all_solutions, x="f_1", y="f_2", mode="markers", name="ASF",showlegend=True)
    #fig.add_scatter(all_solutions, x="f_1", y="f_2", mode="markers", name="ASF",showlegend=True)
    #fig.add_traces(all_solutions)
    fig.update_traces(marker=dict(size=15))
    fig.show()  


def check_PO(sol):
    return (sol[0]**2 + sol[1]**2 + sol[2]**2)**(0.5)

if __name__=="__main__":
    #test_binhkorn_stepbystep()
    #test_binhkorn()
   
    # C1
    # ZDT1
    p1 = {
    "DM1":         {"f_1": 0.1, "f_2": 0.1}, 
    "DM2":         {"f_1": 0.45, "f_2": 0.5},
    "DM3":         {"f_1": 0.5, "f_2": 0.45},

    #"DM4": {"f_1": 0.9, "f_2": 0.7},
    }
    p2 = {
    "DM1": {"f_1": 0.46, "f_2": 0.6}, 
    "DM2": {"f_1": 0.3, "f_2": 0.6},
    "DM3": {"f_1": 0.3, "f_2": 0.4},
    }
    #pref_agg_methods=["maxmin_cones"]
    pref_agg_methods=["maxmin", "maxmin_cones"]
    problem_n = 1
    #pref_agg_methods=["mean", "maxmin","maxmin_cones", "eq_mean", "eq_maxmin", "eq_maxmin_cones"]
    DMs = 3
    test_zdts_stepat3(problem_n, p1, p2, pref_agg_methods, DMs, "c1")

    """
    # ZDT2

    problem_n = 2
    #pref_agg_methods=["maxmin_cones"]
    pref_agg_methods=["mean", "maxmin","maxmin_cones", "eq_mean", "eq_maxmin", "eq_maxmin_cones"]
    DMs = 3
    test_zdts_stepat3(problem_n, p1, p2, pref_agg_methods, DMs, "c1")

    # ZDT3 
    # TODO: IGNORE solving this until getting the locally optima problem solved !
 
    p1 = {
    "DM1": {"f_1": 0.4, "f_2": 0.6 }, 
    "DM2": {"f_1": 0.3, "f_2": 0.4 },
    "DM3": {"f_1": 0.2, "f_2": 0.8},

    }
    p2 = {
    "DM1": {"f_1": 0.36, "f_2": 0.6}, 
    "DM2": {"f_1": 0.3, "f_2": 0.4},
    "DM3": {"f_1": 0.2, "f_2": 0.6},
    }
    #pref_agg_methods=["maxmin", "maxmin_cones"]
    problem_n = 3
    pref_agg_methods=["mean", "maxmin","maxmin_cones", "eq_mean", "eq_maxmin", "eq_maxmin_cones"]
    DMs = 3
    test_zdts_stepat3(problem_n, p1, p2, pref_agg_methods, DMs, "c1")
    """ 


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