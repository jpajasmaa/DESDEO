"""Functions related to the GNIMBUS method.

References:
"""  # noqa: RUF002

import numpy as np

from desdeo.problem import (PolarsEvaluator, Problem, VariableType,
                            flatten_variable_dict, Constraint,
                            ConstraintTypeEnum, Variable, VariableTypeEnum, ScalarizationFunction,
                            objective_dict_to_numpy_array,
                            unflatten_variable_array,
                            )
from desdeo.tools.utils import (
    # get_corrected_ideal_and_nadir,
    get_corrected_ideal,
    get_corrected_nadir,
    get_corrected_reference_point,
)
from desdeo.tools.scalarization import objective_dict_has_all_symbols

from desdeo.tools import (
    BaseSolver,
    SolverOptions,
    SolverResults,
    add_group_asf_diff,
    add_group_asf,
    add_group_guess_sf_diff,
    add_group_guess_sf,
    add_group_nimbus_sf_diff,
    add_group_nimbus_sf,
    add_group_stom_sf_diff,
    add_group_stom_sf,
    guess_best_solver,
    add_asf_diff,
    ScalarizationError,
    add_nimbus_sf_diff, add_nimbus_sf_nondiff,
    add_asf_diff,
    add_asf_nondiff,
    add_guess_sf_diff,
    add_guess_sf_nondiff,
    add_nimbus_sf_diff,
    add_nimbus_sf_nondiff,
    add_stom_sf_diff,
    add_stom_sf_nondiff,
    guess_best_solver,
)
from desdeo.mcdm.nimbus import (
    # generate_starting_point,
    solve_intermediate_solutions,
    infer_classifications,
    NimbusError
)

import polars as pl

from aggregate_classifications import aggregate_classifications

def add_group_nimbus_sfv2(  # noqa: PLR0913
    problem: Problem,
    symbol: str,
    classifications_list: list[dict[str, tuple[str, float | None]]],
    current_objective_vector: dict[str, float],
    ideal: dict[str, float] | None = None,
    nadir: dict[str, float] | None = None,
    delta: float = 0.000001,
    rho: float = 0.000001,
) -> tuple[Problem, str]:
    r"""Implements the multiple decision maker variant of the NIMBUS scalarization function.

    The scalarization function is defined as follows:

    \begin{align}
        &\mbox{minimize} &&\max_{i\in I^<,j\in I^\leq,d} [w_{id}(f_{id}(\mathbf{x})-z^{ideal}_{id}),
        w_{jd}(f_{jd}(\mathbf{x})-\hat{z}_{jd})] +
        \rho \sum^k_{i=1} \sum^{n_d}_{d=1} w_{id}f_{id}(\mathbf{x}) \\
        &\mbox{subject to} &&\mathbf{x} \in \mathbf{X},
    \end{align}

    where $w_{id} = \frac{1}{z^{nad}_{id} - z^{uto}_{id}}$, and $w_{jd} = \frac{1}{z^{nad}_{jd} - z^{uto}_{jd}}$.

    The $I$-sets are related to the classifications given to each objective function value
    in respect to  the current objective vector (e.g., by a decision maker). They
    are as follows:

    - $I^{<}$: values that should improve,
    - $I^{\leq}$: values that should improve until a given aspiration level $\hat{z}_i$,
    - $I^{=}$: values that are fine as they are,
    - $I^{\geq}$: values that can be impaired until some reservation level $\varepsilon_i$, and
    - $I^{\diamond}$: values that are allowed to change freely (not present explicitly in this scalarization function).

    The aspiration levels and the reservation levels are supplied for each classification, when relevant, in
    the argument `classifications` as follows:

    ```python
    classifications = {
        "f_1": ("<", None),
        "f_2": ("<=", 42.1),
        "f_3": (">=", 22.2),
        "f_4": ("0", None)
        }
    ```

    Here, we have assumed four objective functions. The key of the dict is a function's symbol, and the tuple
    consists of a pair where the left element is the classification (self explanatory, '0' is for objective values
    that may change freely), the right element is either `None` or an aspiration or a reservation level
    depending on the classification.

    Args:
        problem (Problem): the problem to be scalarized.
        symbol (str): the symbol given to the scalarization function, i.e., target of the optimization.
        classifications_list (list[dict[str, tuple[str, float  |  None]]]): a list of dicts, where the key is a symbol
            of an objective function, and the value is a tuple with a classification and an aspiration
            or a reservation level, or `None`, depending on the classification. See above for an
            explanation.
        current_objective_vector (dict[str, float]): the current objective vector that corresponds to
            a Pareto optimal solution. The classifications are assumed to been given in respect to
            this vector.
        ideal (dict[str, float], optional): ideal point values. If not given, attempt will be made
            to calculate ideal point from problem.
        nadir (dict[str, float], optional): nadir point values. If not given, attempt will be made
            to calculate nadir point from problem.
        delta (float, optional): a small scalar used to define the utopian point. Defaults to 0.000001.
        rho (float, optional): a small scalar used in the augmentation term. Defaults to 0.000001.

    Raises:
        ScalarizationError: any of the given classifications do not define a classification
            for all the objective functions or any of the given classifications do not allow at
            least one objective function value to improve and one to worsen.

    Returns:
        tuple[Problem, str]: a tuple with the copy of the problem with the added
            scalarization and the symbol of the added scalarization.
    """
    # check that classifications have been provided for all objective functions
    for classifications in classifications_list:
        if not objective_dict_has_all_symbols(problem, classifications):
            msg = (
                f"The given classifications {classifications} do not define "
                "a classification for all the objective functions."
            )
            raise ScalarizationError(msg)

    # check if ideal point is specified
    # if not specified, try to calculate corrected ideal point
    if ideal is not None:
        ideal_point = ideal
    elif problem.get_ideal_point() is not None:
        ideal_point = get_corrected_ideal(problem)
    else:
        msg = "Ideal point not defined!"
        raise ScalarizationError(msg)

    # check if nadir point is specified
    # if not specified, try to calculate corrected nadir point
    if nadir is not None:
        nadir_point = nadir
    elif problem.get_nadir_point() is not None:
        nadir_point = get_corrected_nadir(problem)
    else:
        msg = "Nadir point not defined!"
        raise ScalarizationError(msg)

    corrected_current_point = get_corrected_reference_point(problem, current_objective_vector)

    # calculate the weights
    weights = {obj.symbol: 1 / (nadir_point[obj.symbol] - (ideal_point[obj.symbol] - delta)) for obj in problem.objectives}

    # max term and constraints
    max_args = []
    constraints = []

    for i in range(len(classifications_list)):
        classifications = classifications_list[i]
        for obj in problem.objectives:
            _symbol = obj.symbol
            match classifications[_symbol]:
                case ("<", _):
                    max_expr = f"{weights[_symbol]} * ({_symbol}_min - {ideal_point[_symbol]})"
                    max_args.append(max_expr)

                    con_expr = f"{_symbol}_min - {corrected_current_point[_symbol]}"
                    constraints.append(
                        Constraint(
                            name=f"improvement constraint for {_symbol}",
                            symbol=f"{_symbol}_{i+1}_lt",
                            func=con_expr,
                            cons_type=ConstraintTypeEnum.LTE,
                            is_linear=problem.is_linear,
                            is_convex=problem.is_convex,
                            is_twice_differentiable=problem.is_twice_differentiable,
                        )
                    )
                case ("<=", aspiration):
                    # if obj is to be maximized, then the current aspiration value needs to be multiplied by -1
                    max_expr = (
                        f"{weights[_symbol]} * ({_symbol}_min - {aspiration * -1 if obj.maximize else aspiration})"
                    )
                    max_args.append(max_expr)

                    con_expr = f"{_symbol}_min - {corrected_current_point[_symbol]}"
                    constraints.append(
                        Constraint(
                            name=f"improvement until constraint for {_symbol}",
                            symbol=f"{_symbol}_{i+1}_lte",
                            func=con_expr,
                            cons_type=ConstraintTypeEnum.LTE,
                            is_linear=problem.is_linear,
                            is_convex=problem.is_convex,
                            is_twice_differentiable=problem.is_twice_differentiable,
                        )
                    )
                case ("=", _):
                    con_expr = f"{_symbol}_min - {corrected_current_point[_symbol]}"
                    constraints.append(
                        Constraint(
                            name=f"Stay at least as good constraint for {_symbol}",
                            symbol=f"{_symbol}_{i+1}_eq",
                            func=con_expr,
                            cons_type=ConstraintTypeEnum.LTE,
                            is_linear=problem.is_linear,
                            is_convex=problem.is_convex,
                            is_twice_differentiable=problem.is_twice_differentiable,
                        )
                    )
                case (">=", reservation):
                    # if obj is to be maximized, then the current reservation value needs to be multiplied by -1
                    con_expr = f"{_symbol}_min - {-1 * reservation if obj.maximize else reservation}"
                    constraints.append(
                        Constraint(
                            name=f"Worsen until constraint for {_symbol}",
                            symbol=f"{_symbol}_{i+1}_gte",
                            func=con_expr,
                            cons_type=ConstraintTypeEnum.LTE,
                            is_linear=problem.is_linear,
                            is_convex=problem.is_convex,
                            is_twice_differentiable=problem.is_twice_differentiable,
                        )
                    )
                case ("0", _):
                    # not relevant for this scalarization
                    pass
                case (c, _):
                    msg = (
                        f"Warning! The classification {c} was supplied, but it is not supported."
                        "Must be one of ['<', '<=', '0', '=', '>=']"
                    )
    max_expr = f"Max({','.join(max_args)})"

    # form the augmentation term
    aug_exprs = []
    for _ in range(len(classifications_list)):
        aug_expr = " + ".join([f"({weights[obj.symbol]} * {obj.symbol}_min)" for obj in problem.objectives])
        aug_exprs.append(aug_expr)
    aug_exprs = " + ".join(aug_exprs)

    func = f"{max_expr} + {rho} * ({aug_exprs})"
    scalarization = ScalarizationFunction(
        name="NIMBUS scalarization objective function for multiple decision makers",
        symbol=symbol,
        func=func,
        is_linear=problem.is_linear,
        is_convex=problem.is_convex,
        is_twice_differentiable=False,
    )

    _problem = problem.add_scalarization(scalarization)
    return _problem.add_constraints(constraints), symbol


def explain(problem, classifications, results: [SolverResults]):
    """
    # pseudocode
    """
    print(f"Pareto optimal solutions found: {[results[i].optimal_objectives for i in range(len(results))]}")

    # TOO COMPLEX way to do simple thing
    key_flag_dict = {"f_1": 0, "f_2": 0}  # number of objs
    for i, dm in enumerate(classifications):
        print(dm)
        key_flag = 0  # t'mä nollaa ne josssai kohtaa
        for key in dm:
            # print(key, (value))
            print(dm[key])
            # take first the sign of the tuple
            if dm[key][0] == ">=":
                key_flag = 1
        key_flag_dict[key] = key_flag

    print(key_flag_dict)


def find_boundaries(reference_points, objective_keys, maxormin):
    """
        Find bounds from the reference points and jth objective function
    """
    import pandas as pd
    reference_points_list = dict_of_rps_to_list_of_rps(reference_points)
    df = pd.DataFrame(reference_points_list)

    bounds = []
    for o in objective_keys:
        if maxormin == "max":
            bounds.append(max(df[o]))
        if maxormin == "min":
            bounds.append(min(df[o]))

    return bounds

def majority_rule(votes: dict[str, int]):
    from collections import Counter

    counts = Counter(votes.values())
    all_votes = sum(counts.values())
    for vote, c in counts.items():
        if c > all_votes // 2:
            return vote
    return None

def plurality_rule(votes: dict[str, int]):
    from collections import Counter

    counts = Counter(votes.values())
    max_votes = max(counts.values())
    winners = [vote for vote, c in counts.items() if c == max_votes]

    return winners

def voting_procedure(problem: Problem, solutions, votes_idxs: dict[str, float]) -> SolverResults:
    winner_idx = None

    # call majority
    winner_idx = majority_rule(votes_idxs)
    if winner_idx is not None:
        print("Majority winner", winner_idx)
        return solutions[winner_idx]

    # call plurality
    winners = plurality_rule(votes_idxs)
    print("winners")
    # TODO: handle if there are three or more same number of votes
    if len(winners) == 1:
        print("Plurality winner", winners[0])
        return solutions[winners[0]]  # need to unlist the winners list
    if len(winners) == 2:
        # if two same solutions with same number of votes, call intermediate
        # TODO: change this to dec vars when they exist
        # wsol1, wsol2 = solutions[winners[0]].optimal_variables, solutions[winners[1]].optimal_variables
        wsol1, wsol2 = solutions[winners[0]].optimal_objectives, solutions[winners[1]].optimal_objectives
        print("Finding intermediate solution between", wsol1, wsol2)
        # return solve_intermediate_solutions_only_objs(problem, wsol1, wsol2, num_desired=3)
        return solve_intermediate_solutions(problem, wsol1, wsol2, num_desired=1)[0]
    else:
        print("TIE-breaking, select first solution (group nimbus scalarization)")
        # TODO: go to tie-breaking rule
    # TODO: make according to what GNIMBUS think is the best, for now let us select the first one
        return solutions[0]  # need to unlist the winners list


def agg_cardinal(classification_list, problem, current_objectives):
    sums = {}
    M = len(classification_list)

    # TODO: make chatgpt code simpler xd
    for record in classification_list:
        for key, (operator, value) in record.items():
            if value is not None:  # Only add non-None values
                if key not in sums:
                    sums[key] = 0.0
                sums[key] += (current_objectives[key] - value) / M

    # Convert the sums to a vector of sums (in key order)
    result_vector = [sums[key] for key in sorted(sums.keys())]

    return result_vector
    """
    R = []
    for j in range(len(classification_list)):  # each c[i][j]
        res = 0
        for key, val in classification_list[j].items():

            print(f"Key: {key}, Float value: {val}")
            if val is not None:
                res += current_objectives[key] -


            # R.append(classification_list[j])
    # for key, (str_value, float_value) in classification_list.items():
    """
def solve_intermediate_solutions_only_objs(  # noqa: PLR0913
    problem: Problem,
    solution_1: dict[str, VariableType],
    solution_2: dict[str, VariableType],
    num_desired: int,
    scalarization_options: dict | None = None,
    solver: BaseSolver | None = None,
    solver_options: SolverOptions | None = None,
) -> list[SolverResults]:
    """Generates a desired number of intermediate solutions between two given solutions.

    Generates a desires number of intermediate solutions given two Pareto optimal solutions.
    The solutions are generated by taking n number of steps between the two solutions in the
    objective space. The objective vectors corresponding to these solutions are then
    utilized as reference points in the achievement scalarizing function. Solving the functions
    for each reference point will project the reference point on the Pareto optimal
    front of the problem. These projected solutions are then returned. Note that the
    intermediate solutions are generated _between_ the two given solutions, this means the
    returned solutions will not include the original points.

    Args:
        problem (Problem): the problem being solved.
        solution_1 (dict[str, VariableType]): the first of the solutions between which the intermediate
            solutions are to be generated.
        solution_2 (dict[str, VariableType]): the second of the solutions between which the intermediate
            solutions are to be generated.
        num_desired (int): the number of desired intermediate solutions to be generated. Must be at least `1`.
        scalarization_options (dict | None, optional): optional kwargs passed to the scalarization function.
            Defaults to None.
        solver (BaseSolver | None, optional): solver used to solve the problem.
            If not given, an appropriate solver will be automatically determined based on the features of `problem`.
            Defaults to None.
        solver_options (SolverOptions | None, optional): optional options passed
            to the `solver`. Ignored if `solver` is `None`.
            Defaults to None.

    Returns:
        list[SolverResults]: a list with the projected intermediate solutions as
            `SolverResults` objects.
    """

    if int(num_desired) < 1:
        msg = f"The given number of desired intermediate ({num_desired=}) solutions must be at least 1."
        raise NimbusError(msg)

    init_solver = guess_best_solver(problem) if solver is None else solver
    _solver_options = None if solver_options is None or solver is None else solver_options

    # compute the element-wise difference between each solution (in the decision space)
    solution_1_arr = objective_dict_to_numpy_array(problem, solution_1)
    solution_2_arr = objective_dict_to_numpy_array(problem, solution_2)
    delta = solution_1_arr - solution_2_arr

    # the '2' is in the denominator because we want to calculate the steps
    # between the two given points; we are not interested in the given points themselves.
    step_size = delta / (2 + num_desired)

    intermediate_points = np.array([solution_2_arr + i * step_size for i in range(1, num_desired + 1)])

    intermediate_var_values = pl.DataFrame(
        [unflatten_variable_array(problem, x) for x in intermediate_points],
        schema=[
            (var.symbol, pl.Float64 if isinstance(var, Variable) else pl.Array(pl.Float64, tuple(var.shape)))
            for var in problem.variables
        ],
    )

    # evaluate the intermediate points to get reference points
    # TODO(gialmisi): an evaluator might have to be selected depending on the problem
    evaluator = PolarsEvaluator(problem)

    print("intermediate points:", intermediate_points)
    print("==")
    reference_points = (
        evaluator.evaluate(intermediate_var_values).select([obj.symbol for obj in problem.objectives]).to_dicts()
    )

    print("==")
    print("intermediate rps:", reference_points)
    # for each reference point, add and solve the ASF scalarization problem
    # projecting the reference point onto the Pareto optimal front of the problem.
    # TODO(gialmisi): this can be done in parallel.
    intermediate_solutions = []
    for rp in reference_points:
        # add scalarization
        add_asf = add_asf_diff if problem.is_twice_differentiable else add_asf_nondiff
        asf_problem, target = add_asf(problem, "target", rp, **(scalarization_options or {}))

        solver = init_solver(asf_problem, _solver_options)

        # solve and store results
        result: SolverResults = solver.solve(target)

        intermediate_solutions.append(result)

    return intermediate_solutions

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

def combine_ord_card_prefs(problem, rps, ord_agg):
    """

    for obj in problem.objectives:
        for i in classification_list:
            if classification_list[i][obj.symbol] not in [0, 1, 2]:
                agg_value = 0
                agg = [reference_points[rp][obj.symbol] for rp in reference_points]
                agg_value = np.sum(agg)/3  # number of DMs, taking mean value.

                classification = {obj.symbol: str(classification_list[i][obj.symbol], agg_value)}

            classification_list.append(classification)
    """
    pass


def infer_ordinal_classifications(
    problem: Problem, current_objectives: dict[str, float], reference_points: dict[str, dict[str, float]]
) -> dict[str, tuple[str, float | None]]:
    """
    TODO: improve, currently only works proprely if DMs RPS are either near nadir, ideal or the current iteration point.
    Returns lists for each DM containing value for each objective function with integers 0,1,2 for impair, keep the same and improve respectively.
    """

    if None in problem.get_ideal_point() or None in problem.get_nadir_point():
        msg = "The given problem must have both an ideal and nadir point defined."
        raise NimbusError(msg)

    """ TODO:
    if not all(obj.symbol in reference_point for obj in problem.objectives):
        msg = f"The reference point {reference_point} is missing entries " "for one or more of the objective functions."
        raise NimbusError(msg)
    """
    if not all(obj.symbol in current_objectives for obj in problem.objectives):
        msg = (
            f"The current point {current_objectives} is missing entries " "for one or more of the objective functions."
        )
        raise NimbusError(msg)

    # derive the classifications based on the reference point and and previous
    # objective function values

    # example = np.array([[2, 0, 1, 2], [1, 0, 2, 1], [2, 1, 0, 1]])
    classifications = []
    print(current_objectives)

    # print("curr objs", current_objectives)
    # get number of DMs here
    # TODO: this needs to be adapted for somehow handle cardinal information, e.g. just aggregate
    for rp in reference_points:
        class_for_dm = []
        # print("=====")
        for obj in problem.objectives:
            # print("at 1", reference_points[rp])
            # print("at 2", reference_points[rp][obj.symbol])
            if np.isclose(reference_points[rp][obj.symbol], obj.nadir, atol=0.1):
                # the objective is free to change
                # print("free to change")
                class_for_dm.append(0)
            elif np.isclose(reference_points[rp][obj.symbol], obj.ideal, atol=0.1):
                # the objective should improve
                # print("improvmeent needed")
                class_for_dm.append(2)
            elif np.isclose(reference_points[rp][obj.symbol], current_objectives[obj.symbol], atol=0.1):
                # the objective should stay as it is
                # print("stay the same")
                class_for_dm.append(1)
            else:  # TODO: very stupid aggre.lets stupidly aggregate here?
                v = None
                if current_objectives[obj.symbol] < reference_points[rp][obj.symbol] < obj.nadir:
                    print(reference_points[rp][obj.symbol])
                    v = 2
                elif current_objectives[obj.symbol] > reference_points[rp][obj.symbol] > obj.ideal:
                    print(reference_points[rp][obj.symbol])
                    v = 0
                class_for_dm.append(v)

        # print("at 3", class_for_dm)  # 0, 1, 2
        classifications.append(class_for_dm)
        # classifications |= classification

    """
    for obj in problem.objectives:
        if np.isclose(reference_point[obj.symbol], obj.nadir):
            # the objective is free to change
            classification = {obj.symbol: ("0", None)}
        elif np.isclose(reference_point[obj.symbol], obj.ideal):
            # the objective should improve
            classification = {obj.symbol: ("<", None)}
        elif np.isclose(reference_point[obj.symbol], current_objectives[obj.symbol]):
            # the objective should stay as it is
            classification = {obj.symbol: ("=", None)}
        elif not obj.maximize and reference_point[obj.symbol] < current_objectives[obj.symbol]:
            # minimizing objective, reference value smaller, this is an aspiration level
            # improve until
            classification = {obj.symbol: ("<=", reference_point[obj.symbol])}
        elif not obj.maximize and reference_point[obj.symbol] > current_objectives[obj.symbol]:
            # minimizing objective, reference value is greater, this is a reservations level
            # impair until
            classification = {obj.symbol: (">=", reference_point[obj.symbol])}
        elif obj.maximize and reference_point[obj.symbol] < current_objectives[obj.symbol]:
            # maximizing objective, reference value is smaller, this is a reservation level
            # impair until
            classification = {obj.symbol: (">=", reference_point[obj.symbol])}
        elif obj.maximize and reference_point[obj.symbol] > current_objectives[obj.symbol]:
            # maximizing objective, reference value is greater, this is an aspiration level
            # improve until
            classification = {obj.symbol: ("<=", reference_point[obj.symbol])}
        else:
            # could not figure classification
            msg = f"Warning: NIMBUS could not figure out the classification for objective {obj.symbol}."

        classifications |= classification
    """

    print("CLASSES", classifications)
    return np.array(classifications)


# TODO: the version of nimbus g scalarization that can take multiple classifications! We respect everyone's bounds but aspirations must flex.
# TODO: CHECK THE I sets
def add_group_nimbusv2_sf_diff(  # noqa: PLR0913
    problem: Problem,
    symbol: str,
    classifications_list: list[dict[str, tuple[str, float | None]]],
    current_objective_vector: dict[str, float],
    delta: float = 0.000001,
    rho: float = 0.000001,
) -> tuple[Problem, str]:
    r"""Implements the differentiable variant of the multiple decision maker of the group NIMBUS scalarization function.

    The scalarization function is defined as follows:

    \begin{align}
        \mbox{minimize} \quad
         &\alpha +
        \rho \sum^k_{i=1} \sum^{n_d}_{d=1} w_{id}f_{id}(\mathbf{x})\\
        \mbox{subject to} \quad & w_{id}(f_{id}(\mathbf{x})-z^{ideal}_{id}) - \alpha \leq 0 \quad & \forall i \in I^<,\\
        & w_{jd}(f_{jd}(\mathbf{x})-\hat{z}_{jd}) - \alpha \leq 0 \quad & \forall j \in I^\leq ,\\
        & f_i(\mathbf{x}) - f_i(\mathbf{x_c}) \leq 0 \quad & \forall i \in I^< \cup I^\leq \cup I^= ,\\
        & f_i(\mathbf{x}) - \epsilon_i \leq 0 \quad & \forall i \in I^\geq ,\\
        & \mathbf{x} \in \mathbf{X},
    \end{align}

    where $w_{id} = \frac{1}{z^{nad}_{id} - z^{uto}_{id}}$, and $w_{jd} = \frac{1}{z^{nad}_{jd} - z^{uto}_{jd}}$.

    The $I$-sets are related to the classifications given to each objective function value
    in respect to  the current objective vector (e.g., by a decision maker). They
    are as follows:

    - $I^{<}$: values that should improve,
    - $I^{\leq}$: values that should improve until a given aspiration level $\hat{z}_i$. DOES NOT CONTAIN I^< !
    - $I^{=}$: values that are fine as they are,
    - $I^{\geq}$: values that can be impaired until some reservation level $\varepsilon_i$. DOES NOT CONTAIN I^=
    - $I^{\diamond}$: values that are allowed to change freely (not present explicitly in this scalarization function). DOES NOT CONTAIN I^gep

    The aspiration levels and the reservation levels are supplied for each classification, when relevant, in
    the argument `classifications` as follows:

    ```python
    classifications = {
        "f_1": ("<", None),
        "f_2": ("<=", 42.1),
        "f_3": (">=", 22.2),
        "f_4": ("0", None)
        }
    ```

    Here, we have assumed four objective functions. The key of the dict is a function's symbol, and the tuple
    consists of a pair where the left element is the classification (self explanatory, '0' is for objective values
    that may change freely), the right element is either `None` or an aspiration or a reservation level
    depending on the classification.

    Args:
        problem (Problem): the problem to be scalarized.
        symbol (str): the symbol given to the scalarization function, i.e., target of the optimization.
        classifications_list (list[dict[str, tuple[str, float  |  None]]]): a list of dicts, where the key is a symbol
            of an objective function, and the value is a tuple with a classification and an aspiration
            or a reservation level, or `None`, depending on the classification. See above for an
            explanation.
        current_objective_vector (dict[str, float]): the current objective vector that corresponds to
            a Pareto optimal solution. The classifications are assumed to been given in respect to
            this vector.
        delta (float, optional): a small scalar used to define the utopian point. Defaults to 0.000001.
        rho (float, optional): a small scalar used in the augmentation term. Defaults to 0.000001.

    Raises:
        ScalarizationError: any of the given classifications do not define a classification
            for all the objective functions or any of the given classifications do not allow at
            least one objective function value to improve and one to worsen.

    Returns:
        tuple[Problem, str]: a tuple with the copy of the problem with the added
            scalarization and the symbol of the added scalarization.
    """
    # TODO: these check shoudl happen for individual DMs
    # check that classifications have been provided for all objective functions
    """
    for classifications in classifications_list:
        if not objective_dict_has_all_symbols(problem, classifications):
            msg = (
                f"The given classifications {classifications} do not define "
                "a classification for all the objective functions."
            )
            raise ScalarizationError(msg)

        # check that at least one objective function is allowed to be improved and one is
        # allowed to worsen
        if not any(classifications[obj.symbol][0] in ["<", "<="] for obj in problem.objectives) or not any(
            classifications[obj.symbol][0] in [">=", "0"] for obj in problem.objectives
        ):
            msg = (
                f"The given classifications {classifications} should allow at least one objective function value "
                "to improve and one to worsen."
            )
            raise ScalarizationError(msg)
    """

    # check ideal and nadir exist
    ideal, nadir = get_corrected_ideal(problem), get_corrected_nadir(problem)
    corrected_current_point = get_corrected_reference_point(problem, current_objective_vector)

    # define the auxiliary variable
    alpha = Variable(
        name="alpha",
        symbol="_alpha",
        variable_type=VariableTypeEnum.real,
        lowerbound=-float("Inf"),
        upperbound=float("Inf"),
        initial_value=1.0,
    )

    # calculate the weights
    weights = {obj.symbol: 1 / (nadir[obj.symbol] - (ideal[obj.symbol] - delta)) for obj in problem.objectives}

    constraints = []

    """
    for i in range(len(classifications_list)):
        classifications = classifications_list[i]
        for obj in problem.objectives:
            _symbol = obj.symbol
            match classifications[_symbol]:
    """

    # TODO: FIX, does not work because dict's are unhashable, so cannot match with classifications[dict][dict]key stuff figure out
    # for i, dm_class in enumerate(classifications_list):
    for i in range(len(classifications_list)):
        # classifications = classifications_list[dm_class][i]
        classifications = classifications_list[i]
        for obj in problem.objectives:
            _symbol = obj.symbol
            match classifications[_symbol]:
                case ("<", _):
                    max_expr = f"{weights[_symbol]} * ({_symbol}_min - {ideal[_symbol]}) - _alpha"
                    constraints.append(
                        Constraint(
                            name=f"Max term linearization for {_symbol}",
                            symbol=f"max_con_{_symbol}_{i+1}",
                            func=max_expr,
                            cons_type=ConstraintTypeEnum.LTE,
                            is_linear=problem.is_linear,
                            is_convex=problem.is_convex,
                            is_twice_differentiable=problem.is_twice_differentiable,
                        )
                    )
                    con_expr = f"{_symbol}_min - {corrected_current_point[_symbol]}"
                    constraints.append(
                        Constraint(
                            name=f"improvement constraint for {_symbol}",
                            symbol=f"{_symbol}_{i+1}_lt",
                            func=con_expr,
                            cons_type=ConstraintTypeEnum.LTE,
                            is_linear=problem.is_linear,
                            is_convex=problem.is_convex,
                            is_twice_differentiable=problem.is_twice_differentiable,
                        )
                    )
                case ("<=", aspiration):
                    # if obj is to be maximized, then the current aspiration value needs to be multiplied by -1
                    max_expr = (
                        f"{weights[_symbol]} * ({_symbol}_min - {aspiration * -1 if obj.maximize else aspiration}) "
                        "- _alpha"
                    )
                    constraints.append(
                        Constraint(
                            name=f"Max term linearization for {_symbol}",
                            symbol=f"max_con_{_symbol}_{i+1}",
                            func=max_expr,
                            cons_type=ConstraintTypeEnum.LTE,
                            is_linear=problem.is_linear,
                            is_convex=problem.is_convex,
                            is_twice_differentiable=problem.is_twice_differentiable,
                        )
                    )
                    con_expr = f"{_symbol}_min - {corrected_current_point[_symbol]}"
                    constraints.append(
                        Constraint(
                            name=f"improvement until constraint for {_symbol}",
                            symbol=f"{_symbol}_{i+1}_lte",
                            func=con_expr,
                            cons_type=ConstraintTypeEnum.LTE,
                            is_linear=problem.is_linear,
                            is_convex=problem.is_convex,
                            is_twice_differentiable=problem.is_twice_differentiable,
                        )
                    )
                case ("=", _):
                    con_expr = f"{_symbol}_min - {corrected_current_point[_symbol]}"
                    constraints.append(
                        Constraint(
                            name=f"Stay as good constraint for {_symbol}",
                            symbol=f"{_symbol}_{i+1}_eq",
                            func=con_expr,
                            cons_type=ConstraintTypeEnum.LTE,  # OR EQ ?
                            is_linear=problem.is_linear,
                            is_convex=problem.is_convex,
                            is_twice_differentiable=problem.is_twice_differentiable,
                        )
                    )
                case (">=", reservation):
                    # if obj is to be maximized, then the current reservation value needs to be multiplied by -1
                    con_expr = f"{_symbol}_min - {-1 * reservation if obj.maximize else reservation}"
                    constraints.append(
                        Constraint(
                            name=f"Worsen until constraint for {_symbol}",
                            symbol=f"{_symbol}_{i+1}_gte",
                            func=con_expr,
                            cons_type=ConstraintTypeEnum.LTE,
                            is_linear=problem.is_linear,
                            is_convex=problem.is_convex,
                            is_twice_differentiable=problem.is_twice_differentiable,
                        )
                    )
                case ("0", _):
                    # not relevant for this scalarization
                    pass
                case (c, _):
                    msg = (
                        f"Warning! The classification {c} was supplied, but it is not supported."
                        "Must be one of ['<', '<=', '0', '=', '>=']"
                    )

    # form the augmentation term
    aug_exprs = []
    for _ in range(len(classifications_list)):
        aug_expr = " + ".join([f"({weights[obj.symbol]} * {obj.symbol}_min)" for obj in problem.objectives])
        aug_exprs.append(aug_expr)
    aug_exprs = " + ".join(aug_exprs)

    func = f"_alpha + {rho} * ({aug_exprs})"
    scalarization_function = ScalarizationFunction(
        name="Differentiable NIMBUS scalarization objective function for multiple decision makers",
        symbol=symbol,
        func=func,
        is_linear=problem.is_linear,
        is_convex=problem.is_convex,
        is_twice_differentiable=problem.is_twice_differentiable,
    )
    _problem = problem.add_variables([alpha])
    _problem = _problem.add_scalarization(scalarization_function)
    return _problem.add_constraints(constraints), symbol


def convert_to_nimbus_classification(problem: Problem, compromise_classification: list[int]) -> dict[str, tuple[str, float | None]]:
    r""" Converts compromise classification for the group to the NIMBUS classification format.
    """
    classifications = {}
    for i in range(len(compromise_classification)):
        if compromise_classification[i] == 0:
            # the objective is free to change
            classification = {problem.objectives[i].symbol: ("0", None)}
        elif compromise_classification[i] == 1:
            # the objective should stay as it is
            classification = {problem.objectives[i].symbol: ("=", None)}
        elif compromise_classification[i] == 2:
            # the objective should improve
            classification = {problem.objectives[i].symbol: ("<", None)}
        # else:
        #    msg = f"Warning: GNIMBUS could not figure out the classification for objective {problem.objectives[i].symbol}."
        #    raise NimbusError(msg)

        classifications |= classification
    return classifications


def solve_sub_problems(  # noqa: PLR0913
    problem: Problem,
    current_objectives: dict[str, float],
    reference_points: dict[str, dict[str, float]],
    num_desired: int,
    decision_phase: False,
    scalarization_options: dict | None = None,
    create_solver: BaseSolver | None = None,
    solver_options: SolverOptions | None = None,
) -> list[SolverResults]:
    r"""Solves a desired number of sub-problems as defined in the NIMBUS methods.

    Solves 1-4 scalarized problems utilizing different scalarization
    functions. The scalarizations are based on the classification of a
    solutions provided by a decision maker. The classifications
    are represented by a reference point. Returns a number of new solutions
    corresponding to the number of scalarization functions solved.

    Depending on `num_desired`, solves the following scalarized problems corresponding
    the the following scalarization functions:

    1.  the NIMBUS scalarization function,
    2.  the STOM scalarization function,
    3.  the achievement scalarizing function, and
    4.  the GUESS scalarization function.

    Raises:
        NimbusError: the given problem has an undefined ideal or nadir point, or both.
        NimbusError: either the reference point of current objective functions value are
            missing entries for one or more of the objective functions defined in the problem.

    Args:
        problem (Problem): the problem being solved.
        current_objectives (dict[str, float]): an objective dictionary with the objective functions values
            the classifications have been given with respect to.
        reference_point (dict[str, float]): an objective dictionary with a reference point.
            The classifications utilized in the sub problems are derived from
            the reference point.
        num_desired (int): the number of desired solutions to be solved. Solves as
            many scalarized problems. The value must be in the range 1-4.
        scalarization_options (dict | None, optional): optional kwargs passed to the scalarization function.
            Defaults to None.
        create_solver (CreateSolverType | None, optional): a function that given a problem, will return a solver.
            If not given, an appropriate solver will be automatically determined based on the features of `problem`.
            Defaults to None.
        solver_options (SolverOptions | None, optional): optional options passed
            to the `create_solver` routine. Ignored if `create_solver` is `None`.
            Defaults to None.

    Returns:
        list[SolverResults]: a list of `SolverResults` objects. Contains as many elements
            as defined in `num_desired`.
    """
    if None in problem.get_ideal_point() or None in problem.get_nadir_point():
        msg = "The given problem must have both an ideal and nadir point defined."
        raise NimbusError(msg)

    # TODO: update these tests for multiple RPs

    """
    for reference_point in reference_points:
        if not all(obj.symbol in reference_point for obj in problem.objectives):
            msg = f"The reference point {reference_point} is missing entries " "for one or more of the objective functions."
            raise NimbusError(msg)

        if not all(obj.symbol in current_objectives for obj in problem.objectives):
            msg = f"The current point {reference_point} is missing entries " "for one or more of the objective functions."
            raise NimbusError(msg)

    """

    init_solver = create_solver if create_solver is not None else guess_best_solver(problem)
    _solver_options = solver_options if solver_options is not None else None

    print("CURRENT SOLVER", init_solver)

    solutions = []
    classification_list = []

    if decision_phase:
        for dm_rp in reference_points:
            print("RPS", reference_points[dm_rp])
            classification_list.append(infer_classifications(problem, current_objectives, reference_points[dm_rp]))
        gnimbus_scala = add_group_nimbusv2_sf_diff if problem.is_twice_differentiable else add_group_nimbus_sfv2  # non-diff gnimbus
        add_nimbus_sf = gnimbus_scala

        problem_w_nimbus, nimbus_target = add_nimbus_sf(
            problem, "nimbus_sf", classification_list, current_objectives, **(scalarization_options or {})
        )

        if _solver_options:
            nimbus_solver = init_solver(problem_w_nimbus, _solver_options)  # type:ignore
        else:
            nimbus_solver = init_solver(problem_w_nimbus)

        solutions.append(nimbus_solver.solve(nimbus_target))

        return solutions

    else:
        reference_points_list = dict_of_rps_to_list_of_rps(reference_points)

        for dm_rp in reference_points:
            # classification_list.append(infer_classifications(problem, current_objectives, reference_points[dm_rp]))
            """
            classification = infer_classifications(problem, current_objectives, reference_points[dm_rp])
            print("class", classification_list)
            # gnimbus_scala = add_group_nimbusv2_sf_diff if problem.is_twice_differentiable else add_group_nimbus_sfv2  # non-diff gnimbus
            nimbus_scala = add_nimbus_sf_diff if problem.is_twice_differentiable else add_nimbus_sf_nondiff  # non-diff gnimbus
            add_nimbus_sf = nimbus_scala

            problem_w_nimbus, nimbus_target = add_nimbus_sf(
                problem, "nimbus_sf", classification, current_objectives, **(scalarization_options or {})
            )

            if _solver_options:
                nimbus_solver = init_solver(problem_w_nimbus, _solver_options)
            else:
                nimbus_solver = init_solver(problem_w_nimbus)

            solutions.append(nimbus_solver.solve(nimbus_target))
            """

            # solve STOM
            add_stom_sf = add_stom_sf_diff if problem.is_twice_differentiable else add_stom_sf_nondiff

            problem_w_stom, stom_target = add_stom_sf(problem, "stom_sf", reference_points[dm_rp], **(scalarization_options or {}))
            if _solver_options:
                stom_solver = init_solver(problem_w_stom, _solver_options)
            else:
                stom_solver = init_solver(problem_w_stom)

            solutions.append(stom_solver.solve(stom_target))

                    # solve GUESS
            add_guess_sf = add_guess_sf_diff if problem.is_twice_differentiable else add_guess_sf_nondiff

            problem_w_guess, guess_target = add_guess_sf(
                problem, "guess_sf", reference_points[dm_rp], **(scalarization_options or {})
            )

            if _solver_options:
                guess_solver = init_solver(problem_w_guess, _solver_options)
            else:
                guess_solver = init_solver(problem_w_guess)

            solutions.append(guess_solver.solve(guess_target))

        # solve STOM
        add_stom_sf = add_group_stom_sf_diff if problem.is_twice_differentiable else add_group_stom_sf

        d = 1e-06
        ideal = problem.get_ideal_point()
        nadir = problem.get_nadir_point()
        delta = {
            "Rev": d*(ideal["Rev"] - nadir["Rev"]) ,
            "HA": d*(ideal["HA"] - nadir["HA"]),
            "Carb":d*(ideal["Carb"] - nadir["Carb"]),
            "DW":d*(ideal["DW"] - nadir["DW"]),
        }
 
        problem_w_stom, stom_target = add_stom_sf(problem, "stom_sf", reference_points_list, delta, **(scalarization_options or {}))
        if _solver_options:
            stom_solver = init_solver(problem_w_stom, _solver_options)
        else:
            stom_solver = init_solver(problem_w_stom)

        solutions.append(stom_solver.solve(stom_target))

        # solve ASF
        add_asf = add_group_asf_diff if problem.is_twice_differentiable else add_group_asf

        problem_w_asf, asf_target = add_asf(problem, "asf", reference_points_list, delta, **(scalarization_options or {}))

        if _solver_options:
            asf_solver = init_solver(problem_w_asf, _solver_options)
        else:
            asf_solver = init_solver(problem_w_asf)

        solutions.append(asf_solver.solve(asf_target))

        # solve GUESS
        add_guess_sf = add_group_guess_sf_diff if problem.is_twice_differentiable else add_group_guess_sf

        problem_w_guess, guess_target = add_guess_sf(
            problem, "guess_sf", reference_points_list, delta, **(scalarization_options or {})
        )

        if _solver_options:
            guess_solver = init_solver(problem_w_guess, _solver_options)
        else:
            guess_solver = init_solver(problem_w_guess)

        solutions.append(guess_solver.solve(guess_target))

        return solutions
