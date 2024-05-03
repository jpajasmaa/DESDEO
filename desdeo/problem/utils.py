"""Various utilities used accross the framework related to the Problem formulation."""

import numpy as np

from desdeo.problem import Problem


def objective_dict_to_numpy_array(problem: Problem, objective_dict: dict[str, float]) -> np.ndarray:
    """Takes a dict with an objective vector and returns a numpy array.

    Takes a dict with the keys being objective function symbols and the values
    being the corresponding objective function values. Returns a numpy array
    with the objective function values in the same order they have been defined
    in the original problem.

    Args:
        problem (Problem): the problem the objective dict belongs to.
        objective_dict (dict[str, float]): the dict with the objective function values.

    Returns:
        np.ndarray: a numpy array with the objective function values in the order they are
            present in problem.
    """
    if isinstance(objective_dict[problem.objectives[0].symbol], list):
        if len(objective_dict[problem.objectives[0].symbol]) != 1:
            raise ValueError("The objective_dict has multiple values for an objective function")
        return np.array([objective_dict[objective.symbol][0] for objective in problem.objectives])
    return np.array([objective_dict[objective.symbol] for objective in problem.objectives])


def numpy_array_to_objective_dict(problem: Problem, numpy_array: np.ndarray) -> dict[str, float]:
    """Takes a numpy array with objective function values and return a dict.

    The reverse of objective_dict_to_numpy_array.

    Args:
        problem (Problem): the problem the numpy array represents an objective vector of.
        numpy_array (np.ndarray): the objective vector as a numpy array. The
            array is squeezed, i.e., axes or length one are removed: [[42]] -> [42].

    Returns:
        dict[str, float]: a dict with keys being objective function symbols and value being
            objective function values.
    """
    return {objective.symbol: np.squeeze(numpy_array).tolist()[i] for i, objective in enumerate(problem.objectives)}


def variable_dict_to_numpy_array(problem: Problem, variable_dict: dict[str, float]) -> np.ndarray:
    """Takes a dict with a decision variable vector and returns a numpy array.

    Takes a dict with the keys being decision variable symbols and the values
    being the corresponding variable values. Returns a numpy array
    with the decision variable values in the same order they have been defined in
    the original problem.

    Args:
        problem (Problem): the problem the objective dict belongs to.
        variable_dict (dict[str, float]): the dict with the decision variable values.

    Returns:
        np.ndarray: a numpy array with the decision variable values in the order they are
            present in problem.
    """
    if isinstance(variable_dict[problem.variables[0].symbol], list):
        if len(variable_dict[problem.variables[0].symbol]) != 1:
            raise ValueError("The variable_dict has multiple values for a decision variable.")
        return np.array([variable_dict[variable.symbol][0] for variable in problem.variables])
    return np.array([variable_dict[variable.symbol] for variable in problem.variables])


def get_nadir_dict(problem: Problem) -> dict[str, float]:
    """Return a dict representing a problem's nadir point.

    Args:
        problem (Problem): the problem with the nadir point.

    Returns:
        dict[str, float]: key are objective funciton symbols, values are nadir values.
    """
    return {objective.symbol: objective.nadir for objective in problem.objectives}


def get_ideal_dict(problem: Problem) -> dict[str, float]:
    """Return a dict representing a problem's ideal point.

    Args:
        problem (Problem): the problem with the ideal point.

    Returns:
        dict[str, float]: key are objective funciton symbols, values are ideal values.
    """
    return {objective.symbol: objective.ideal for objective in problem.objectives}
