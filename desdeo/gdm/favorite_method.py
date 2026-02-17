"""The Favorite method, a general method for group decision making in multiobjective optimization"""

from typing import Literal, List, Dict
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from typing import Literal

import numpy as np
import polars as pl
import pydantic
from pydantic import ConfigDict, Field

from desdeo.emo import DesirableRangesOptions, emo_constructor, nsga3_options
from desdeo.emo.options.generator import ArchiveGeneratorOptions
from desdeo.emo.options.templates import EMOOptions
from desdeo.gdm.gdmtools import (
    agg_aspbounds,
    alpha_fairness,
    dict_of_rps_to_list_of_rps,
    get_top_n_fair_solutions,
    min_max_regret_no_impro,
    scale_rp,
)
from desdeo.gdm.voting_rules import majority_rule
from desdeo.mcdm.nautilus_navigator import calculate_navigation_point
from desdeo.problem import (
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)
from desdeo.problem.schema import Problem
from desdeo.tools import PyomoIpoptSolver
from desdeo.tools.GenerateReferencePoints import generate_points, rotate_in, rotate_out, get_hull_equations, numba_random_gen
from desdeo.tools.generics import EMOResult
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint, choose_reference_point
from desdeo.tools.scalarization import add_asf_diff


# --- Classes & Options ---

class IPR_Options(pydantic.BaseModel):
    """Options specific to iterative_pareto_representer applied with the favorite method."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    num_initial_reference_points: int = Field(default=10000, ge=1)
    """Big number"""
    version: Literal["convex_hull", "box"] = "convex_hull"
    (
        """Version "convex_hull": evaluate in the convex hull of MPS."""
        """ Version "box": evaluate in the box of fake_ideal and fake_nadir."""
    )
    most_preferred_solutions: dict[str, dict[str, float]] | None = None
    """Most preferred solutions of the decision makers. Should be filled in by code, not by user."""


class GPRMOptions(pydantic.BaseModel):
    """Pydantic model to contain options for the `get_representative_set` function."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    method_options: IPR_Options | None = Field(default_factory=IPR_Options)
    """Options specific to the selected method. None for EMO"""
    fake_ideal: dict[str, float] | None = None
    """Fake ideal point. Should be filled in by code, not by user."""
    fake_nadir: dict[str, float] | None = None
    """Fake nadir point. Should be filled in by code, not by user."""
    num_points_to_evaluate: int = Field(default=100, ge=1)
    """Number of points to evaluate in the IPR method, or population size in EMO methods."""


class IPR_Results(pydantic.BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    evaluated_points: list[_EvaluatedPoint]  # I dont know how to use this properly.


class GPRMResults(pydantic.BaseModel):
    """Pydantic model to contain results from the `get_representative_set` function."""

    model_config = ConfigDict(use_attribute_docstrings=True, arbitrary_types_allowed=True)

    raw_results: IPR_Results | EMOResult
    """Raw results from the selected method."""
    solutions: pl.DataFrame | None
    """DataFrame containing the evaluated solutions (inputs)."""
    outputs: pl.DataFrame
    """DataFrame containing the evaluated outputs."""


class FairSolution(pydantic.BaseModel):
    """Represents a single solution identified as 'fair' or a 'candidate'."""
    objective_values: dict[str, float]
    """Objective values map (e.g., {'f_1': 0.2})."""

    fairness_criterion: str
    """The criterion used to select this solution (e.g., 'mm', 'nash', 'avg_hausdorff')."""

    fairness_value: float
    """The numerical score associated with the fairness criterion."""

class ZoomOptions(pydantic.BaseModel):
    """Pydantic model to contain options for zooming strategy."""

    method: Literal["nautilus"] = "nautilus"
    """Zooming method to use."""
    num_steps_remaining: int = Field(default=5, ge=1)
    """Number of remaining zooming steps. Determines step size. Must be positive integer."""


class FavOptions(pydantic.BaseModel):
    """Pydantic model to contain options for the favorite method."""

    GPRMoptions: GPRMOptions
    """Options for the representative set method. EMO and IPR supported."""
    candidate_generation_options: str
    (
        """Options for generating candidate fair solutions. For now, just a string to determine the fairness criterion applied."""
        """ Support more options later."""
    )
    zoom_options: ZoomOptions = Field(default_factory=ZoomOptions)
    """Options for the zooming strategy. Support more options later."""
    original_most_preferred_solutions: dict[str, dict[str, float]]
    """Dictionary of the original most preferred solutions for each decision maker."""
    votes: dict[str, int] | None = None
    (
        """The votes for each decision maker's most preferred solution."""
        """ The candidates are from `fair_solutions` in FavResults of the previous iteration."""
        """Not required for the first iteration."""
    )


class FavResults(pydantic.BaseModel):
    """Pydantic model to contain results from one iteration of the favorite method."""

    FavOptions: FavOptions
    """Options used in this iteration of the favorite method."""
    GPRMResults: GPRMResults
    """Results from the representative set method."""
    fair_solutions: list[FairSolution]
    """List of candidate fair solutions found in this iteration."""


# --- Core logic ---

class ProblemWrapper:
    """Wraps a DESDEO Problem to manage Ideal/Nadir updates and solving with IPR."""

    def __init__(self, problem: Problem, fake_ideal: dict[str, float], fake_nadir: dict[str, float]):
        """
        Args:
            problem: The DESDEO Problem.
            fake_ideal: The current fake ideal point.
            fake_nadir: The current fake nadir point.
        """
        self.problem = problem
        self.ideal, self.nadir = fake_ideal, fake_nadir
        self.problem = problem.update_ideal_and_nadir(new_ideal=self.ideal, new_nadir=self.nadir)
        self.evaluated_points: list[_EvaluatedPoint] = []

    # TODO: set solver
    def solve(self, scaled_refp: np.ndarray) -> list[_EvaluatedPoint]:
        """
        Solves a scalarized version of the problem using ASF.

        Args:
            scaled_refp: Reference point coordinates scaled between 0 and 1.

        Returns:
            list[_EvaluatedPoint]: The list of evaluated points including the new solution.
        """
        refp = {
            obj: val * (self.nadir[obj] - self.ideal[obj]) + self.ideal[obj]
            for obj, val in zip(self.ideal.keys(), scaled_refp)
        }
        scaled_problem, target = add_asf_diff(self.problem, "target", refp)
        solver = PyomoIpoptSolver(scaled_problem)
        # solver = guess_best_solver(scaled_problem)
        results = solver.solve(target)
        objs = results.optimal_objectives
        scaled_objs = {obj: (objs[obj] - self.ideal[obj]) / (self.nadir[obj] - self.ideal[obj]) for obj in objs.keys()}
        self.evaluated_points.append(
            _EvaluatedPoint(
                reference_point=dict(zip(self.ideal.keys(), scaled_refp)), targets=scaled_objs, objectives=objs
            )
        )
        return self.evaluated_points

def find_group_solutions(
    problem: Problem,
    solutions: pl.DataFrame,
    targets: pl.DataFrame,
    most_preferred_solutions: dict[str, dict[str, float]],
    fairness_criterion: str
) -> tuple[list[FairSolution], dict]:
    """
        Identifies a fair compromise solution from a set of generated solutions based on a criterion.

        Args:
            problem: The DESDEO problem object.
            solutions: DataFrame of evaluated solutions (input to GPRM).
            targets: DataFrame of evaluated targets (output of GRPM).
            most_preferred_solutions: The most preferred solutions of the DMs.
            fairness_criterion: The string identifier for the fairness rule (e.g., 'utilitarian', 'nash', 'mm').

        Returns:
            tuple: A list containing the best FairSolution found, and a dict of extra metrics (currently empty).
    """

    # TODO: Normalize MPSes
    normalized_mpses = most_preferred_solutions

    # convert to numpy array for numba in UFs
    normalized_mpses_arr = []
    for i, dm in enumerate(normalized_mpses):
        normalized_mpses_arr.append(objective_dict_to_numpy_array(problem, normalized_mpses[dm]).tolist())

    # TODO: change this to match how many FairSolution we want. For now, we just get a single one
    ranking = None
    if fairness_criterion == "utilitarian":
        ranking = alpha_fairness(targets, normalized_mpses_arr, alpha=0.0)  # utilitarian
    elif fairness_criterion == "nash":
        ranking = alpha_fairness(targets, normalized_mpses_arr, alpha=1)  # nash
    elif fairness_criterion == "mm":
        ranking = min_max_regret_no_impro(targets, normalized_mpses_arr)  # minmax regret no improvements
    else:
        raise NotImplementedError("Given fairness criterion not implemented.")
    print(ranking)

    # convert to numpy array for get top fair solutions
    solutions_arr = solutions.to_numpy()
    ranking_r, ranking_i = get_top_n_fair_solutions(solutions_arr, ranking, 1)
    print("fairness rankings")
    print(ranking_r, ranking_i)

    FairSolutions_arr = []
    # Loop for more
    FairSolutions_arr.append(
        FairSolution(
            objective_values=numpy_array_to_objective_dict(problem, ranking_r[0]),
            fairness_criterion=fairness_criterion,
            fairness_value=ranking_i[0],
        )
    )
    # TODO: regret values needed or?
    regret_values = {
        "mean": [],
    }

    return FairSolutions_arr, regret_values


def shift_points(
    problem: Problem,
    most_preferred_solutions,
    group_preferred_solution: dict[str, float],
    steps_remaining: int
) -> dict[str, dict[str, float]]:
    """
    Shifts the DMs' most preferred solutions towards a group compromise using Nautilus navigation logic.

    Args:
        problem: The DESDEO problem.
        most_preferred_solutions: The current reference points of the DMs.
        group_preferred_solution: The objective values of the selected compromise solution (winner).
        steps_remaining: Number of steps left in the process (determines shift magnitude).

    Returns:
        dict: The new, shifted most preferred solutions for each DM.
    """

    shifted_mps = {}
    for dm in most_preferred_solutions:
        shifted_point = calculate_navigation_point(
            problem, most_preferred_solutions[dm], group_preferred_solution, steps_remaining
        )
        p, target = add_asf_diff(
            problem,
            symbol="asf",
            reference_point=shifted_point,
        )
        solver = PyomoIpoptSolver(p)
        res = solver.solve(target)
        shifted_mps.update({dm: res.optimal_objectives})

    return shifted_mps


def get_representative_set_IPR(problem: Problem, options: GPRMOptions, results_list: list[GPRMResults]) -> GPRMResults:
    """
    Generates a set of Pareto optimal solutions using the Iterative Pareto Representer (IPR).

    This method generates reference points in a specific region (hull or box) and solves
    scalarization problems for each.
    """
    if not isinstance(options.method_options, IPR_Options):
        raise TypeError("Expected IPR_Options for IPR method.")

    evaluated_points = []

    # Normalize mps for fairness and IPR. Convert to array for now.
    mps = {}
    for dm in options.method_options.most_preferred_solutions:
        mps.update(
            {
                dm: scale_rp(
                    problem,
                    options.method_options.most_preferred_solutions[dm],
                    options.fake_ideal,
                    options.fake_nadir,
                    False,
                )
            }
        )

    # RPs as array for methods to come
    rp_arr = []
    for _, dm in enumerate(mps):
        rp_arr.append(objective_dict_to_numpy_array(problem, mps[dm]).tolist())

    dims = len(problem.get_nadir_point())

    # get the representative set
    # set n or the possibilities of n according to the num points to evaluate
    for n in [options.num_points_to_evaluate, options.num_points_to_evaluate / 2, 10]:
        try:
            # generate points in the convex hull of RPs or fake_ideal and fake_nadir
            if options.method_options.version == "convex_hull":
                _, refp = generate_points(
                    num_points=options.method_options.num_initial_reference_points,
                    num_dims=dims,
                    reference_points=rp_arr,
                )
            else:
                _, refp = generate_points(
                    num_points=options.method_options.num_initial_reference_points, num_dims=dims, reference_points=None
                )

            num_runs = n
            wrapped_problem = ProblemWrapper(problem, fake_ideal=options.fake_ideal, fake_nadir=options.fake_nadir)
            for i in range(num_runs):
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"Run {i + 1}/{num_runs}")
                reference_point, _ = choose_reference_point(refp, evaluated_points)
                evaluated_points = wrapped_problem.solve(reference_point)
            break
        except Exception:
            break

    ipr_res = IPR_Results(evaluated_points=evaluated_points)

    results = GPRMResults(
        raw_results=ipr_res,
        solutions=None,
        outputs=pl.DataFrame([point.objectives for point in evaluated_points]),
    )

    return results


def get_representative_set_EMO(problem: Problem, options: GPRMOptions, results_list: list[GPRMResults]) -> GPRMResults:
    """Get the representative set according to EMOOptions.
    Generates a set of solutions using an Evolutionary Algorithm (NSGA-III).
    Uses the previous iteration's population as a seed if available.

    Args:
        problem: DESDEO Problem object
        options: EMOOptions
        results_list: list of previous EMOResult objects

    Returns:

    """
    opts = nsga3_options()
    dr_opts = DesirableRangesOptions(
        aspiration_levels=options.fake_ideal,
        reservation_levels=options.fake_nadir,
        method="DF transformation",
        desirability_levels=(0.999, 0.001),
    )
    opts.preference = dr_opts
    opts.template.generator.n_points = options.num_points_to_evaluate
    opts.template.selection.reference_vector_options.number_of_vectors = options.num_points_to_evaluate
    opts.template.selection.invert_reference_vectors = True
    if results_list:
        opts.template.generator = ArchiveGeneratorOptions(
            solutions=results_list[-1].solutions.select([var.symbol for var in problem.get_flattened_variables()]),
            outputs=results_list[-1].outputs.select([obj.name for obj in problem.objectives]),
        )
    solver, extras = emo_constructor(problem=problem, emo_options=opts)
    res = solver()
    archive_results = extras.archive.results
    var_cols = archive_results.optimal_variables.columns
    obj_cols = archive_results.optimal_outputs.columns
    solutions = pl.concat([archive_results.optimal_variables, archive_results.optimal_outputs], how="horizontal")

    for obj in problem.objectives:
        if obj.maximize:
            solutions = solutions.filter(
                (
                    pl.col(obj.symbol) >= options.fake_nadir[obj.symbol]
                    # & (pl.col(obj.symbol) <= options.fake_ideal[obj.symbol])
                    # uncomment for stricter filtering
                )
            )
        else:
            solutions = solutions.filter(
                (
                    pl.col(obj.symbol) <= options.fake_nadir[obj.symbol]
                    # & (pl.col(obj.symbol) >= options.fake_ideal[obj.symbol])
                    # uncomment for stricter filtering
                )
            )
    return GPRMResults(
        raw_results=res,
        solutions=solutions.select(var_cols),
        outputs=solutions.select(obj_cols),
    )


def get_representative_set(problem: Problem, options: GPRMOptions, results_list: list[GPRMResults]) -> GPRMResults:
    """Get the representative set according to the given MethodOptions.

    Switches between IPR and EMO based on the type of options given.

    Args:
        problem: DESDEO Problem object
        options: MethodOptions, either IPR_Options or EMOOptions
        results_list: list of previous MethodResults objects

    Returns:
        tuple: (DataFrame of evaluated points, MethodResults)

    Raises:
        TypeError: If the provided MethodOptions type is invalid.
    """
    if isinstance(options.method_options, IPR_Options):
        return get_representative_set_IPR(problem, options, results_list)
    if options.method_options is None:
        return get_representative_set_EMO(problem, options, results_list)
    raise TypeError("Invalid MethodOptions type provided.")

def setup(problem: Problem, options: FavOptions, results_list: list[FavResults]) -> FavOptions:
    """Setup function for favorite method.

    Args:
        problem: DESDEO Problem object
        options: FavOptions for the favorite method.
        results_list: List of previous FavResults.

    Returns:
        FavOptions: Updated options for the favorite method.
    """
    options = options.model_copy()
    winner = None

    orig_mps = options.original_most_preferred_solutions
    orig_mps_list = dict_of_rps_to_list_of_rps(orig_mps)
    fake_ideal, fake_nadir = agg_aspbounds(orig_mps_list, problem)
    # first iteration
    if not results_list:  # noqa:SIM102
        if isinstance(options.GPRMoptions.method_options, IPR_Options):
            options.GPRMoptions.method_options.most_preferred_solutions = orig_mps

    if results_list:  # not the first iteration
        if options.votes is None:
            raise ValueError("Votes must be provided for iterations after the first.")
        # handle voting
        old_candidates = results_list[-1].fair_solutions
        print(results_list)
        print(old_candidates)
        votes = options.votes
        winner = majority_rule(votes=votes)
        print("WINNER", winner)
        if winner is None:
            raise ValueError("No winner could be determined from the votes provided.")
        winner = old_candidates[winner]
        fake_nadir = results_list[-1].FavOptions.GPRMoptions.fake_nadir
        if fake_nadir is None:
            raise ValueError("Previous fake_nadir is None, cannot proceed with zooming.")
        fake_nadir = calculate_navigation_point(
            problem=problem,
            previous_navigation_point=fake_nadir,
            reachable_objective_vector=winner.objective_values,
            number_of_steps_remaining=options.zoom_options.num_steps_remaining,
        )  # TODO: Is this still needed? -> redefine fake_nadir based on winner. Note that fake_ideal stays the same.
        shifted_mps = shift_points(
            problem,
            most_preferred_solutions=orig_mps,
            group_preferred_solution=winner.objective_values,
            steps_remaining=options.zoom_options.num_steps_remaining,
        )
        if isinstance(options.GPRMoptions.method_options, IPR_Options):
            options.GPRMoptions.method_options.most_preferred_solutions = shifted_mps

    options.GPRMoptions.fake_ideal = fake_ideal
    options.GPRMoptions.fake_nadir = fake_nadir
    return options


def favorite_method(problem: Problem, options: FavOptions, results_list: list[FavResults]) -> FavResults:
    """Run one complete iteration of the favorite method.

    For multiple iterations, call this function multiple times, passing the previous results in results_list.
    Make note to change the votes in options for each iteration after the first.
    Also change options.zoom_options.num_steps_remaining accordingly.

    Args:
        problem: DESDEO Problem object
        options: FavOptions for the favorite method.
        results_list: List of previous FavResults. Can be None in the first iteration.

    Returns:
        FavResults: Results from this iteration of the favorite method. It also contains a filled up version of
        FavOptions (which includes, e.g., updated most preferred solutions and fake_nadir after zooming)
    """
    options = setup(problem, options, results_list)

    # Generate representative set
    gprm_results = get_representative_set(problem, options.GPRMoptions, [result.GPRMResults for result in results_list])

    fair_solutions = []
    # Add previous iteration's winner solution
    if results_list:
        # Get the candidates from the previous iteration
        previous_candidates = results_list[-1].fair_solutions

        # Note: setup() ensures options.votes is not None if results_list exists
        winner_idx = majority_rule(votes=options.votes)

        if winner_idx is not None:
            winner_solution = previous_candidates[winner_idx]
            fair_solutions.append(winner_solution)
        else:
            # Fallback if voting fails (though setup() usually catches this)
            raise ValueError("No winner could be determined from the votes provided.")

    targets = pl.DataFrame([point.targets for point in gprm_results.raw_results.evaluated_points])
    new_fair_solutions_list, _ = find_group_solutions(
        problem,
        solutions=gprm_results.outputs,
        targets=targets,
        most_preferred_solutions=options.original_most_preferred_solutions,
        fairness_criterion=options.candidate_generation_options
    )
    fair_solutions.extend(new_fair_solutions_list)

    return FavResults(
        FavOptions=options,
        GPRMResults=gprm_results,
        fair_solutions=fair_solutions,
    )


def hausdorff_candidates(all_points: list[_EvaluatedPoint], fair_solutions: list[FairSolution], n_of_candidates: int) -> list[FairSolution]:
    """
    Selects additional candidates using the Average Hausdorff Distance metric.

    This ensures the new candidates are diverse and representative of the clusters
    formed around the existing 'fair' solutions.

    Args:
        all_points: The full pool of evaluated points.
        fair_solutions: The existing selected candidates (seeds).
        n_of_candidates: How many new candidates to pick.

    Returns:
        list[FairSolution]: The list of candidates extended with new selections.
    """
    new_candidates = []

    obj_keys = all_points[0].objectives.keys()
    candidates_arr = np.array([
        [p.objectives[k] for k in obj_keys]
        for p in all_points
    ])
    # if We need to be on the reference plane
    # candidates_arr = np.array([
    #    [p.reference_point[k] for k in obj_keys]
    #    for p in all_points
    # ])

    # use fair solutions as the seeds
    seeds_arr = np.array([
        [s.objective_values[k] for k in obj_keys]
        for s in fair_solutions
    ])

    # min_dists[i] = distance from point i to the nearest existing seed
    dists = cdist(candidates_arr, seeds_arr, metric='euclidean')
    min_dists = np.min(dists, axis=1)

    # Track selected indices to avoid duplicates
    n_total = candidates_arr.shape[0]
    is_selected = np.zeros(n_total, dtype=bool)
    selected_indices = []

    for _ in range(n_of_candidates):
        best_idx = -1

        # avg hausdorff: Pick point that minimizes the sum of distances for everyone
        lowest_total_dist = float('inf')

        # Identify candidates (indices not yet selected)
        candidate_indices = np.where(~is_selected)[0]

        for idx in candidate_indices:
            cand_point = candidates_arr[idx].reshape(1, -1)

            # Distance from this specific candidate to everyone. Then, take min and sum the distances.
            dists_from_cand = cdist(candidates_arr, cand_point, metric='euclidean').flatten()
            potential_min_dists = np.minimum(min_dists, dists_from_cand)
            total_dist = np.sum(potential_min_dists)

            if total_dist < lowest_total_dist:
                lowest_total_dist = total_dist
                best_idx = idx

        #  Update State with the Winner
        if best_idx != -1:
            selected_indices.append(best_idx)
            is_selected[best_idx] = True

            # Permanently update min_dists for the next iteration
            winner_point = candidates_arr[best_idx].reshape(1, -1)
            dists_to_winner = cdist(candidates_arr, winner_point, metric='euclidean').flatten()
            min_dists = np.minimum(min_dists, dists_to_winner)

    #  TODO: as fairness value or criterion is not relevant here, consider using some other type.
    new_candidates = []
    for idx in selected_indices:
        point = all_points[idx]
        new_sol = FairSolution(
            objective_values=point.objectives,
            fairness_criterion="avg_hausdorff",
            fairness_value=0.0
        )
        new_candidates.append(new_sol)

    new_candidates = fair_solutions + new_candidates
    return new_candidates

def cluster_points(all_points: List[_EvaluatedPoint], centers: List[FairSolution]):
    """
    Assigns each point in `all_points` to the cluster of the nearest center (Voronoi partition).
    Returns the points, centres and cluster labels (integers) for every point.

    Args:
        all_points: List of evaluated points.
        centers: List of selected candidate solutions acting as cluster centers.

    Returns:
        tuple: (points_array, centers_array, labels_array)
    """
    obj_keys = all_points[0].objectives.keys()

    points_arr = np.array([[p.objectives[k] for k in obj_keys] for p in all_points])
    # if wanting to be in the preference space
    # points_arr = np.array([[p.reference_point[k] for k in obj_keys] for p in all_points])
    centers_arr = np.array([[c.objective_values[k] for k in obj_keys] for c in centers])

    dists = cdist(points_arr, centers_arr, metric='euclidean')
    # labels[i] = index of the center closest to point i
    labels = np.argmin(dists, axis=1)
    return points_arr, centers_arr, labels


def calculate_dist_to_hull(points_kminus: np.ndarray, hull: ConvexHull) -> np.ndarray:
    """
    TODO: check clanker explanation
    Calculates the algebraic distance from points to a Convex Hull.

    This is a fast vectorized approximation of distance.
    - Value > 0: Point is outside the hull. (Distance to nearest face plane).
    - Value <= 0: Point is inside the hull. (Distance to nearest face plane).

    The distance to the hull is determined by the plane the point is
    "most outside" of (the maximum positive value).
    If all values are negative, it is inside, and the max value represents
    how close it is to the boundary (least negative).

    # Source - https://stackoverflow.com/q/41000123
    # Posted by Woltan, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-02-11, License - CC BY-SA 3.0

    np.max(np.dot(self.equations[:, :-1], points.T).T + self.equations[:, -1], axis=-1)

    Args:
        points_kminus (np.ndarray): N x (k-1) array of points.
        hull (scipy.spatial.ConvexHull): The convex hull object.

    Returns:
        np.ndarray: Array of distances.
    """
    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]
    distances = np.max(np.dot(normals, points_kminus.T) + offsets[:, np.newaxis], axis=0)

    return distances

def expand_and_generate_candidates(
    winning_cluster_k: np.ndarray,
    all_points_k: np.ndarray,
    fraction_keep: float = 0.8,
    num_new_points: int = 1000
) -> np.ndarray:
    """
        Expands the region of interest around a winning cluster and generates new candidate solutions.

        This function implements the core expansion logic (Steps 1-5) of the Favorite Method:
        1. Projects (rotates) points from k-dimensional space to a (k-1)-dimensional hyperplane.
        2. Constructs a convex hull for the winning cluster and calculates the distance of all other points to this hull.
        3. Selects the top `fraction_keep` of points closest to the hull to form an expanded set.
        4. Constructs a new convex hull around the expanded set and generates uniform random points inside it.
        5. Projects (rotates) the new points back to the original k-dimensional objective space.

        Args:
            winning_cluster_k (np.ndarray): An array of shape (N, k) containing the points of the winning cluster in the objective space.
            all_points_k (np.ndarray): An array of shape (M, k) containing all available evaluated points in the objective space.
            fraction_keep (float, optional): The fraction (0.0 to 1.0) of points from `all_points_k` to include in the expanded region.
                Points are selected based on proximity to the winning cluster's hull. Defaults to 0.8.
            num_new_points (int, optional): The number of new candidate points to generate within the expanded convex hull. Defaults to 1000.

        Returns:
            np.ndarray: An array of shape (num_new_points, k) containing the new candidate points projected back into the k-dimensional objective space.
        """
    # 1. Rotate In
    cluster_kminus = rotate_in(winning_cluster_k)
    all_kminus = rotate_in(all_points_k)

    # 2. Calculate Hull of Winning Cluster
    if len(cluster_kminus) > cluster_kminus.shape[1]:
        win_hull = ConvexHull(cluster_kminus)
        dists = calculate_dist_to_hull(all_kminus, win_hull)
    else:
        # Fallback if cluster is too small (e.g., 1 point): Use distance to mean. Should not happen?
        center = np.mean(cluster_kminus, axis=0)
        dists = np.linalg.norm(all_kminus - center, axis=1)

    # 3. Top X% Closest
    n_keep = max(int(np.ceil(len(all_kminus) * fraction_keep)), cluster_kminus.shape[1] + 1)

    # Argsort gives indices of smallest distances first
    top_indices = np.argsort(dists)[:n_keep]
    expanded_set_kminus = all_kminus[top_indices]
    print(f"Expanded set: {len(expanded_set_kminus)} points selected.")

    # 4. Generate Random Points in Bounding Box using numba random gen
    # need to create the convex hull
    expanded_hull = ConvexHull(expanded_set_kminus, qhull_options='QJ')
    A_exp, b_exp = get_hull_equations(expanded_hull)
    # Bounding box: [min_coords, max_coords]
    bounding_box = np.array([
        np.min(expanded_set_kminus, axis=0),
        np.max(expanded_set_kminus, axis=0)
    ])
    new_points_kminus = numba_random_gen(num_new_points, bounding_box, A_exp, b_exp)

    # 5. Rotate Out (Project back to K-dims)
    new_points_k = rotate_out(new_points_kminus)

    return new_points_k


if __name__ == "__main__":

    print("Go run experiment/run_favorite.py")
