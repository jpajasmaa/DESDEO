"""Implementation of the FAVORITE interactive Group Decision Making method."""

import copy
import logging
import random
from itertools import product
from typing import Literal

import numpy as np
import polars as pl
import pydantic
from pydantic import ConfigDict, Field
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist

from desdeo.gdm.gdmtools import (
    alpha_fairness,
    get_top_n_fair_solutions,
    min_max_regret_no_impro,
    regret_allDMs_no_impro,
)
from desdeo.gdm.voting_rules import majority_rule
from desdeo.problem import (
    numpy_array_to_objective_dict,
    objective_dict_to_numpy_array,
)
from desdeo.problem.schema import Problem
from desdeo.tools import guess_best_solver, is_duplicate_solution
from desdeo.tools.generateReferencePoints import (
    generate_points,
    get_hull_equations,
    numba_random_gen,
    rotate_in,
    rotate_out,
)
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint, choose_reference_point
from desdeo.tools.scalarization import add_asf_diff, add_asf_nondiff

logger = logging.getLogger(__name__)

# --- Classes & Options ---


class IPR_Options(pydantic.BaseModel):  # noqa: N801
    """Options specific to iterative_pareto_representer applied with the Favorite method."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    num_initial_reference_points: int = Field(default=10000, ge=1)
    """The number of points to generate uniformly to represent the reference space."""
    version: Literal["convex_hull", "box"] = "convex_hull"
    """Sampling domain: 'convex_hull' within convex hull of points, or 'box' across bounding box."""
    most_preferred_solutions: dict[str, dict[str, float]] | None = None
    """Most preferred solutions of the decision makers. Should be filled in by code, not by user."""


class GPRMOptions(pydantic.BaseModel):
    """Pydantic model to contain options for the `get_representative_set` function."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    method_options: IPR_Options | None = Field(default_factory=IPR_Options)
    """Options specific to the selected method."""
    fake_ideal: dict[str, float] | None = None
    """Fake ideal point. Should be filled in by code, not by user."""
    fake_nadir: dict[str, float] | None = None
    """Fake nadir point. Should be filled in by code, not by user."""
    num_points_to_evaluate: int = Field(default=100, ge=1)
    """Number of points to evaluate in the IPR method."""


class IPR_Results(pydantic.BaseModel):  # noqa: N801
    """Results specific to iterative_pareto_representer applied with the Favorite method."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    evaluated_points: list[_EvaluatedPoint]
    """List of points evaluated by the solver along with their reference points and targets."""


class GPRMResults(pydantic.BaseModel):
    """Pydantic model to contain results from the `get_representative_set` function."""

    model_config = ConfigDict(use_attribute_docstrings=True, arbitrary_types_allowed=True)

    raw_results: IPR_Results
    """Raw results from the selected method."""
    solutions: pl.DataFrame | None
    """DataFrame containing the evaluated solutions (inputs)."""
    outputs: pl.DataFrame
    """DataFrame containing the evaluated outputs."""


class FairSolution(pydantic.BaseModel):
    """Represents a single solution identified as 'fair' or a 'candidate'."""

    objective_values: dict[str, float]
    """Objective values (e.g., {'f_1': 0.2})."""

    fairness_criterion: str
    """The criterion used to select this solution (e.g., 'mm', 'nash', 'avg_hausdorff')."""

    fairness_value: float
    """The numerical score associated with the fairness criterion, if applicable."""

    variable_values: dict[str, float | int] | None = None
    """Optional decision variable values (if available)."""


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
        """Options for generating candidate fair solutions.
        For now, just a string to determine the fairness criterion applied."""
        """ Support more options later."""
    )
    zoom_options: ZoomOptions = Field(default_factory=ZoomOptions)
    """Options for the zooming strategy. Support more options later."""
    original_most_preferred_solutions: dict[str, dict[str, float]]
    """Dictionary of the original most preferred solutions for each decision maker."""
    current_most_preferred_solutions: dict[str, dict[str, float]] | None = None
    """Dictionary of the active most preferred solutions (adapted across iterations)."""
    total_n_of_candidates: int = Field(default=5, ge=1)
    """The total number of candidate solutions to present to the DMs."""
    votes: dict[str, int] | None = None
    (
        """The votes for each decision maker's most preferred solution."""
        """ The candidates are from `fair_solutions` in FavResults of the previous iteration."""
        """Not required for the first iteration."""
    )
    tie_state: dict | None = None
    "Tracks possible revoting"
    preferences_already_adapted: bool = False
    """Flag indicating whether DM preferred solutions have already been adapted for this iteration."""


class FavResults(pydantic.BaseModel):
    """Pydantic model to contain results from one iteration of the favorite method."""

    FavOptions: FavOptions
    """Options used in this iteration of the Favorite method."""
    GPRMResults: GPRMResults
    """Results from the representative set method."""
    fair_solutions: list[FairSolution]
    """List of candidate fair solutions found in this iteration."""
    status: Literal["success", "revote_pending"] = "success"
    tie_state: dict | None = None
    "Tracks possible revoting"


# --- Core logic ---


class ProblemWrapper:
    """Wraps a DESDEO Problem to manage solving with IPR."""

    def __init__(self, problem: Problem, fake_ideal: dict[str, float], fake_nadir: dict[str, float]):
        """Initialize problem wrapper with ideal and nadir points.

        Args:
            problem: The DESDEO Problem.
            fake_ideal: The current fake ideal point.
            fake_nadir: The current fake nadir point.
        """
        self.problem = problem
        self.ideal, self.nadir = fake_ideal, fake_nadir
        self.problem = problem.update_ideal_and_nadir(new_ideal=self.ideal, new_nadir=self.nadir)
        self.solver_class = guess_best_solver(self.problem)
        self.evaluated_points: list[_EvaluatedPoint] = []

    def solve(self, scaled_refp: np.ndarray) -> list[_EvaluatedPoint]:
        """Solves a scalarized version of the problem using ASF.

        Args:
            scaled_refp: Reference point coordinates scaled between 0 and 1.

        Returns:
            list[_EvaluatedPoint]: The list of evaluated points including the new solution.
        """
        refp = {
            obj: val * (self.nadir[obj] - self.ideal[obj]) + self.ideal[obj]
            for obj, val in zip(self.ideal.keys(), scaled_refp, strict=True)
        }
        if self.problem.is_twice_differentiable:
            scaled_problem, target = add_asf_diff(self.problem, "target", refp)
        else:
            scaled_problem, target = add_asf_nondiff(self.problem, "target", refp)
        solver = self.solver_class(scaled_problem)
        results = solver.solve(target)

        objs = results.optimal_objectives
        scaled_objs = {obj: (objs[obj] - self.ideal[obj]) / (self.nadir[obj] - self.ideal[obj]) for obj in objs}
        self.evaluated_points.append(
            _EvaluatedPoint(
                reference_point=dict(zip(self.ideal.keys(), scaled_refp, strict=True)),
                targets=scaled_objs,
                objectives=objs,
            )
        )
        return self.evaluated_points


def find_group_solutions(
    problem: Problem,
    solutions: pl.DataFrame,
    targets: pl.DataFrame,
    most_preferred_solutions: dict[str, dict[str, float]],
    fairness_criterion: str,
) -> list[FairSolution]:
    """Identifies fair compromise solution(s) from a set of generated solutions based on a criterion.

    Currently, returns only one according to the fairness_criterion.

    Args:
        problem: The DESDEO problem object.
        solutions: DataFrame of evaluated solutions.
        targets: DataFrame of evaluated targets.
        most_preferred_solutions: The most preferred solutions of the DMs.
        fairness_criterion: The string identifier for the fairness criterion (e.g., 'utilitarian', 'nash', 'mm').

    Returns:
        list[FairSolution]: A list containing the FairSolution(s) found.
    """
    normalized_mpses = {}
    ideal, nadir = problem.get_ideal_point(), problem.get_nadir_point()
    for dm, mps in most_preferred_solutions.items():
        normalized_mpses[dm] = {obj: (mps[obj] - ideal[obj]) / (nadir[obj] - ideal[obj]) for obj in mps}

    # convert to numpy array for numba in UFs
    normalized_mpses_arr = []
    for _, dm in enumerate(normalized_mpses):
        normalized_mpses_arr.append(objective_dict_to_numpy_array(problem, normalized_mpses[dm]).tolist())

    ranking = None
    if fairness_criterion == "utilitarian":
        ranking = alpha_fairness(targets, normalized_mpses_arr, alpha=0.0)  # utilitarian
    elif fairness_criterion == "nash":
        ranking = alpha_fairness(targets, normalized_mpses_arr, alpha=1)  # nash
    elif fairness_criterion == "mm":
        ranking = min_max_regret_no_impro(targets, normalized_mpses_arr)  # minmax regret no improvements
    else:
        raise NotImplementedError("Given fairness criterion not implemented.")

    # convert to numpy array for get top fair solutions
    solutions_arr = solutions.to_numpy()
    ranking_r, ranking_i = get_top_n_fair_solutions(solutions_arr, ranking, 1)  # get the top fair

    fair_solutions_arr = []
    fair_solutions_arr.append(
        FairSolution(
            objective_values=numpy_array_to_objective_dict(problem, ranking_r[0]),
            fairness_criterion=fairness_criterion,
            fairness_value=float(ranking[ranking_i[0]]),
        )
    )
    return fair_solutions_arr


def get_representative_set_IPR(  # noqa: N802
    problem: Problem, options: GPRMOptions, results_list: list[GPRMResults]
) -> GPRMResults:
    """Generates a set of Pareto optimal solutions using the Iterative Pareto Representer (IPR).

    This method generates reference points in a specific region (hull or box) and solves
    scalarization problems for each.
    """
    if not isinstance(options.method_options, IPR_Options):
        raise TypeError("Expected IPR_Options for IPR method.")

    evaluated_points = [] if len(results_list) == 0 else results_list[-1].raw_results.evaluated_points

    # Normalize mps for fairness and IPR
    normalized_mpses = {}
    ideal, nadir = problem.get_ideal_point(), problem.get_nadir_point()
    for dm, mps in options.method_options.most_preferred_solutions.items():
        normalized_mpses[dm] = {obj: (mps[obj] - ideal[obj]) / (nadir[obj] - ideal[obj]) for obj in mps}

    # Reference points as array for methods to come
    rp_arr = []
    for _, dm in enumerate(normalized_mpses):
        rp_arr.append(objective_dict_to_numpy_array(problem, normalized_mpses[dm]).tolist())

    dims = len(problem.get_nadir_point())

    # Get the representative set according to the num points to evaluate
    for n in [options.num_points_to_evaluate, int(options.num_points_to_evaluate / 2), 10]:
        try:
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
                    logger.info("IPR run %d/%d", i + 1, num_runs)
                try:
                    reference_point, _ = choose_reference_point(refp, evaluated_points)
                except AssertionError as ae:
                    if "No reference points available" in str(ae):
                        logger.info("IPR: Reference points fully explored after %d evaluations.", len(evaluated_points))
                        break
                    raise
                evaluated_points = wrapped_problem.solve(reference_point)
            break
        except Exception as e:
            logger.warning("IPR error: %r", e)
            continue

    ipr_res = IPR_Results(evaluated_points=evaluated_points)

    return GPRMResults(
        raw_results=ipr_res,
        solutions=None,
        outputs=pl.DataFrame([point.objectives for point in evaluated_points]),
    )


def get_representative_set(problem: Problem, options: GPRMOptions, results_list: list[GPRMResults]) -> GPRMResults:
    """Get the representative set according to the given MethodOptions.

    Generates solutions using the Iterative Pareto Representer (IPR).

    Args:
        problem: DESDEO Problem object.
        options: GPRMOptions with IPR_Options.
        results_list: List of previous GPRMResults objects.

    Returns:
        GPRMResults: The generated representative set of solutions.

    Raises:
        TypeError: If the provided MethodOptions type is invalid.
    """
    if isinstance(options.method_options, IPR_Options):
        return get_representative_set_IPR(problem, options, results_list)
    raise TypeError("Invalid MethodOptions type provided. Expected IPR_Options.")


def get_tied_candidates(votes: dict[str, int]) -> list[int]:
    """Returns a list of candidate indices that are tied for the most votes.

    Args:
        votes: Dictionary mapping DM identifier to voted candidate index.

    Returns:
        list[int]: List of candidate indices with the highest vote count.
    """
    vote_counts: dict[int, int] = {}
    for v in votes.values():
        vote_counts[v] = vote_counts.get(v, 0) + 1

    max_votes = max(vote_counts.values()) if vote_counts else 0
    return [cand for cand, count in vote_counts.items() if count == max_votes]


def check_adjacency(pts_mat: np.ndarray, labels: np.ndarray, idx_a: int, idx_b: int) -> bool:
    """Checks if two clusters are geometrically adjacent based on Euclidean proximity.

    Args:
        pts_mat: N x k array of evaluated points in objective space.
        labels: 1D array of cluster indices mapping to each point.
        idx_a: Index of the first candidate/cluster.
        idx_b: Index of the second candidate/cluster.

    Returns:
        bool: True if the clusters are adjacent under the 1.5x internal distance heuristic.
    """
    pts_a = pts_mat[labels == idx_a]
    pts_b = pts_mat[labels == idx_b]

    if len(pts_a) == 0 or len(pts_b) == 0:
        return False

    dists = cdist(pts_a, pts_b, metric="euclidean")
    min_dist = np.min(dists)

    internal_dists = cdist(pts_a, pts_a, metric="euclidean")
    avg_internal_dist = np.mean(internal_dists) if len(internal_dists) > 0 else float("inf")

    return bool(min_dist < (avg_internal_dist * 1.5))


def random_tie_breaker(tied_indices: list[int], candidates: list[FairSolution]) -> tuple[FairSolution, int]:
    """Randomly selects a winner from the tied candidates.

    Args:
        tied_indices: List of candidate indices that are tied.
        candidates: List of candidate FairSolution objects.

    Returns:
        tuple[FairSolution, int]: The selected winning candidate and its index.
    """
    winner_idx = random.choice(tied_indices)  # noqa: S311
    return candidates[winner_idx], winner_idx


def handle_ties(
    problem: Problem,
    votes: dict[str, int],
    candidates: list[FairSolution],
    fav_results_previous: FavResults,
    tie_state: dict | None,
) -> tuple[FairSolution | None, dict | None]:
    """Evaluates a voting tie and routes it through the 3-step hierarchy.

    1. 2 regions Adjacent Check -> Average Projection
    2. more than 2 regions or Non-Adjacent -> Request Re-Vote
    3. Re-Vote Tied -> Random Fallback

    Returns:
        tuple: (winning_solution, updated_tie_state).
    """
    tied_indices = get_tied_candidates(votes)

    # Check Adjacency (Only applies if exactly 2 candidates tie)
    is_adjacent = False
    if len(tied_indices) == 2:  # noqa: PLR2004
        pts_mat, _, labels = cluster_points(fav_results_previous)
        is_adjacent = check_adjacency(pts_mat, labels, tied_indices[0], tied_indices[1])

    if is_adjacent:
        # Combine via Average Projection
        tied_votes = {dm: v for dm, v in votes.items() if v in tied_indices}
        compromise_solution = tie_breaker_avgproj(problem, tied_votes, candidates)
        return compromise_solution, None

    # Request re-vote among tied candidates
    if tie_state is None:
        # Sub-Route B1: First tie -> Trigger Re-Vote UI
        new_tie_state = {"tied_indices": tied_indices, "strategy": "Simple Vote-Again"}
        return None, new_tie_state

    # Re-Vote tied again -> Random Fallback
    winner_candidate, _ = random_tie_breaker(tied_indices, candidates)
    return winner_candidate, None


def setup(
    problem: Problem, options: FavOptions, results_list: list[FavResults]
) -> tuple[FavOptions, FairSolution | None, dict | None]:
    """Setup function for Favorite method.

    Args:
        problem: DESDEO Problem object
        options: FavOptions for the Favorite method.
        results_list: List of previous FavResults.

    Returns:
        FavOptions: Updated options for the Favorite method.
    """
    options = options.model_copy()
    winner_solution = None
    new_tie_state = None

    orig_mps = options.original_most_preferred_solutions
    if not options.current_most_preferred_solutions:
        options.current_most_preferred_solutions = copy.deepcopy(orig_mps)

    fake_ideal, fake_nadir = problem.get_ideal_point(), problem.get_nadir_point()
    # first iteration
    if not results_list:
        if isinstance(options.GPRMoptions.method_options, IPR_Options):
            options.GPRMoptions.method_options.most_preferred_solutions = orig_mps
    else:
        if options.votes is None:
            raise ValueError("Votes must be provided for iterations after the first.")
        previous_results = results_list[-1]
        old_candidates = previous_results.fair_solutions

        # Adapt DM preferred solutions if any DM voted for a non-optimal candidate
        if not options.preferences_already_adapted:
            adapted_mps, _ = adapt_all_dm_preferences(
                problem=problem,
                current_mps=options.current_most_preferred_solutions,
                candidates=old_candidates,
                votes=options.votes,
            )
            options.current_most_preferred_solutions = adapted_mps
            options.preferences_already_adapted = True

        # Determine Winner using Majority Rule else Tie-Breaker
        winner_idx = majority_rule(votes=options.votes)
        if winner_idx is not None:
            winner_solution = old_candidates[winner_idx]
        else:
            winner_solution, new_tie_state = handle_ties(
                problem=problem,
                votes=options.votes,
                candidates=old_candidates,
                fav_results_previous=previous_results,
                tie_state=options.tie_state,
            )

        fake_nadir = previous_results.FavOptions.GPRMoptions.fake_nadir
    options.GPRMoptions.fake_ideal = fake_ideal
    options.GPRMoptions.fake_nadir = fake_nadir

    return options, winner_solution, new_tie_state


def favorite_method(problem: Problem, options: FavOptions, results_list: list[FavResults]) -> FavResults:
    """Run one complete iteration of the Favorite method.

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
    options, winner_solution, new_tie_state = setup(problem, options, results_list)

    # check for re-vote
    if new_tie_state is not None:
        return FavResults(
            FavOptions=options,
            GPRMResults=results_list[-1].GPRMResults,
            fair_solutions=results_list[-1].fair_solutions,
            status="revote_pending",
            tie_state=new_tie_state,
        )

    # Generate representative set
    gprm_results = get_representative_set(problem, options.GPRMoptions, [result.GPRMResults for result in results_list])

    targets = pl.DataFrame([point.targets for point in gprm_results.raw_results.evaluated_points])
    active_mps = options.current_most_preferred_solutions or options.original_most_preferred_solutions
    new_fair_solutions_list = find_group_solutions(
        problem,
        solutions=gprm_results.outputs,
        targets=targets,
        most_preferred_solutions=active_mps,
        fairness_criterion=options.candidate_generation_options,
    )

    fair_solutions = []
    # Add previous iteration's winner solution
    if winner_solution is not None:
        if new_fair_solutions_list and is_duplicate_solution(
            winner_solution, new_fair_solutions_list[0], check_variables=True, problem=problem
        ):
            # Previous winner is also top group fair solution in this iteration
            winner_solution.fairness_criterion = f"winner_and_{options.candidate_generation_options}"
            winner_solution.fairness_value = new_fair_solutions_list[0].fairness_value
            fair_solutions = [winner_solution]
        else:
            fair_solutions = [winner_solution, *new_fair_solutions_list]
    else:
        fair_solutions = list(new_fair_solutions_list)

    # Generate Hausdorff Candidates
    all_points = gprm_results.raw_results.evaluated_points
    n_missing = options.total_n_of_candidates - len(fair_solutions)

    if n_missing > 0:
        # hausdorff_candidates returns the existing fairs + the new ones
        fair_solutions = hausdorff_candidates(all_points, fair_solutions, n_missing)

    return FavResults(
        FavOptions=options, GPRMResults=gprm_results, fair_solutions=fair_solutions, status="success", tie_state=None
    )


def hausdorff_candidates(
    all_points: list[_EvaluatedPoint], fair_solutions: list[FairSolution], n_of_candidates: int
) -> list[FairSolution]:
    """Selects additional candidates using the Modified Hausdorff Distance metric.

    This ensures the new candidates are diverse and representative of the clusters
    formed around the existing 'fair' solutions.

    Args:
        all_points: The full pool of evaluated points.
        fair_solutions: The existing selected candidates (seeds).
        n_of_candidates: How many new candidates to pick.

    Returns:
        list[FairSolution]: The list of candidates extended with new selections.
    """
    obj_keys = all_points[0].objectives.keys()
    candidates_arr = np.array([[p.objectives[k] for k in obj_keys] for p in all_points])

    # use fair solutions as the seeds
    seeds_arr = np.array([[s.objective_values[k] for k in obj_keys] for s in fair_solutions])

    # min_dists[i] = distance from point i to the nearest existing seed
    dists = cdist(candidates_arr, seeds_arr, metric="euclidean")
    min_dists = np.min(dists, axis=1)

    # Track selected indices to avoid duplicates
    n_total = candidates_arr.shape[0]
    is_selected = np.zeros(n_total, dtype=bool)
    selected_indices = []

    for _ in range(n_of_candidates):
        best_idx = -1

        # mod hausdorff: Pick point that minimizes the sum of distances for everyone
        lowest_total_dist = float("inf")

        # Identify candidates (indices not yet selected)
        candidate_indices = np.where(~is_selected)[0]

        for idx in candidate_indices:
            cand_point = candidates_arr[idx].reshape(1, -1)

            # Distance from this specific candidate to everyone. Then, take min and sum the distances.
            dists_from_cand = cdist(candidates_arr, cand_point, metric="euclidean").flatten()
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
            dists_to_winner = cdist(candidates_arr, winner_point, metric="euclidean").flatten()
            min_dists = np.minimum(min_dists, dists_to_winner)

    #  TODO: as fairness value or criterion is not relevant here, consider using some other type.
    new_candidates = []
    for idx in selected_indices:
        point = all_points[idx]
        new_sol = FairSolution(
            objective_values=point.objectives, fairness_criterion="avg_hausdorff", fairness_value=1e6
        )
        new_candidates.append(new_sol)

    return fair_solutions + new_candidates


def cluster_points(fav_results: FavResults) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assigns each point in evaluated_points to the cluster of the nearest candidate (Voronoi partition).

    Returns the points, centres and cluster labels (integers) for every point.

    Args:
        fav_results: FavResults

    Returns:
        tuple: (points_array, centers_array, labels_array)
    """
    all_points = fav_results.GPRMResults.raw_results.evaluated_points
    candidates = fav_results.fair_solutions
    obj_keys = all_points[0].objectives.keys()

    points_arr = np.array([[p.objectives[k] for k in obj_keys] for p in all_points])
    candidates_arr = np.array([[c.objective_values[k] for k in obj_keys] for c in candidates])

    dists = cdist(points_arr, candidates_arr, metric="euclidean")
    # labels[i] = index of the center closest to point i
    labels = np.argmin(dists, axis=1)
    return points_arr, candidates_arr, labels


def recluster_for_tie_breaker(
    all_points: list[_EvaluatedPoint], existing_candidates: list[FairSolution], compromise_solution: FairSolution
) -> tuple[list[FairSolution], np.ndarray, int]:
    """Re-calculates the Voronoi partitions (clusters) when a tie-breaker introduces a brand new compromise solution."""
    # The compromise becomes our primary geometric seed
    updated_candidates = [compromise_solution]

    # Retrieve the pure mathematical Fair Solution (e.g., maxmin).
    # Assuming index 1 is the group fair solution from your generation logic.
    if len(existing_candidates) > 1:
        updated_candidates.append(existing_candidates[1])

    # The compromise is now exactly at the start of the list
    winning_idx = 0

    # ==========================================
    # Extract arrays for distance calculation
    # ==========================================
    obj_keys = all_points[0].objectives.keys()
    points_arr = np.array([[p.objectives[k] for k in obj_keys] for p in all_points])
    candidates_arr = np.array([[c.objective_values[k] for k in obj_keys] for c in updated_candidates])

    # ==========================================
    # Re-calculate the clusters (Voronoi Partition)
    # ==========================================
    dists = cdist(points_arr, candidates_arr, metric="euclidean")
    new_labels = np.argmin(dists, axis=1)

    return updated_candidates, new_labels, winning_idx


def calculate_dist_to_hull(points_kminus: np.ndarray, hull: ConvexHull) -> np.ndarray:
    """Calculates the algebraic distance from points to a Convex Hull.

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
    return np.max(np.dot(normals, points_kminus.T) + offsets[:, np.newaxis], axis=0)


def expand_and_generate_candidates(
    winning_cluster_k: np.ndarray, all_points_k: np.ndarray, fraction_keep: float = 0.8, num_new_points: int = 1000
) -> np.ndarray:
    """Expands the region of interest around a winning cluster and generates new candidate solutions.

    This function implements the core expansion logic (Steps 1-5) of the Favorite Method:
    1. Projects (rotates) points from k-dimensional space to a (k-1)-dimensional hyperplane.
    2. Constructs a convex hull for the winning cluster and calculates the distance of all other points to this hull.
    3. Selects the top `fraction_keep` of points closest to the hull to form an expanded set.
    4. Constructs a new convex hull around the expanded set and generates uniform random points inside it.
    5. Projects (rotates) the new points back to the original k-dimensional objective space.

    Args:
        winning_cluster_k (np.ndarray): An array of shape (N, k) containing the points of the winning cluster
        in the objective space.
        all_points_k (np.ndarray): An array of shape (M, k) containing all available evaluated points
        in the objective space.
        fraction_keep (float, optional): The fraction (0.0 to 1.0) of points from `all_points_k` to include
        in the expanded region.
            Points are selected based on proximity to the winning cluster's hull. Defaults to 0.8.
        num_new_points (int, optional): The number of new candidate points to generate within
        the expanded convex hull. Defaults to 1000.

    Returns:
        np.ndarray: An array of shape (num_new_points, k) containing the new candidate points projected back
        into the k-dimensional objective space.
    """
    # Rotate In
    cluster_kminus = rotate_in(winning_cluster_k)
    all_kminus = rotate_in(all_points_k)

    # Calculate Hull of Winning Cluster
    if len(cluster_kminus) > cluster_kminus.shape[1]:
        try:
            win_hull = ConvexHull(cluster_kminus)
            dists = calculate_dist_to_hull(all_kminus, win_hull)
        except (QhullError, ValueError):
            dists = np.min(cdist(all_kminus, cluster_kminus), axis=1)
    else:
        # Fallback if cluster has fewer points than dimension + 1 (e.g., 1 or 2 points):
        # calculate minimum Euclidean distance to cluster points
        dists = np.min(cdist(all_kminus, cluster_kminus), axis=1)

    # How many solutions to keep, at least as many as winning_cluster columns + 1
    n_keep = max(int(np.ceil(len(all_kminus) * fraction_keep)), cluster_kminus.shape[1] + 1)

    # Argsort gives indices of smallest distances first
    top_indices = np.argsort(dists)[:n_keep]
    expanded_set_kminus = all_kminus[top_indices]
    logger.info("Expanded set: %d points selected.", len(expanded_set_kminus))

    # Generate Random Points in Bounding Box using numba random gen
    try:
        expanded_hull = ConvexHull(expanded_set_kminus, qhull_options="QJ")
    except (QhullError, ValueError):
        mins = np.min(expanded_set_kminus, axis=0)
        maxs = np.max(expanded_set_kminus, axis=0)
        diff = maxs - mins
        degenerate_tol = 1e-6
        padding = 1e-4
        maxs = np.where(diff < degenerate_tol, maxs + padding, maxs)
        mins = np.where(diff < degenerate_tol, mins - padding, mins)
        corners = np.array(list(product(*zip(mins, maxs, strict=True))))
        expanded_hull = ConvexHull(np.vstack([expanded_set_kminus, corners]))

    a_exp, b_exp = get_hull_equations(expanded_hull)
    # Bounding box: [min_coords, max_coords]
    bounding_box = np.array([np.min(expanded_set_kminus, axis=0), np.max(expanded_set_kminus, axis=0)])
    new_points_kminus = numba_random_gen(num_new_points, bounding_box, a_exp, b_exp)

    # return new points in k, Rotate Out (Project back to K-dims)
    return rotate_out(new_points_kminus)


def generate_next_iteration_mps(
    fav_results: FavResults,
    cluster_labels: np.ndarray,
    winning_idx: int,
    fraction_to_keep: float = 0.8,
    num_new_points: int = 1000,
) -> dict[str, dict[str, float]]:
    """Clusters points, expands convex hull in reference space, and returns MPS dict for next iteration.

    Args:
        fav_results: FavResults from previous iteration.
        cluster_labels: Array mapping points to clusters.
        winning_idx: Index of the winning cluster.
        fraction_to_keep: Fraction of points to keep in expanded hull.
        num_new_points: Number of reference points to generate.

    Returns:
        dict[str, dict[str, float]]: next_iter_mps_dict
    """
    all_points = fav_results.GPRMResults.raw_results.evaluated_points
    obj_names = list(fav_results.FavOptions.GPRMoptions.fake_ideal.keys())

    ref_matrix = np.array([[p.reference_point[k] for k in obj_names] for p in all_points])

    winning_refs = ref_matrix[cluster_labels == winning_idx]

    # Expand the hull in the reference space
    new_candidates_scaled = expand_and_generate_candidates(
        winning_cluster_k=winning_refs,
        all_points_k=ref_matrix,
        fraction_keep=fraction_to_keep,
        num_new_points=num_new_points,
    )

    # Scale candidates back to the Objective Space
    fake_ideal_arr = np.array([fav_results.FavOptions.GPRMoptions.fake_ideal[k] for k in obj_names])
    fake_nadir_arr = np.array([fav_results.FavOptions.GPRMoptions.fake_nadir[k] for k in obj_names])

    new_candidates_obj = new_candidates_scaled * (fake_nadir_arr - fake_ideal_arr) + fake_ideal_arr

    # Build the dictionary for IPR Options
    next_iter_mps = {}
    for i, point in enumerate(new_candidates_obj):
        point_dict = dict(zip(obj_names, point, strict=True))
        next_iter_mps[f"gen_{i}"] = point_dict

    return next_iter_mps


def select_final_candidates(
    problem: Problem, fav_results: FavResults, cluster_labels: np.ndarray, winning_idx: int, n_candidates: int = 5
) -> list[FairSolution]:
    """Selects final candidates by keeping the winning solution as the core candidate.

    Uses Modified Hausdorff distance to pick the remaining candidates strictly from within
    the winning cluster.

    Args:
        problem: DESDEO Problem
        fav_results: The results from the final iteration.
        cluster_labels: Array mapping evaluated points to cluster indices.
        winning_idx: The index of the cluster chosen by the DMs.
        n_candidates: Total number of final candidates to generate (default 5).

    Returns:
        list[FairSolution]: The list of final solutions to present to the DMs.
    """
    all_points = fav_results.GPRMResults.raw_results.evaluated_points

    # Isolate the "Winning Region"
    winning_points = [p for i, p in enumerate(all_points) if cluster_labels[i] == winning_idx]

    # Identify the Winning Candidate
    core_candidate = fav_results.fair_solutions[winning_idx]
    core_candidate.fairness_criterion = "last_winner"

    # TODO:
    # we want the FairSolution to be included here?
    # 3. NEW: Compute the fair group solution from inside this winning cluster
    # Reconstruct the solutions (outputs) and targets dataframes for the winning region
    winning_outputs_df = pl.DataFrame([p.objectives for p in winning_points])
    winning_targets_df = pl.DataFrame([p.targets for p in winning_points])

    active_mps = (
        fav_results.FavOptions.current_most_preferred_solutions
        or fav_results.FavOptions.original_most_preferred_solutions
    )
    fair_group_list = find_group_solutions(
        problem=problem,
        solutions=winning_outputs_df,
        targets=winning_targets_df,
        most_preferred_solutions=active_mps,
        fairness_criterion=fav_results.FavOptions.candidate_generation_options,
    )

    # Extract the solution and brand its criterion clearly
    fair_cluster_candidate = fair_group_list[0]
    fair_cluster_candidate.fairness_criterion = f"final_{fav_results.FavOptions.candidate_generation_options}"

    # Check if the winning candidate is also the top-ranked group-fair solution in both objective and decision space
    if is_duplicate_solution(core_candidate, fair_cluster_candidate, check_variables=True, problem=problem):
        core_candidate.fairness_criterion = f"winner_and_{fav_results.FavOptions.candidate_generation_options}"
        core_candidate.fairness_value = fair_cluster_candidate.fairness_value
        final_solutions = [core_candidate]
    else:
        final_solutions = [core_candidate, fair_cluster_candidate]

    n_missing = n_candidates - len(final_solutions)
    # Safety catch: just in case the cluster is unusually small
    n_missing = min(n_missing, len(winning_points))

    if n_missing > 0:
        n_seeds = len(final_solutions)
        final_solutions = hausdorff_candidates(
            all_points=winning_points, fair_solutions=final_solutions, n_of_candidates=n_missing
        )
        # Make sure any newly added elements have their tags explicitly overwritten
        for i in range(len(final_solutions)):
            if i >= n_seeds:
                final_solutions[i].fairness_criterion = "final_hausdorff"

    return final_solutions


def project_point_to_pareto_front(problem: Problem, point: dict[str, float]) -> dict[str, float]:
    """Projects an aspiration or reference point onto the Pareto front using Achievement Scalarizing Function (ASF).

    Args:
        problem: DESDEO Problem object.
        point: Dictionary of objective values representing the reference/aspiration point.

    Returns:
        dict[str, float]: The projected Pareto optimal objective values.
    """
    if problem.is_twice_differentiable:
        scaled_problem, target = add_asf_diff(problem, "target", point)
    else:
        scaled_problem, target = add_asf_nondiff(problem, "target", point)

    solver_class = guess_best_solver(scaled_problem)
    solver = solver_class(scaled_problem)
    results = solver.solve(target)
    return results.optimal_objectives


def calculate_dm_utility(
    problem: Problem,
    dm_mps: dict[str, float],
    candidate_objectives: dict[str, float],
) -> float:
    """Calculates the utility of a candidate solution for a single DM using normalized one-sided regret.

    Uses regret_allDMs_no_impro with ideal scaled to 0 and nadir scaled to 1.
    Higher value represents higher utility for the DM.

    Args:
        problem: DESDEO Problem object.
        dm_mps: Current most preferred solution for the DM.
        candidate_objectives: Objective values of the candidate.

    Returns:
        float: Utility value.
    """
    ideal = problem.get_ideal_point()
    nadir = problem.get_nadir_point()
    obj_keys = list(candidate_objectives.keys())

    # Map to [0, 1] minimization space
    sol_norm = np.array([(candidate_objectives[k] - ideal[k]) / (nadir[k] - ideal[k]) for k in obj_keys])
    mps_norm = np.array([(dm_mps[k] - ideal[k]) / (nadir[k] - ideal[k]) for k in obj_keys])

    utilities = regret_allDMs_no_impro(sol_norm, [mps_norm])
    return float(utilities[0])


def minimum_adjustment_mps(
    problem: Problem,
    dm_mps: dict[str, float],
    voted_candidate: dict[str, float],
    all_candidates: list[dict[str, float]],
    epsilon: float = 1e-4,
    bisection_steps: int = 20,
) -> tuple[dict[str, float], bool, float]:
    """Updates a DM's preferred solution via minimum adjustment towards the voted candidate.

    Note on Epsilon Margin:
    - If the voted candidate's utility is already within epsilon of the maximum candidate utility
      (u_voted >= max_other - epsilon), no adjustment is made and the DM's existing preferred solution
      is kept in play.
    - If adjustment is needed, binary search over lambda in [0, 1] on the line segment
      z(lambda) = (1 - lambda) * dm_mps + lambda * voted_candidate finds the smallest lambda where
      the projected point onto the Pareto front makes the voted candidate achieve strictly highest utility:
      u(c_voted; z_proj) >= max_other(z_proj) + epsilon.

    Args:
        problem: DESDEO Problem object.
        dm_mps: Current preferred solution of the DM.
        voted_candidate: Objective values of the candidate the DM voted for.
        all_candidates: List of objective values of all candidates presented in this iteration.
        epsilon: Margin threshold to keep original in play and ensure strict preference after adjustment.
        bisection_steps: Number of bisection iterations.

    Returns:
        tuple: (new_mps, was_adjusted, lambda_star)
    """
    # 1. Consistency check: Is voted_candidate already the best candidate under current dm_mps?
    u_voted_initial = calculate_dm_utility(problem, dm_mps, voted_candidate)
    competing_candidates = [
        c for c in all_candidates if not all(np.isclose(c[k], voted_candidate[k], atol=1e-7) for k in voted_candidate)
    ]

    if not competing_candidates:
        return dm_mps, False, 0.0

    max_other_initial = max(calculate_dm_utility(problem, dm_mps, c) for c in competing_candidates)

    # If already the best candidate within epsilon margin, keep original in play
    if u_voted_initial >= max_other_initial - epsilon:
        return dm_mps, False, 0.0

    # 2. Binary search over lambda in [0, 1]
    low = 0.0
    high = 1.0
    best_proj = None

    for _ in range(bisection_steps):
        if (high - low) < epsilon:
            break
        mid = (low + high) / 2.0
        z_mid = {k: (1.0 - mid) * dm_mps[k] + mid * voted_candidate[k] for k in dm_mps}
        z_proj = project_point_to_pareto_front(problem, z_mid)

        u_v = calculate_dm_utility(problem, z_proj, voted_candidate)
        u_max_others = max(calculate_dm_utility(problem, z_proj, c) for c in competing_candidates)

        if u_v >= u_max_others + epsilon:
            high = mid
            best_proj = z_proj
        else:
            low = mid

    if best_proj is None:
        best_proj = project_point_to_pareto_front(problem, voted_candidate)
        return best_proj, True, 1.0

    return best_proj, True, float(high)


def adapt_all_dm_preferences(
    problem: Problem,
    current_mps: dict[str, dict[str, float]],
    candidates: list[FairSolution],
    votes: dict[str, int],
    epsilon: float = 1e-4,
) -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
    """Adapts the preferred solutions of all DMs based on their cast votes and candidates.

    Args:
        problem: DESDEO Problem object.
        current_mps: Current most preferred solutions for each DM.
        candidates: List of FairSolution candidates presented to the DMs.
        votes: Map of DM ID to candidate index voted for.
        epsilon: Margin threshold.

    Returns:
        tuple: (updated_mps, adjustments_summary)
    """
    updated_mps = {}
    adjustments_summary = {}
    all_candidate_objs = [c.objective_values for c in candidates]

    for dm_id, vote_idx in votes.items():
        if dm_id not in current_mps:
            continue

        dm_current_pref = current_mps[dm_id]
        if vote_idx < 0 or vote_idx >= len(candidates):
            updated_mps[dm_id] = dm_current_pref
            continue

        voted_obj = candidates[vote_idx].objective_values
        new_pref, was_adjusted, lam = minimum_adjustment_mps(
            problem=problem,
            dm_mps=dm_current_pref,
            voted_candidate=voted_obj,
            all_candidates=all_candidate_objs,
            epsilon=epsilon,
        )

        updated_mps[dm_id] = new_pref
        adjustments_summary[dm_id] = {
            "was_adjusted": was_adjusted,
            "adjusted": was_adjusted,
            "lambda": lam,
            "lambda_shift": lam,
            "voted_candidate_idx": vote_idx,
            "previous_mps": dm_current_pref,
            "new_mps": new_pref,
        }

    # Retain any DMs who did not vote without changes
    for dm_id, dm_current_pref in current_mps.items():
        if dm_id not in updated_mps:
            updated_mps[dm_id] = dm_current_pref

    return updated_mps, adjustments_summary


def tie_breaker_avgproj(problem: Problem, votes: dict[str, int], candidates: list[FairSolution]) -> FairSolution:
    """Resolves a voting tie by averaging the objective values of all voted candidates.

    Projects that average point onto the Pareto front using ASF.

    Args:
        problem: DESDEO Problem object
        votes: dictionary of the votes from the DMs.
        candidates: List of candidates as FairSolutions.

    Returns:
        FairSolution: A candidate according to the tie-breaker.
    """
    obj_names = list(candidates[0].objective_values.keys())
    n_voters = len(votes)

    # Calculate the average objective vector based on the votes
    avg_point = dict.fromkeys(obj_names, 0.0)

    for _, vote_idx in votes.items():
        voted_candidate = candidates[vote_idx]
        for obj in obj_names:
            avg_point[obj] += voted_candidate.objective_values[obj]

    for obj in obj_names:
        avg_point[obj] /= n_voters

    logger.info("Tie detected. Calculated Average Reference Point: %s", avg_point)

    # Project the average point to the Pareto front using ASF
    projected_objectives = project_point_to_pareto_front(problem, avg_point)

    # Return the new projected solution as the winning FairSolution
    return FairSolution(
        objective_values=projected_objectives, fairness_criterion="tie_breaker_average_projection", fairness_value=0.0
    )


def calculate_fraction_to_keep(current_iter: int, max_iters: int, num_objectives: int) -> float:
    """Calculates the fraction of points to retain in the Convex Hull for the next iteration.

    This geometrically shrinks the search space volume. Because the hull is calculated
    on a (k-1)-dimensional hyperplane, the linear decay of the radius is raised to
    the power of (k-1).

    Args:
        current_iter: The current iteration index (0-indexed).
        max_iters: Total number of zooming iterations planned.
        num_objectives: The number of objectives (k) in the problem.

    Returns:
        float: The fraction of points to keep (between 0.0 and 1.0).
    """
    if current_iter >= max_iters - 1:
        return 0.0
    remaining_steps = max_iters - current_iter
    power = num_objectives - 1
    # ratio: ((R - 1) / R) ^ (k-1)
    fraction = ((remaining_steps - 1) / remaining_steps) ** power

    return float(fraction)
