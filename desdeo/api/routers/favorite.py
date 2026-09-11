"""FastAPI router for the FAVORITE group decision making method."""

import copy
import uuid
from typing import Annotated

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from desdeo.api.db import engine
from desdeo.api.models import ProblemDB
from desdeo.api.models.gdm.favorite import (
    FavoriteInitRequest,
    FavoriteSessionDB,
    FavoriteSessionState,
    FavoriteVoteRequest,
)
from desdeo.gdm.favorite_method import (
    FairSolution,
    FavOptions,
    GPRMOptions,
    IPR_Options,
    ZoomOptions,
    adapt_all_dm_preferences,
    calculate_fraction_to_keep,
    check_adjacency,
    cluster_points,
    favorite_method,
    generate_next_iteration_mps,
    recluster_for_tie_breaker,
    tie_breaker_avgproj,
)
from desdeo.problem.schema import Problem
from desdeo.problem.testproblems import (
    dmitry_forest_problem_disc,
    dtlz2,
    metallurgical_application_discrete,
    re34,
    river_pollution_problem_discrete,
)

router = APIRouter(prefix="/favorite", tags=["Favorite Method"])


def _tally_votes(current_votes: dict[str, int]) -> tuple[list[int], int]:
    """Tally votes and return top candidates and max vote count."""
    vote_counts = {v: list(current_votes.values()).count(v) for v in set(current_votes.values())}
    max_votes = max(vote_counts.values()) if vote_counts else 0
    top_candidates = [cand for cand, count in vote_counts.items() if count == max_votes]
    return top_candidates, max_votes


def _serialize_state(state: FavoriteSessionState) -> dict:
    """Serialize FavoriteSessionState to a JSON-compatible dictionary."""
    raw_dict = state.model_dump(mode="python")
    for result in raw_dict.get("results_history", []):
        gprm = result.get("GPRMResults", {})
        if isinstance(gprm.get("solutions"), pl.DataFrame):
            gprm["solutions"] = gprm["solutions"].to_dicts()
        if isinstance(gprm.get("outputs"), pl.DataFrame):
            gprm["outputs"] = gprm["outputs"].to_dicts()
    return raw_dict


def _deserialize_state(raw_dict: dict) -> FavoriteSessionState:
    """Deserialize dictionary into FavoriteSessionState with polars DataFrames."""
    safe_dict = copy.deepcopy(raw_dict)
    for result in safe_dict.get("results_history", []):
        gprm = result.get("GPRMResults", {})
        if gprm.get("solutions") is not None and isinstance(gprm["solutions"], list):
            gprm["solutions"] = pl.DataFrame(gprm["solutions"])
        if gprm.get("outputs") is not None and isinstance(gprm["outputs"], list):
            gprm["outputs"] = pl.DataFrame(gprm["outputs"])
    return FavoriteSessionState(**safe_dict)


def get_session():
    """Yield database session dependency."""
    with Session(engine) as session:
        yield session


def fetch_dm_preferences_from_db(
    dm_ids: list[str],
    problem: Problem,
) -> dict[str, dict[str, float]]:
    """Fetches or generates realistic starting preferences for each DM."""
    prob_name = problem.name or ""
    if prob_name == "Discrete River Pollution (4 objectives)" or "river" in prob_name.lower():
        default_mps = {
            "dm1": {"f1": 5.9066, "f2": 3.2894, "f3": 6.5792, "f4": -4.5460},
            "dm2": {"f1": 5.4290, "f2": 3.0121, "f3": 7.2395, "f4": -0.9135},
            "dm3": {"f1": 6.1623, "f2": 2.8839, "f3": 5.2575, "f4": -0.0045},
            "dm4": {"f1": 5.8790, "f2": 3.3617, "f3": 6.6493, "f4": -6.6835},
        }
        return {
            dm_id: default_mps.get(
                dm_id,
                {
                    "f1": 5.80 + (idx * 0.1),
                    "f2": 3.10 + (idx * 0.05),
                    "f3": 6.00 + (idx * 0.2),
                    "f4": -3.00 + (idx * 0.5),
                },
            )
            for idx, dm_id in enumerate(dm_ids)
        }

    if prob_name == "Dmitry Forest Problem (Discrete)" or "forest" in prob_name.lower():
        dmitry_mps = {
            "dm1": {"Rev": 249.5904, "HA": 12497.6850, "Carb": 2880.2038, "DW": 96.9443},
            "dm2": {"Rev": 141.1089, "HA": 20224.8348, "Carb": 3952.1429, "DW": 211.6469},
            "dm3": {"Rev": 86.2129, "HA": 18288.0717, "Carb": 4448.7892, "DW": 206.2755},
            "dm4": {"Rev": 232.3387, "HA": 18328.0238, "Carb": 3347.8052, "DW": 186.4134},
        }
        ideal = problem.get_ideal_point()
        nadir = problem.get_nadir_point()
        obj_keys = list(ideal.keys())
        return {
            dm_id: dmitry_mps.get(
                dm_id,
                {k: float(ideal[k] + (nadir[k] - ideal[k]) * (0.2 * (idx + 1))) for k in obj_keys},
            )
            for idx, dm_id in enumerate(dm_ids)
        }

    if "metall" in prob_name.lower():
        metall_mps = {
            "dm1": {"YS": 796.16, "UTS": 870.8688, "ELON": 19.5884, "CE": 0.3452, "COST": 6.4521},
            "dm2": {"YS": 598.4819, "UTS": 1898.2363, "ELON": 22.892, "CE": 1.2768, "COST": 180.4054},
            "dm3": {"YS": 557.6385, "UTS": 576.9771, "ELON": 41.1, "CE": 0.3058, "COST": 3.3261},
            "dm4": {"YS": 638.744, "UTS": 612.2479, "ELON": 29.9485, "CE": 0.1782, "COST": 0.922},
            "dm5": {"YS": 652.8835, "UTS": 1061.3929, "ELON": 30.7727, "CE": 0.5317, "COST": 7.108},
        }
        ideal = problem.get_ideal_point()
        nadir = problem.get_nadir_point()
        obj_keys = list(ideal.keys())
        return {
            dm_id: metall_mps.get(
                dm_id,
                {k: float(ideal[k] + (nadir[k] - ideal[k]) * (0.2 * (idx + 1))) for k in obj_keys},
            )
            for idx, dm_id in enumerate(dm_ids)
        }

    if "re34" in prob_name.lower() or "crash" in prob_name.lower():
        re34_mps = {
            "dm1": {"f_1": 1666.4106, "f_2": 6.9593, "f_3": 0.0923},
            "dm2": {"f_1": 1675.4896, "f_2": 6.1428, "f_3": 0.264},
            "dm3": {"f_1": 1674.3033, "f_2": 9.0811, "f_3": 0.0523},
        }
        ideal = problem.get_ideal_point()
        nadir = problem.get_nadir_point()
        obj_keys = list(ideal.keys())
        return {
            dm_id: re34_mps.get(
                dm_id,
                {k: float(ideal[k] + (nadir[k] - ideal[k]) * (0.2 * (idx + 1))) for k in obj_keys},
            )
            for idx, dm_id in enumerate(dm_ids)
        }

    ideal = problem.get_ideal_point()
    nadir = problem.get_nadir_point()
    obj_keys = list(ideal.keys())
    return {
        dm_id: {k: float(ideal[k] + (nadir[k] - ideal[k]) * (0.2 * (idx + 1))) for k in obj_keys}
        for idx, dm_id in enumerate(dm_ids)
    }


RIVER_POLLUTION_PROBLEM_ID = 1
DTLZ2_PROBLEM_ID = 2
DMITRY_FOREST_PROBLEM_ID = 3
METALLURGICAL_PROBLEM_ID = 4
RE34_PROBLEM_ID = 5


def get_problem_instance(problem_id: int, db: Session) -> Problem:
    """Retrieve the problem instance corresponding to problem_id."""
    problem_db = db.get(ProblemDB, problem_id)

    if problem_id == RIVER_POLLUTION_PROBLEM_ID or (
        problem_db and problem_db.name and "river" in problem_db.name.lower()
    ):
        return river_pollution_problem_discrete(five_objective_variant=False)
    if problem_id == DTLZ2_PROBLEM_ID or (problem_db and problem_db.name and "dtlz2" in problem_db.name.lower()):
        return dtlz2(10, 3)
    if problem_id == DMITRY_FOREST_PROBLEM_ID or (
        problem_db and problem_db.name and ("forest" in problem_db.name.lower() or "dmitry" in problem_db.name.lower())
    ):
        return dmitry_forest_problem_disc()
    if problem_id == METALLURGICAL_PROBLEM_ID or (
        problem_db and problem_db.name and "metall" in problem_db.name.lower()
    ):
        return metallurgical_application_discrete()
    if problem_id == RE34_PROBLEM_ID or (
        problem_db and problem_db.name and ("re34" in problem_db.name.lower() or "crash" in problem_db.name.lower())
    ):
        return re34()

    if not problem_db:
        raise HTTPException(status_code=404, detail=f"Problem ID {problem_id} not found in DB.")

    raise HTTPException(status_code=500, detail="Problem deserialization not implemented for this ID.")


@router.post("/init", status_code=status.HTTP_201_CREATED)
async def init_favorite_session(
    request: FavoriteInitRequest,
    db: Annotated[Session, Depends(get_session)],
):
    """Initialize a new Favorite Method session."""
    problem = get_problem_instance(request.problem_id, db)
    mps = request.most_preferred_solutions or fetch_dm_preferences_from_db(request.dm_ids, problem)

    # 1. Missing DM check
    for dm in request.dm_ids:
        if dm not in mps:
            raise HTTPException(status_code=400, detail=f"Missing most preferred solution for Decision Maker '{dm}'.")

    # 2. Setup FavOptions with GPRMoptions
    gprm_options = GPRMOptions(
        method_options=IPR_Options(
            num_initial_reference_points=request.num_initial_reference_points,
            most_preferred_solutions=mps,
            version="box",
        )
    )

    current_mps = copy.deepcopy(mps)
    fav_options = FavOptions(
        total_n_of_candidates=request.total_n_of_candidates,
        candidate_generation_options=request.candidate_generation_options,
        original_most_preferred_solutions=mps,
        current_most_preferred_solutions=current_mps,
        zoom_options=ZoomOptions(num_steps_remaining=request.max_iterations),
        GPRMoptions=gprm_options,
    )

    # 3. Call favorite_method for iteration 1
    try:
        initial_results = favorite_method(problem, fav_options, results_list=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}") from e

    session_id = str(uuid.uuid4())
    is_final = request.max_iterations <= 1

    session_state = FavoriteSessionState(
        session_id=session_id,
        problem_id=request.problem_id,
        dm_ids=request.dm_ids,
        current_iteration=1,
        max_iterations=request.max_iterations,
        phase="decision" if is_final else "consensus_reaching",
        status="voting",
        options=initial_results.FavOptions,
        results_history=[initial_results],
        current_votes={},
        candidates=initial_results.fair_solutions,
        tie_state=initial_results.tie_state,
        final_solution=None,
        current_most_preferred_solutions=current_mps,
        mps_history=[copy.deepcopy(mps)],
        mps_adjustments_history=[],
    )

    results_to_save = _serialize_state(session_state)

    db_session = FavoriteSessionDB(
        session_id=session_id,
        problem_id=request.problem_id,
        state_data=results_to_save,
    )
    db.add(db_session)
    db.commit()

    return results_to_save


@router.get("/state/{session_id}")
async def get_favorite_state(
    session_id: str,
    db: Annotated[Session, Depends(get_session)],
):
    """Retrieve the current state snapshot of an active Favorite Method session."""
    statement = select(FavoriteSessionDB).where(FavoriteSessionDB.session_id == session_id)
    db_session = db.exec(statement).first()

    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found.")

    return db_session.state_data


def _handle_revote_completion(state: FavoriteSessionState) -> None:
    """Resolve revote outcome when all DMs have cast revotes."""
    top_candidates, _ = _tally_votes(state.current_votes)
    is_final_voting = state.phase == "decision"

    if len(top_candidates) > 1:
        winning_idx = min(top_candidates, key=lambda idx: state.candidates[idx].fairness_value)
        state.tie_state = {
            "tied_candidate_indices": top_candidates,
            "resolved_winner_idx": winning_idx,
        }
    else:
        winning_idx = top_candidates[0]
        state.tie_state = None

    if is_final_voting:
        state.final_solution = state.candidates[winning_idx]
        state.status = "completed"
    else:
        state.status = "voting"


@router.post("/vote/{session_id}")
async def submit_favorite_vote(  # noqa: C901
    session_id: str,
    vote: FavoriteVoteRequest,
    db: Annotated[Session, Depends(get_session)],
):
    """Record a vote from a Decision Maker, handling tie detection and revotes."""
    statement = select(FavoriteSessionDB).where(FavoriteSessionDB.session_id == session_id)
    db_session = db.exec(statement).first()

    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = _deserialize_state(db_session.state_data)

    if vote.dm_id not in state.dm_ids:
        raise HTTPException(status_code=403, detail=f"User '{vote.dm_id}' is not an authorized DM.")

    if vote.vote_idx < 0 or vote.vote_idx >= len(state.candidates):
        raise HTTPException(status_code=400, detail=f"Candidate index {vote.vote_idx} out of bounds.")

    if state.status not in ["voting", "revote_pending"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot submit vote when session status is '{state.status}'.",
        )

    if state.status == "revote_pending":
        tied_indices = (state.tie_state or {}).get("tied_candidate_indices", [])
        if vote.vote_idx not in tied_indices:
            raise HTTPException(
                status_code=400,
                detail=f"Candidate index {vote.vote_idx} is not among tied candidates {tied_indices}.",
            )
        state.current_votes[vote.dm_id] = vote.vote_idx

        # When all DMs have revoted
        if len(state.current_votes) == len(state.dm_ids):
            _handle_revote_completion(state)

    elif state.status == "voting":
        state.current_votes[vote.dm_id] = vote.vote_idx

        if len(state.current_votes) == len(state.dm_ids):
            top_candidates, _ = _tally_votes(state.current_votes)

            if len(top_candidates) > 1:
                is_adjacent = False
                compromise_solution = None

                if len(top_candidates) == 2:  # noqa: PLR2004
                    prev_results = state.results_history[-1]
                    pts_mat, _, labels = cluster_points(prev_results)
                    idx_a, idx_b = top_candidates[0], top_candidates[1]
                    is_adjacent = check_adjacency(pts_mat, labels, idx_a, idx_b)

                    if is_adjacent:
                        problem = get_problem_instance(state.problem_id, db)
                        tied_votes = {dm: v for dm, v in state.current_votes.items() if v in top_candidates}
                        compromise_solution = tie_breaker_avgproj(problem, tied_votes, state.candidates)

                if is_adjacent and compromise_solution is not None:
                    if state.phase == "decision":
                        state.final_solution = compromise_solution
                        state.status = "completed"
                        state.tie_state = {
                            "strategy": "tie_breaker_avgproj",
                            "tied_candidate_indices": sorted(top_candidates),
                            "is_adjacent": True,
                        }
                    else:
                        state.tie_state = {
                            "strategy": "tie_breaker_avgproj",
                            "tied_candidate_indices": sorted(top_candidates),
                            "is_adjacent": True,
                            "compromise_solution": compromise_solution.model_dump(),
                        }
                else:
                    # Non-adjacent or >2 tied candidates: Trigger revote
                    state.status = "revote_pending"
                    state.tie_state = {
                        "tied_candidate_indices": sorted(top_candidates),
                        "strategy": "Simple Vote-Again",
                        "is_adjacent": False,
                    }
                    state.current_votes = {}
            else:
                state.tie_state = None
                if state.phase == "decision":
                    winner_idx = top_candidates[0]
                    state.final_solution = state.candidates[winner_idx]
                    state.status = "completed"
                    state.tie_state = {"final_winner_idx": winner_idx}

    db_session.state_data = _serialize_state(state)
    db.add(db_session)
    db.commit()

    return {
        "message": f"Vote recorded for {vote.dm_id}",
        "current_votes": state.current_votes,
        "phase": state.phase,
        "status": state.status,
        "is_ready": len(state.current_votes) == len(state.dm_ids),
        "tie_state": state.tie_state,
        "final_solution": state.final_solution.model_dump() if state.final_solution else None,
    }


@router.post("/iterate/{session_id}")
async def iterate_favorite_session(
    session_id: str,
    db: Annotated[Session, Depends(get_session)],
):
    """Advance to the next iteration or the final voting phase of the Favorite Method session."""
    statement = select(FavoriteSessionDB).where(FavoriteSessionDB.session_id == session_id)
    db_session = db.exec(statement).first()

    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = _deserialize_state(db_session.state_data)
    problem = get_problem_instance(state.problem_id, db)

    if state.status == "revote_pending":
        raise HTTPException(status_code=400, detail="Cannot proceed while revote is pending.")

    if state.status == "completed":
        raise HTTPException(status_code=400, detail="Session is already completed.")

    if state.phase == "decision":
        raise HTTPException(
            status_code=400,
            detail="In final decision phase: waiting for Decision Makers to cast final votes.",
        )

    if len(state.current_votes) < len(state.dm_ids):
        raise HTTPException(status_code=400, detail="Cannot iterate until all DMs have voted.")

    # 1. Adapt DM preferred solutions based on cast votes and presented candidates
    current_mps = state.current_most_preferred_solutions or state.options.original_most_preferred_solutions
    adapted_mps, adjustments_meta = adapt_all_dm_preferences(
        problem=problem,
        current_mps=current_mps,
        candidates=state.candidates,
        votes=state.current_votes,
        epsilon=1e-4,
    )
    state.current_most_preferred_solutions = adapted_mps
    state.mps_history.append(copy.deepcopy(adapted_mps))
    if adjustments_meta:
        state.mps_adjustments_history.append(adjustments_meta)

    # Determine winning candidate index or tie-breaker compromise
    if (
        state.tie_state
        and state.tie_state.get("strategy") == "tie_breaker_avgproj"
        and "compromise_solution" in state.tie_state
    ):
        compromise_dict = state.tie_state["compromise_solution"]
        compromise_sol = FairSolution(**compromise_dict)
        prev_results = state.results_history[-1]
        all_points = prev_results.GPRMResults.raw_results.evaluated_points
        _, new_labels, winning_idx = recluster_for_tie_breaker(all_points, state.candidates, compromise_sol)
        labels = new_labels
    elif state.tie_state and "resolved_winner_idx" in state.tie_state:
        winning_idx = state.tie_state["resolved_winner_idx"]
        prev_results = state.results_history[-1]
        _, _, labels = cluster_points(prev_results)
    else:
        top_candidates, _ = _tally_votes(state.current_votes)

        if len(top_candidates) > 1:
            # Fallback tie check
            state.status = "revote_pending"
            state.tie_state = {"tied_candidate_indices": sorted(top_candidates), "strategy": "Simple Vote-Again"}
            state.current_votes = {}
            db_session.state_data = _serialize_state(state)
            db.add(db_session)
            db.commit()
            return db_session.state_data

        winning_idx = top_candidates[0]
        prev_results = state.results_history[-1]
        _, _, labels = cluster_points(prev_results)

    num_objectives = len(problem.get_ideal_point())
    fraction = calculate_fraction_to_keep(state.current_iteration - 1, state.max_iterations, num_objectives)

    num_ref_points = max(
        state.options.GPRMoptions.method_options.num_initial_reference_points or 1000,
        1000,
    )

    next_mps = generate_next_iteration_mps(
        fav_results=prev_results,
        cluster_labels=labels,
        winning_idx=winning_idx,
        fraction_to_keep=fraction,
        num_new_points=num_ref_points,
    )

    next_options = state.options.model_copy(deep=True)
    next_options.original_most_preferred_solutions = state.options.original_most_preferred_solutions
    next_options.current_most_preferred_solutions = adapted_mps
    next_options.preferences_already_adapted = True
    next_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
    next_options.GPRMoptions.method_options.num_initial_reference_points = num_ref_points
    next_options.GPRMoptions.method_options.version = "convex_hull"
    next_options.votes = dict.fromkeys(state.dm_ids, winning_idx)

    try:
        new_results = favorite_method(problem, next_options, state.results_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if new_results.status == "revote_pending":
        state.status = "revote_pending"
        tied_idxs = (new_results.tie_state or {}).get("tied_indices", [])
        state.tie_state = {"tied_candidate_indices": tied_idxs}
        state.current_votes = {}
    else:
        state.results_history.append(new_results)
        state.options = new_results.FavOptions
        state.candidates = new_results.fair_solutions
        state.current_iteration += 1
        state.current_votes = {}
        state.tie_state = None
        state.status = "voting"
        state.current_most_preferred_solutions = new_results.FavOptions.current_most_preferred_solutions or adapted_mps

        if state.current_iteration >= state.max_iterations:
            state.phase = "decision"
        else:
            state.phase = "consensus_reaching"

    db_session.state_data = _serialize_state(state)
    db.add(db_session)
    db.commit()

    return db_session.state_data
