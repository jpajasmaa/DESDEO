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
    FavOptions,
    GPRMOptions,
    IPR_Options,
    ZoomOptions,
    calculate_fraction_to_keep,
    cluster_points,
    favorite_method,
    generate_next_iteration_mps,
)
from desdeo.problem.schema import Problem
from desdeo.problem.testproblems import dtlz2, river_pollution_problem_discrete

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


def fetch_dm_preferences_from_db(dm_ids: list[str], problem: Problem) -> dict[str, dict[str, float]]:
    """Fetch or compute default reference points for decision makers based on ideal and nadir points."""
    if getattr(problem, "name", "") == "The river pollution problem (Discrete)":
        river_mps = {
            "dm1": {"f1": 5.9066, "f2": 3.2894, "f3": 6.5792, "f4": -4.5460},
            "dm2": {"f1": 5.4290, "f2": 3.0121, "f3": 7.2395, "f4": -0.9135},
            "dm3": {"f1": 6.1623, "f2": 2.8839, "f3": 5.2575, "f4": -0.0045},
        }
        return {
            dm_id: river_mps.get(dm_id, {
                "f1": 5.80 + (idx * 0.1),
                "f2": 3.10 + (idx * 0.05),
                "f3": 6.00 + (idx * 0.2),
                "f4": -3.00 + (idx * 0.5),
            })
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


def get_problem_instance(problem_id: int, db: Session) -> Problem:
    """Retrieve the problem instance corresponding to problem_id."""
    problem_db = db.get(ProblemDB, problem_id)
    if not problem_db:
        raise HTTPException(status_code=404, detail=f"Problem ID {problem_id} not found in DB.")

    if problem_id == RIVER_POLLUTION_PROBLEM_ID:
        return river_pollution_problem_discrete(five_objective_variant=False)
    if problem_id == DTLZ2_PROBLEM_ID:
        return dtlz2(10, 3)

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

    fav_options = FavOptions(
        total_n_of_candidates=request.total_n_of_candidates,
        candidate_generation_options=request.candidate_generation_options,
        original_most_preferred_solutions=mps,
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
async def submit_favorite_vote(
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
                # Tie detected! Trigger revote
                state.status = "revote_pending"
                state.tie_state = {"tied_candidate_indices": sorted(top_candidates)}
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

    # Determine winning candidate index
    if state.tie_state and "resolved_winner_idx" in state.tie_state:
        winning_idx = state.tie_state["resolved_winner_idx"]
    else:
        top_candidates, _ = _tally_votes(state.current_votes)

        if len(top_candidates) > 1:
            # Tie detected during tallying in iterate
            state.status = "revote_pending"
            state.tie_state = {"tied_candidate_indices": sorted(top_candidates)}
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

    next_mps = generate_next_iteration_mps(
        fav_results=prev_results,
        cluster_labels=labels,
        winning_idx=winning_idx,
        fraction_to_keep=fraction,
        num_new_points=state.options.GPRMoptions.method_options.num_initial_reference_points,
    )

    next_options = state.options.model_copy(deep=True)
    next_options.GPRMoptions.method_options.most_preferred_solutions = next_mps
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

        if state.current_iteration >= state.max_iterations:
            state.phase = "decision"
        else:
            state.phase = "consensus_reaching"

    db_session.state_data = _serialize_state(state)
    db.add(db_session)
    db.commit()

    return db_session.state_data

