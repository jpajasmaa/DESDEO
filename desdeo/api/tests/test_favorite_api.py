"""Tests for the FAVORITE FastAPI router endpoints and workflow."""

from unittest.mock import patch

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session

from desdeo.api.db import engine
from desdeo.api.models import ProblemDB, User
from desdeo.api.routers.favorite import router
from desdeo.gdm import calculate_dm_utility
from desdeo.gdm.favorite_method import FairSolution, FavResults, GPRMResults, IPR_Results
from desdeo.problem.testproblems import dmitry_forest_problem_disc, river_pollution_problem_discrete
from desdeo.tools.iterative_pareto_representer import _EvaluatedPoint

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def _mock_favorite_method_call(problem, fav_options, results_list=None):
    """Fast mock for favorite_method router integration tests."""
    obj_names = list(problem.get_ideal_point().keys())
    var_names = [v.name for v in problem.variables] if problem.variables else ["x1"]
    cands = [
        FairSolution(
            fairness_criterion="mmfair" if i == 0 else "haus",
            fairness_value=float(i) * 0.1,
            objective_values={name: float(i + 1) for name in obj_names},
            variable_values=dict.fromkeys(var_names, 1.0),
        )
        for i in range(fav_options.total_n_of_candidates)
    ]
    eval_pts = [
        _EvaluatedPoint(
            reference_point={name: float(i + 1) for name in obj_names},
            targets={name: float(i + 1) for name in obj_names},
            objectives={name: float(i + 1) for name in obj_names},
        )
        for i in range(fav_options.total_n_of_candidates)
    ]
    gprm = GPRMResults(
        raw_results=IPR_Results(evaluated_points=eval_pts),
        solutions=None,
        outputs=pl.DataFrame(
            {name: [float(i + 1) for i in range(fav_options.total_n_of_candidates)] for name in obj_names}
        ),
    )
    fav_options.GPRMoptions.fake_ideal = problem.get_ideal_point()
    fav_options.GPRMoptions.fake_nadir = problem.get_nadir_point()
    return FavResults(
        FavOptions=fav_options,
        GPRMResults=gprm,
        fair_solutions=cands,
        status="success",
    )


@pytest.fixture
def sample_init_payload():
    """Sample initialization payload for River Pollution Discrete with 3 DMs."""
    return {
        "problem_id": 1,  # river_pollution_problem_discrete
        "dm_ids": ["dm1", "dm2", "dm3"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 3,
        "num_initial_reference_points": 20,
        "most_preferred_solutions": {
            "dm1": {"f1": 5.9066, "f2": 3.2894, "f3": 6.5792, "f4": -4.5460},
            "dm2": {"f1": 5.4290, "f2": 3.0121, "f3": 7.2395, "f4": -0.9135},
            "dm3": {"f1": 6.1623, "f2": 2.8839, "f3": 5.2575, "f4": -0.0045},
        },
    }


def test_init_session_success(sample_init_payload):
    """Test successful initialization with standard payload."""
    response = client.post("/favorite/init", json=sample_init_payload)
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["current_iteration"] == 1
    assert len(data["candidates"]) == 5
    assert data["status"] == "voting"
    # Ensure standard objective_values is present
    assert "objective_values" in data["candidates"][0]


def test_init_schema_resilience():
    """Test that schema accepts string problem_id or empty values without 422 error and defaults to 3 DMs."""
    payload_str_id = {
        "problem_id": "1",
        "total_n_of_candidates": 5,
        "num_initial_reference_points": 20,
        "most_preferred_solutions": {
            "dm1": {"f1": 5.9066, "f2": 3.2894, "f3": 6.5792, "f4": -4.5460},
            "dm2": {"f1": 5.4290, "f2": 3.0121, "f3": 7.2395, "f4": -0.9135},
            "dm3": {"f1": 6.1623, "f2": 2.8839, "f3": 5.2575, "f4": -0.0045},
        },
    }
    res = client.post("/favorite/init", json=payload_str_id)
    assert res.status_code == 201
    assert res.json()["problem_id"] == 1
    assert res.json()["dm_ids"] == ["dm1", "dm2", "dm3"]

    payload_empty_id = {
        "problem_id": "",
        "num_initial_reference_points": 20,
        "most_preferred_solutions": {
            "dm1": {"f1": 5.9066, "f2": 3.2894, "f3": 6.5792, "f4": -4.5460},
            "dm2": {"f1": 5.4290, "f2": 3.0121, "f3": 7.2395, "f4": -0.9135},
            "dm3": {"f1": 6.1623, "f2": 2.8839, "f3": 5.2575, "f4": -0.0045},
        },
    }
    res2 = client.post("/favorite/init", json=payload_empty_id)
    assert res2.status_code == 201
    assert res2.json()["problem_id"] == 1
    assert res2.json()["dm_ids"] == ["dm1", "dm2", "dm3"]


def test_init_problem_2_dtlz2():
    """Test initialization with problem 2 (dtlz2) using mocked solver."""
    payload = {
        "problem_id": 2,
        "dm_ids": ["dm1", "dm2", "dm3"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 2,
        "num_initial_reference_points": 50,
        "most_preferred_solutions": {
            "dm1": {"f_1": 0.6666, "f_2": 0.6666, "f_3": 0.3333},
            "dm2": {"f_1": 0.6666, "f_2": 0.3333, "f_3": 0.6666},
            "dm3": {"f_1": 0.3333, "f_2": 0.6666, "f_3": 0.6666},
        },
    }
    with patch("desdeo.api.routers.favorite.favorite_method", side_effect=_mock_favorite_method_call):
        res = client.post("/favorite/init", json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["problem_id"] == 2
        assert len(data["candidates"]) == 5


def test_init_missing_dm_mps_raises_400():
    """Test that missing DM in manual MPS dictionary raises 400 Bad Request."""
    bad_payload = {
        "problem_id": 1,
        "dm_ids": ["dm1", "dm2", "dm3"],
        "most_preferred_solutions": {
            "dm1": {"f1": 5.9066, "f2": 3.2894, "f3": 6.5792, "f4": -4.5460},
            "dm2": {"f1": 5.4290, "f2": 3.0121, "f3": 7.2395, "f4": -0.9135},
            # dm3 missing
        },
    }
    response = client.post("/favorite/init", json=bad_payload)
    assert response.status_code == 400
    assert "Missing most preferred solution" in response.json()["detail"]


def test_get_session_state(sample_init_payload):
    """Test retrieving session state by ID."""
    init_res = client.post("/favorite/init", json=sample_init_payload).json()
    session_id = init_res["session_id"]

    state_res = client.get(f"/favorite/state/{session_id}")
    assert state_res.status_code == 200
    state_data = state_res.json()
    assert state_data["session_id"] == session_id
    assert state_data["dm_ids"] == ["dm1", "dm2", "dm3"]


def test_voting_flow_and_iteration(sample_init_payload):
    """Test voting flow, error cases, and advancing iteration with 3 DMs."""
    # 1. Initialize
    init_res = client.post("/favorite/init", json=sample_init_payload).json()
    session_id = init_res["session_id"]

    # 2. Premature iterate should fail (0 votes)
    premature_res = client.post(f"/favorite/iterate/{session_id}")
    assert premature_res.status_code == 400

    # 3. DM1 votes
    v1 = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 0})
    assert v1.status_code == 200
    assert v1.json()["is_ready"] is False

    # 4. DM2 votes
    v2 = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 0})
    assert v2.status_code == 200
    assert v2.json()["is_ready"] is False

    # 5. Unauthorized DM vote rejection
    v_bad = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "intruder", "vote_idx": 0})
    assert v_bad.status_code == 403

    # 6. Invalid candidate index rejection
    v_out_of_bounds = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": 99})
    assert v_out_of_bounds.status_code == 400

    # 7. DM3 votes for candidate 1 (Candidate 0 wins 2-1)
    v3 = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": 1})
    assert v3.status_code == 200
    assert v3.json()["is_ready"] is True
    assert v3.json()["status"] == "voting"

    # 8. Advance Iteration
    iter_res = client.post(f"/favorite/iterate/{session_id}")
    assert iter_res.status_code == 200
    iter_data = iter_res.json()
    assert iter_data["current_iteration"] == 2
    assert iter_data["current_votes"] == {}
    assert iter_data["status"] == "voting"


def test_tie_detection_and_revote_flow(sample_init_payload):
    """Test 3-way tie triggering revote_pending, restricted revoting, and successful iteration."""
    init_res = client.post("/favorite/init", json=sample_init_payload).json()
    session_id = init_res["session_id"]

    # 1. Cast 3-way tie: dm1 -> 0, dm2 -> 1, dm3 -> 2
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 0})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 1})
    tie_vote_res = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": 2})

    assert tie_vote_res.status_code == 200
    tie_data = tie_vote_res.json()
    assert tie_data["status"] == "revote_pending"
    assert tie_data["is_ready"] is False
    assert tie_data["tie_state"]["tied_candidate_indices"] == [0, 1, 2]
    assert tie_data["current_votes"] == {}

    # 2. Premature iterate during revote_pending should fail
    iter_fail = client.post(f"/favorite/iterate/{session_id}")
    assert iter_fail.status_code == 400
    detail_lower = iter_fail.json()["detail"].lower()
    assert "revote while" in detail_lower or "revote is pending" in detail_lower

    # 3. Voting for non-tied candidate (e.g. 3) should be rejected
    bad_vote = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 3})
    assert bad_vote.status_code == 400
    assert "not among tied candidates" in bad_vote.json()["detail"]

    # 4. Cast valid revotes breaking the tie (dm1 -> 0, dm2 -> 0, dm3 -> 1)
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 0})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 0})
    final_revote = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": 1})

    assert final_revote.status_code == 200
    assert final_revote.json()["status"] == "voting"
    assert final_revote.json()["is_ready"] is True

    # 5. Iteration advances successfully
    iter_res = client.post(f"/favorite/iterate/{session_id}")
    assert iter_res.status_code == 200
    iter_data = iter_res.json()
    assert iter_data["current_iteration"] == 2
    assert iter_data["status"] == "voting"
    assert iter_data["current_votes"] == {}


def test_revote_persisting_tie_random_fallback(sample_init_payload):
    """Test that if a tie persists after revoting, it is broken by randomly selecting from tied candidates."""
    init_res = client.post("/favorite/init", json=sample_init_payload).json()
    session_id = init_res["session_id"]

    # Initial tie: 0, 1, 2
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 0})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 1})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": 2})

    # Revote still tied: dm1 -> 0, dm2 -> 1, dm3 -> 2
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 0})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 1})
    res = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": 2})

    assert res.status_code == 200
    assert res.json()["status"] == "voting"
    resolved_winner = res.json()["tie_state"]["resolved_winner_idx"]
    assert resolved_winner in [0, 1, 2]

    # Advance iteration successfully using the resolved winner
    iter_res = client.post(f"/favorite/iterate/{session_id}")
    assert iter_res.status_code == 200
    assert iter_res.json()["current_iteration"] == 2


def test_full_lifecycle_to_final_decision(sample_init_payload):
    """Test full lifecycle to final decision.

    1. Initialize session with max_iterations = 2.
    2. Complete Iteration 1 voting and trigger /iterate.
    3. Verify status remains 'voting' and phase becomes 'decision'.
    4. Submit final votes for all DMs.
    5. Verify session status transitions to 'completed' and final_solution is populated.
    """
    sample_init_payload["max_iterations"] = 2
    init_res = client.post("/favorite/init", json=sample_init_payload).json()
    session_id = init_res["session_id"]
    assert init_res["status"] == "voting"
    assert init_res["current_iteration"] == 1

    # Complete Iteration 1 voting
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 1})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 1})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": 0})

    # Trigger /iterate to advance to final contracted region
    iter_res = client.post(f"/favorite/iterate/{session_id}")
    assert iter_res.status_code == 200
    iter_data = iter_res.json()
    assert iter_data["current_iteration"] == 2
    assert iter_data["status"] == "voting"
    assert iter_data["phase"] == "decision"
    assert iter_data["current_votes"] == {}

    # DMs submit final votes
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 0})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 0})
    final_vote = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": 2})

    assert final_vote.status_code == 200
    vote_data = final_vote.json()
    assert vote_data["status"] == "completed"
    assert vote_data["final_solution"] is not None
    assert vote_data["final_solution"]["objective_values"] == iter_data["candidates"][0]["objective_values"]

    # Verify state snapshot has final_solution populated
    state_res = client.get(f"/favorite/state/{session_id}").json()
    assert state_res["status"] == "completed"
    assert state_res["final_solution"] is not None
    assert state_res["final_solution"]["objective_values"] == iter_data["candidates"][0]["objective_values"]


def test_init_problem_3_dmitry_forest():
    """Test initialization with Dmitry Forest Problem (problem_id 3) with 4 DMs and auto-fetched MPS."""
    with Session(engine) as db:
        prob = db.get(ProblemDB, 3)
        if not prob:
            user = db.get(User, 1)
            prob_db = ProblemDB.from_problem(dmitry_forest_problem_disc(), user=user)
            db.add(prob_db)
            db.commit()

    payload = {
        "problem_id": 3,
        "dm_ids": ["dm1", "dm2", "dm3", "dm4"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 3,
        "num_initial_reference_points": 50,
    }
    with patch("desdeo.api.routers.favorite.favorite_method", side_effect=_mock_favorite_method_call):
        res = client.post("/favorite/init", json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["problem_id"] == 3
        assert data["dm_ids"] == ["dm1", "dm2", "dm3", "dm4"]
        assert len(data["candidates"]) == 5
        assert data["status"] == "voting"

        # Verify candidates have the 4 forest objectives: Rev, HA, Carb, DW
        candidate_objs = data["candidates"][0]["objective_values"]
        assert set(candidate_objs.keys()) == {"Rev", "HA", "Carb", "DW"}

        # Test submitting a vote from DM4
        session_id = data["session_id"]
        vote_res = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm4", "vote_idx": 0})
        assert vote_res.status_code == 200
        assert vote_res.json()["current_votes"]["dm4"] == 0


def test_adjacent_tie_breaker_avgproj():
    """Test that an adjacent tie between 2 candidates triggers tie_breaker_avgproj instead of a revote."""
    payload = {
        "problem_id": 1,
        "dm_ids": ["dm1", "dm2"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 2,
        "num_initial_reference_points": 1000,
    }
    init_res = client.post("/favorite/init", json=payload).json()
    session_id = init_res["session_id"]

    # Cast votes creating a 2-way tie (candidate 0 vs candidate 1)
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 0})
    res_vote = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 1})

    assert res_vote.status_code == 200
    data = res_vote.json()
    assert data["is_ready"] is True
    assert data["tie_state"] is not None
    assert data["tie_state"]["strategy"] == "tie_breaker_avgproj"
    assert data["tie_state"]["is_adjacent"] is True
    assert "compromise_solution" in data["tie_state"]
    assert data["tie_state"]["compromise_solution"]["fairness_criterion"] == "tie_breaker_average_projection"

    # Advance iteration using the compromise solution
    iter_res = client.post(f"/favorite/iterate/{session_id}")
    assert iter_res.status_code == 200
    iter_data = iter_res.json()
    assert iter_data["current_iteration"] == 2
    assert iter_data["phase"] == "decision"
    assert len(iter_data["candidates"]) == 5


def test_final_decision_adjacent_tie_breaker_avgproj():
    """Test that an adjacent tie in the final decision phase completes the session with compromise solution."""
    payload = {
        "problem_id": 1,
        "dm_ids": ["dm1", "dm2"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 1,  # Direct to final decision phase
        "num_initial_reference_points": 1000,
    }
    init_res = client.post("/favorite/init", json=payload).json()
    session_id = init_res["session_id"]
    assert init_res["phase"] == "decision"

    # Cast votes creating an adjacent tie in final decision phase
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 0})
    res_vote = client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 1})

    assert res_vote.status_code == 200
    data = res_vote.json()
    assert data["status"] == "completed"
    assert data["final_solution"] is not None
    assert data["final_solution"]["fairness_criterion"] == "tie_breaker_average_projection"


def test_dtlz2_multi_iteration_continuous():
    """Test that DTLZ2 (continuous problem) runs multi-iteration state transitions cleanly."""
    payload = {
        "problem_id": 2,
        "dm_ids": ["dm1", "dm2", "dm3"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 2,
        "num_initial_reference_points": 50,
    }
    with patch("desdeo.api.routers.favorite.favorite_method", side_effect=_mock_favorite_method_call):
        init_res = client.post("/favorite/init", json=payload).json()
        session_id = init_res["session_id"]
        assert init_res["current_iteration"] == 1
        assert len(init_res["candidates"]) == 5

        # Vote for candidate 0 across all DMs
        client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 0})
        client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 0})
        client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": 0})

        # Advance iteration
        iter_res = client.post(f"/favorite/iterate/{session_id}")
        assert iter_res.status_code == 200
        iter_data = iter_res.json()
        assert iter_data["current_iteration"] == 2
        assert iter_data["phase"] == "decision"
        assert len(iter_data["candidates"]) == 5


def test_dm_preferred_solutions_adaptation_suboptimal_vote():
    """Test that voting for a suboptimal candidate triggers minimum adjustment of MPS in the API."""
    payload = {
        "problem_id": 1,
        "dm_ids": ["dm1", "dm2", "dm3"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 2,
        "num_initial_reference_points": 20,
    }
    init_res = client.post("/favorite/init", json=payload).json()
    session_id = init_res["session_id"]
    candidates = init_res["candidates"]

    state_res = client.get(f"/favorite/state/{session_id}").json()
    orig_mps_dm1 = state_res["options"]["original_most_preferred_solutions"]["dm1"]
    assert state_res["current_most_preferred_solutions"]["dm1"] == orig_mps_dm1
    assert len(state_res["mps_history"]) == 1
    assert len(state_res["mps_adjustments_history"]) == 0

    problem = river_pollution_problem_discrete(five_objective_variant=False)
    # Calculate utility of all 5 candidates for dm1
    utils = [calculate_dm_utility(problem, orig_mps_dm1, c["objective_values"]) for c in candidates]
    best_idx = int(max(range(len(utils)), key=lambda i: utils[i]))
    worst_idx = int(min(range(len(utils)), key=lambda i: utils[i]))
    assert worst_idx != best_idx

    # DM1 votes for worst candidate; DM2 and DM3 vote for best_idx to ensure a decisive majority winner
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": worst_idx})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": best_idx})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": best_idx})

    # Advance iteration
    iter_res = client.post(f"/favorite/iterate/{session_id}")
    assert iter_res.status_code == 200

    updated_state = client.get(f"/favorite/state/{session_id}").json()
    # 1. Original MPS must remain unchanged
    assert updated_state["options"]["original_most_preferred_solutions"]["dm1"] == orig_mps_dm1
    # 2. History tracked
    assert len(updated_state["mps_history"]) == 2
    assert len(updated_state["mps_adjustments_history"]) == 1

    dm1_meta = updated_state["mps_adjustments_history"][0]["dm1"]
    assert dm1_meta["adjusted"] is True
    assert dm1_meta["lambda_shift"] > 0.0

    # 3. Current MPS adapted towards voted candidate
    new_mps_dm1 = updated_state["current_most_preferred_solutions"]["dm1"]
    assert new_mps_dm1 != orig_mps_dm1


def test_dm_preferred_solutions_adaptation_consistent_vote():
    """Test that voting for the top utility candidate leaves MPS unchanged (lambda=0)."""
    payload = {
        "problem_id": 1,
        "dm_ids": ["dm1", "dm2"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 2,
        "num_initial_reference_points": 20,
    }
    init_res = client.post("/favorite/init", json=payload).json()
    session_id = init_res["session_id"]
    candidates = init_res["candidates"]

    state_res = client.get(f"/favorite/state/{session_id}").json()
    orig_mps_dm1 = state_res["options"]["original_most_preferred_solutions"]["dm1"]

    problem = river_pollution_problem_discrete(five_objective_variant=False)
    utils = [calculate_dm_utility(problem, orig_mps_dm1, c["objective_values"]) for c in candidates]
    best_idx = int(max(range(len(utils)), key=lambda i: utils[i]))

    # Both DMs vote for best_idx
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": best_idx})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": best_idx})

    iter_res = client.post(f"/favorite/iterate/{session_id}")
    assert iter_res.status_code == 200

    updated_state = client.get(f"/favorite/state/{session_id}").json()
    assert updated_state["current_most_preferred_solutions"]["dm1"] == orig_mps_dm1
    dm1_meta = updated_state["mps_adjustments_history"][0]["dm1"]
    assert dm1_meta["adjusted"] is False
    assert dm1_meta["lambda_shift"] == 0.0


def test_init_problem_4_discrete_metallurgical():
    """Test initialization and iteration for Problem 4 (Discrete Metallurgical Application Problem) with 5 DMs."""
    payload = {
        "problem_id": 4,
        "dm_ids": ["dm1", "dm2", "dm3", "dm4", "dm5"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 2,
        "num_initial_reference_points": 50,
    }
    with patch("desdeo.api.routers.favorite.favorite_method", side_effect=_mock_favorite_method_call):
        init_res = client.post("/favorite/init", json=payload)
        assert init_res.status_code == 201
        data = init_res.json()
        assert data["problem_id"] == 4
        assert len(data["dm_ids"]) == 5
        assert len(data["candidates"]) == 5
        assert data["current_iteration"] == 1
        assert data["status"] == "voting"

        # Verify objective keys
        cand0_objs = data["candidates"][0]["objective_values"]
        assert set(cand0_objs.keys()) == {"YS", "UTS", "ELON", "CE", "COST"}

        session_id = data["session_id"]

        # All 5 DMs vote: 3 vote for candidate 0, 2 vote for candidate 1
        for dm in ["dm1", "dm2", "dm3"]:
            client.post(f"/favorite/vote/{session_id}", json={"dm_id": dm, "vote_idx": 0})
        for dm in ["dm4", "dm5"]:
            client.post(f"/favorite/vote/{session_id}", json={"dm_id": dm, "vote_idx": 1})

        # Advance iteration
        iter_res = client.post(f"/favorite/iterate/{session_id}")
        assert iter_res.status_code == 200
        iter_data = iter_res.json()
        assert iter_data["current_iteration"] == 2
        assert iter_data["status"] == "voting"
        assert len(iter_data["candidates"]) == 5


def test_init_problem_5_re34_pyomo():
    """Test initialization and continuous solving for Problem 5 (RE34 Crashworthiness) with 3 DMs."""
    payload = {
        "problem_id": 5,
        "dm_ids": ["dm1", "dm2", "dm3"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 2,
        "num_initial_reference_points": 10,
    }
    init_res = client.post("/favorite/init", json=payload)
    assert init_res.status_code == 201
    data = init_res.json()
    assert data["problem_id"] == 5
    assert len(data["dm_ids"]) == 3
    assert len(data["candidates"]) == 5
    assert data["current_iteration"] == 1
    assert data["status"] == "voting"

    # Verify objective keys
    cand0_objs = data["candidates"][0]["objective_values"]
    assert set(cand0_objs.keys()) == {"f_1", "f_2", "f_3"}

    session_id = data["session_id"]

    # All 3 DMs vote: 2 vote for candidate 0, 1 votes for candidate 1
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm1", "vote_idx": 0})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm2", "vote_idx": 0})
    client.post(f"/favorite/vote/{session_id}", json={"dm_id": "dm3", "vote_idx": 1})

    # Advance iteration using Pyomo + Ipopt
    iter_res = client.post(f"/favorite/iterate/{session_id}")
    assert iter_res.status_code == 200
    iter_data = iter_res.json()
    assert iter_data["current_iteration"] == 2
    assert iter_data["status"] == "voting"
    assert len(iter_data["candidates"]) == 5
