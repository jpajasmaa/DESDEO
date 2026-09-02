"""Automated integration tests for the Favorite method FastAPI router."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from desdeo.api.routers.favorite import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.fixture
def sample_init_payload():
    """Sample initialization payload for River Pollution Discrete with 3 DMs."""
    return {
        "problem_id": 1,  # river_pollution_problem_discrete
        "dm_ids": ["dm1", "dm2", "dm3"],
        "total_n_of_candidates": 5,
        "candidate_generation_options": "mm",
        "max_iterations": 3,
        "num_initial_reference_points": 50,
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
    """Test initialization with problem 2 (dtlz2)."""
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

