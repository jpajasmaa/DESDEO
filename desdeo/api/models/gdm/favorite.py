"""Pydantic schemas for the Favorite method Web-API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import JSON, Column
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel

from desdeo.gdm.favorite_method import FairSolution, FavOptions, FavResults


class FavoriteInitRequest(BaseModel):
    """Payload for initializing a new Favorite method session."""

    model_config = ConfigDict(extra="ignore")

    problem_id: int | str = Field(
        default=1,
        description="ID of the problem in the database or test registry.",
    )
    dm_ids: list[str] = Field(
        default_factory=lambda: ["dm1", "dm2", "dm3"],
        description="List of Decision Maker usernames or IDs (e.g., ['dm1', 'dm2', 'dm3']).",
    )
    total_n_of_candidates: int = Field(default=5, ge=1, description="Total candidates presented per iteration.")
    fairness_criterion: str = Field(
        default="mm",
        description="Fairness criterion ('mm', 'utilitarian', 'nash').",
    )
    candidate_generation_options: str = Field(
        default="mm",
        description="Alias for fairness_criterion ('mm', 'utilitarian', 'nash').",
    )
    voting_rule: Literal["plurality", "majority"] = Field(
        default="plurality",
        description="Voting rule used to determine the Round 1 winner: 'plurality' or 'majority'.",
    )
    borda_weights: tuple[float, float] | tuple[int, int] = Field(
        default=(2, 1),
        description="Weights (w1, w2) for Round 1 and Round 2 votes in Borda scoring.",
    )
    max_iterations: int = Field(default=5, ge=1, description="Total planned zooming iterations.")
    num_initial_reference_points: int = Field(default=1000, ge=1, description="IPR sample points.")
    most_preferred_solutions: dict[str, dict[str, float]] | None = Field(
        default=None,
        description="Optional manual MPS map (e.g., {'dm1': {'f_1': 0.1, 'f_2': 0.05}}). If None, fetched from DB.",
    )

    @field_validator("candidate_generation_options", mode="before")
    @classmethod
    def sync_cand_gen_options(cls, v: Any, info: Any) -> Any:
        return v

    @model_validator(mode="before")
    @classmethod
    def sync_fairness_criterion(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "fairness_criterion" in data and "candidate_generation_options" not in data:
                data["candidate_generation_options"] = data["fairness_criterion"]
            elif "candidate_generation_options" in data and "fairness_criterion" not in data:
                data["fairness_criterion"] = data["candidate_generation_options"]
        return data

    @field_validator("problem_id", mode="before")
    @classmethod
    def parse_problem_id(cls, v: Any) -> int:
        """Parse problem_id into an integer, defaulting to 1 for empty/missing values."""
        if v is None or v == "":
            return 1
        try:
            return int(v)
        except (ValueError, TypeError):
            return 1

    @field_validator("dm_ids", mode="before")
    @classmethod
    def parse_dm_ids(cls, v: Any) -> list[str]:
        """Ensure dm_ids is a non-empty list of strings."""
        if not v:
            return ["dm1", "dm2", "dm3"]
        return [str(item) for item in v]


class FavoriteVoteRequest(BaseModel):
    """Payload for a Decision Maker submitting their preferred candidate index."""

    dm_id: str = Field(..., description="Username/ID of the voting DM (e.g., 'dm1').")
    vote_idx: int = Field(..., ge=0, description="0-indexed candidate chosen from fair_solutions.")


class FavoriteSessionState(BaseModel):
    """Full snapshot of an ongoing Favorite method GDM session."""

    session_id: str
    problem_id: int
    dm_ids: list[str]
    current_iteration: int
    max_iterations: int
    phase: Literal["consensus_reaching", "decision"] = "consensus_reaching"
    status: Literal["voting", "revote_pending", "ready_for_iteration", "completed"] = "voting"
    options: FavOptions
    results_history: list[FavResults]
    current_votes: dict[str, int] = Field(default_factory=dict)
    candidates: list[FairSolution] = Field(default_factory=list)
    tie_state: dict | None = None
    final_solution: FairSolution | None = None
    current_most_preferred_solutions: dict[str, dict[str, float]] | None = None
    mps_history: list[dict[str, dict[str, float]]] = Field(default_factory=list)
    mps_adjustments_history: list[dict[str, Any]] = Field(default_factory=list)


class FavoriteSessionDB(SQLModel, table=True):
    """Database table for storing Favorite Method sessions."""

    id: int | None = SQLField(default=None, primary_key=True)
    session_id: str = SQLField(index=True, unique=True, description="The UUID of the session.")
    problem_id: int = SQLField(foreign_key="problemdb.id", description="Links to DESDEO's ProblemDB.")

    # Store the entire FavoriteSessionState Pydantic object as a JSON blob
    state_data: dict[str, Any] = SQLField(default_factory=dict, sa_column=Column(JSON))
