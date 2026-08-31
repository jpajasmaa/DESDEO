"""Production database initialisation script with GDM SCORE bands setup.

Run once as a Kubernetes Job after the first deployment (or after a full
database wipe). It is intentionally idempotent: running it multiple times
against the same database is safe.

What it does
------------
1. Creates all SQLModel tables if they do not already exist.
   (Uses create_all which is a no-op for tables that are present.)
2. Seeds an initial analyst user whose credentials come from env vars.
   If the user already exists the step is skipped.

Environment variables required
-------------------------------
DATABASE_URL          PostgreSQL DSN, e.g.
                      postgresql://desdeo:<pw>@desdeo-postgres:5432/desdeo
DESDEO_ADMIN_USERNAME Username for the seeded analyst account.
DESDEO_ADMIN_PASSWORD Password for the seeded analyst account.

Optional
--------
DESDEO_ADMIN_GROUP    Group name for the seeded user (default: "admin").
"""

import os
import sys

from sqlmodel import Session, SQLModel, select

# Import the engine after DATABASE_URL is in the environment so the config
# module picks it up correctly.
from desdeo.api.db import engine
from desdeo.api.models import ProblemDB, User, UserRole
from desdeo.api.models.gdm.gdm_aggregate import Group, GroupSessionDB
from desdeo.api.routers.user_authentication import get_password_hash
from desdeo.problem.testproblems import river_pollution_problem_discrete


def create_tables() -> None:
    print("[db-init] Creating database tables (create_all is a no-op for existing tables)...")
    SQLModel.metadata.create_all(engine)
    print("[db-init] Tables ready.")


def seed_admin_user() -> None:
    username = os.environ.get("DESDEO_ADMIN_USERNAME")
    password = os.environ.get("DESDEO_ADMIN_PASSWORD")
    group = os.environ.get("DESDEO_ADMIN_GROUP", "admin")

    if not username or not password:
        print("[db-init] WARNING: DESDEO_ADMIN_USERNAME or DESDEO_ADMIN_PASSWORD not set — skipping user seed.")
        return

    with Session(engine) as session:
        existing = session.exec(select(User).where(User.username == username)).first()

        if existing:
            print(f"[db-init] User '{username}' already exists — skipping.")
            return

        user = User(
            username=username,
            password_hash=get_password_hash(password),
            role=UserRole.analyst,
            group=group,
        )
        session.add(user)
        session.commit()
        print(f"[db-init] Created user '{username}' (role=analyst, group={group}).")


def set_scorebands_data() -> None:
    """Sets the specific GDM SCORE bands test users, problem, and group."""
    with Session(engine) as session:
        # Idempotency check: if analyst1 exists, assume data is already seeded
        if session.exec(select(User).where(User.username == "analyst1")).first():
            print("[db-init] SCORE bands test data already exists — skipping.")
            return

        print("[db-init] Seeding SCORE bands test data...")

        # 1. Create Users
        analyst1 = User(
            username="analyst1", password_hash=get_password_hash("12345"), role=UserRole.analyst, group="test"
        )
        dm1 = User(username="dm1", password_hash=get_password_hash("12345"), role=UserRole.dm, group="test")
        dm2 = User(username="dm2", password_hash=get_password_hash("12345"), role=UserRole.dm, group="test")

        session.add_all([analyst1, dm1, dm2])
        session.commit()
        session.refresh(analyst1)
        session.refresh(dm1)
        session.refresh(dm2)

        # 2. Create Problem
        problem = river_pollution_problem_discrete(five_objective_variant=False)
        problem_db = ProblemDB.from_problem(problem, analyst1)

        session.add(problem_db)
        session.commit()
        session.refresh(problem_db)

        # 3. Create Group
        group = Group(name="tingalinga", owner_id=analyst1.id, users=[dm1, dm2])
        session.add(group)
        session.commit()
        session.refresh(group)

        # 4. Create Group Session
        group_session = GroupSessionDB(
            group_id=group.id,
            problem_id=problem_db.id,
            method="gdm-score-bands",
            head_iteration_id=None,
        )
        session.add(group_session)
        session.commit()

        print(f"[db-init] Created SCORE Bands group session for group '{group.name}'.")


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("[db-init] ERROR: DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"[db-init] Using database: {database_url.split('@')[-1]}")  # hide credentials
    create_tables()
    seed_admin_user()
    set_scorebands_data()
    print("[db-init] Done.")


if __name__ == "__main__":
    main()
