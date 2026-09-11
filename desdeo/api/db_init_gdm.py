"""This module initializes the database."""

import warnings

import numpy as np
from sqlalchemy_utils import database_exists
from sqlmodel import Session, SQLModel

from desdeo.api.config import SettingsConfig
from desdeo.api.db import engine
from desdeo.api.models import (
    ProblemDB,
    User,
    UserRole,
)
from desdeo.api.models.gdm.gdm_aggregate import Group
from desdeo.api.routers.user_authentication import get_password_hash
from desdeo.problem.testproblems import (
    dmitry_forest_problem_disc,
    dtlz2,
    metallurgical_application_discrete,
    re34,
    river_pollution_problem_discrete,
)

problems = [
    river_pollution_problem_discrete(five_objective_variant=False),
    dtlz2(10, 3),
    dmitry_forest_problem_disc(),
    metallurgical_application_discrete(),
    re34(),
]

num_analysts = 1
num_dms = 5

usernames_analyst = [f"analyst{i + 1}" for i in range(num_analysts)]
usernames_dm = [f"dm{i + 1}" for i in range(num_dms)]

user_owner = User(
    id="1",
    username=usernames_analyst[0],
    password_hash=get_password_hash("12345"),
    role=UserRole.analyst,
    group="test",
)

id_user = 1
if __name__ == "__main__":
    if SettingsConfig.debug:
        # debug stuff

        print("Creating database tables.")  # noqa: T201
        if not database_exists(engine.url):
            SQLModel.metadata.create_all(engine)
        else:
            warnings.warn("Database already exists. Clearing it.", stacklevel=1)
            # Drop all tables
            SQLModel.metadata.reflect(bind=engine)
            SQLModel.metadata.drop_all(bind=engine)
            SQLModel.metadata.create_all(engine)
        print("Database tables created.")  # noqa: T201

        with Session(engine) as session:
            for user in usernames_analyst:
                user_analyst = User(
                    id=str(id_user),
                    username=user,
                    password_hash=get_password_hash("12345"),
                    role=UserRole.analyst,
                    group="test",
                    group_ids=[1, 2, 3, 4, 5],
                )
                session.add(user_analyst)
                session.commit()
                session.refresh(user_analyst)
                id_user += 1

            dm_ids = []
            three_dms_cutoff = 3
            for idx, user in enumerate(usernames_dm):
                # dm1-dm3 participate in groups 1-5 (3, 4, 5-obj problems)
                # dm4 participates in groups 1, 3, 4 (4, 5-obj problems)
                # dm5 participates in group 4 (5-obj problem)
                if idx < three_dms_cutoff:
                    assigned_groups = [1, 2, 3, 4, 5]
                elif idx == three_dms_cutoff:
                    assigned_groups = [1, 3, 4]
                else:
                    assigned_groups = [4]

                user_dm = User(
                    id=str(id_user),
                    username=user,
                    password_hash=get_password_hash("12345"),
                    role=UserRole.dm,
                    group="test",
                    group_ids=assigned_groups,
                )
                session.add(user_dm)
                session.commit()
                session.refresh(user_dm)
                dm_ids.append(id_user)
                id_user += 1

            rng = np.random.default_rng(seed=42)

            for problem in problems:
                # Add the problem to the analyst1
                problem_db = ProblemDB.from_problem(problem, user_owner)
                session.add(problem_db)
                session.commit()
                session.refresh(problem_db)

            # Create a group 1 for River Pollution (4 objectives -> 4 DMs)
            group_name = "tingalinga"
            group = Group(
                name=group_name,
                owner_id=1,
                user_ids=dm_ids[:4],
                problem_id=1,
            )

            group.model_rebuild()

            session.add(group)
            session.commit()
            session.refresh(group)

            # Create a group 2 for DTLZ2 (3 objectives -> 3 DMs)
            group_name2 = "secondtingalinga"
            group2 = Group(
                name=group_name2,
                owner_id=1,
                user_ids=dm_ids[:3],
                problem_id=2,
            )

            group2.model_rebuild()

            session.add(group2)
            session.commit()
            session.refresh(group2)

            # Create group 3 for Dmitry Forest Problem (Discrete) (4 objectives -> 4 DMs)
            group_name3 = "forest_group"
            group3 = Group(
                name=group_name3,
                owner_id=1,
                user_ids=dm_ids[:4],
                problem_id=3,
            )

            group3.model_rebuild()

            session.add(group3)
            session.commit()
            session.refresh(group3)

            # Create group 4 for Metallurgical Application Problem (Discrete) (5 objectives -> 5 DMs)
            group_name4 = "metallurgical_group"
            group4 = Group(
                name=group_name4,
                owner_id=1,
                user_ids=dm_ids[:5],
                problem_id=4,
            )

            group4.model_rebuild()

            session.add(group4)
            session.commit()
            session.refresh(group4)

            # Create group 5 for RE34 Vehicle Crashworthiness (3 objectives -> 3 DMs)
            group_name5 = "re34_group"
            group5 = Group(
                name=group_name5,
                owner_id=1,
                user_ids=dm_ids[:3],
                problem_id=5,
            )

            group5.model_rebuild()

            session.add(group5)
            session.commit()
            session.refresh(group5)

            session.close()

    else:
        # deployment stuff
        pass
