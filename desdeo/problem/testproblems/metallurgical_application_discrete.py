"""A metallurgical application problem with discrete representation."""

import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from desdeo.problem.schema import (
    DiscreteRepresentation,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    Variable,
    VariableTypeEnum,
)


def metallurgical_application_discrete() -> Problem:
    """Defines the microalloyed steel design problem (MOP_II) using Pareto optimal front representation.

    The 2000 non-dominated solutions are loaded from the archived EMO runs in metallfronts.zip.
    Five objectives are considered:
      1. Yield strength (YS, MPa) - maximized
      2. Ultimate tensile strength (UTS, MPa) - maximized
      3. Elongation (ELON, %) - maximized
      4. Carbon equivalent (CE) - minimized
      5. Material cost (COST, USD/kg) - minimized

    Returns:
        Problem: A problem instance representing the discrete metallurgical application problem.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    zip_path = repo_root / "experiment/results/metallurgical_problem/metallfronts.zip"

    with zipfile.ZipFile(zip_path) as z, z.open("metallappl_mop2_m5.npz") as f:
        data = np.load(f)
        pts = data["points"].copy()

    # Invert maximized objectives back to true problem values
    pts[:, 0] = -pts[:, 0]  # YS
    pts[:, 1] = -pts[:, 1]  # UTS
    pts[:, 2] = -pts[:, 2]  # ELON
    # CE (pts[:, 3]) and COST (pts[:, 4]) are minimized analytical values

    cols = ["YS", "UTS", "ELON", "CE", "COST"]
    obj_definitions = [
        ("Yield strength", "YS", True),
        ("Ultimate tensile strength", "UTS", True),
        ("Elongation", "ELON", True),
        ("Carbon equivalent", "CE", False),
        ("Material cost", "COST", False),
    ]

    df = pl.DataFrame(pts, schema=cols)

    var_name = "index"
    variables = [
        Variable(
            name=var_name,
            symbol=var_name,
            variable_type=VariableTypeEnum.integer,
            lowerbound=0,
            upperbound=len(df) - 1,
            initial_value=0,
        )
    ]

    objectives = [
        Objective(
            name=name,
            symbol=symbol,
            objective_type=ObjectiveTypeEnum.data_based,
            ideal=float(df[symbol].max() if maximize else df[symbol].min()),
            nadir=float(df[symbol].min() if maximize else df[symbol].max()),
            maximize=maximize,
        )
        for name, symbol, maximize in obj_definitions
    ]

    discrete_def = DiscreteRepresentation(
        variable_values={"index": list(range(len(df)))},
        objective_values=df[cols].to_dict(),
    )

    return Problem(
        name="Metallurgical Application Problem (Discrete)",
        description=(
            "Defines the microalloyed steel design problem (MOP_II) from Saini et al. (2023) "
            "represented as a discrete set of 2,000 Pareto optimal solutions with 5 objectives."
        ),
        variables=variables,
        objectives=objectives,
        discrete_representation=discrete_def,
        is_twice_differentiable=False,
    )
