import numpy as np
import numpy.testing as npt
import pytest

from fairness import PUtility, UF, solve_UFs, scale_rp


@pytest.mark.fairness
def test_solve_UFs():
    X = np.array([
        [1, 0],
        [0, 1],
        [0.5, 0.5],
        [0.75, 0.25],
        [0.25, 0.75],
    ])
    P = np.array([
        [0.75, 0.25],
        [1, 0],
    ])

    rw = 0.
    UF_vals, UF_ws = solve_UFs(X, P, rw, None, False)
    print(UF_vals)
    print(UF_ws)

    assert len(UF_vals) == 10
    assert len(UF_ws) == 5
