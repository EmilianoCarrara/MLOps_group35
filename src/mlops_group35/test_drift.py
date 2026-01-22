import pandas as pd
from mlops_group35.drift import psi


def test_psi_is_small_for_same_distribution():
    s = pd.Series([1, 2, 3, 4, 5] * 50)
    assert psi(s, s) < 0.05
