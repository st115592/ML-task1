"""Integration checks for generated, independently reproducible artifacts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from servo_model import COLUMNS, coefficients, metrics

ROOT = Path(__file__).resolve().parent


def test_saved_results_match_locked_test() -> None:
    """Saved metrics must equal predictions on exactly the locked row IDs."""
    if not (ROOT / "results/model.joblib").exists():
        pytest.skip("Run run_analysis.py to generate experiment artifacts")
    split = json.loads((ROOT / "results/split.json").read_text(encoding="utf-8"))
    train, test = split["train_row_ids"], split["test_row_ids"]
    assert len(train) == 133 and len(test) == 34
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(167))
    model = joblib.load(ROOT / "results/model.joblib")
    assert model.func is None  # strict linearity in original target units
    data = pd.read_csv(ROOT / "data/raw/servo.data", names=COLUMNS + ["y"])
    x = data.iloc[test][COLUMNS]
    prediction = model.predict(x)
    beta = coefficients(model)
    design = model.regressor_[:2].transform(x)
    np.testing.assert_allclose(
        prediction, beta.iloc[0] + design @ beta.iloc[1:], rtol=1e-9, atol=1e-9
    )
    chosen = json.loads(
        (ROOT / "results/selection_frozen.json").read_text(encoding="utf-8")
    )
    table = pd.read_csv(ROOT / "results/metrics.csv")
    actual = table[
        (table.family == chosen["family"])
        & (table.target == "raw")
        & (table.split == "test")
    ].iloc[0]
    expected = metrics(data.iloc[test].y.to_numpy(), prediction)
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value)
    oof = pd.read_csv(ROOT / "results/selected_oof_predictions.csv", index_col=0)
    assert set(oof.index) == set(train)
    assert oof.shape == (133, 3)
    assert np.isfinite(oof.to_numpy()).all()
