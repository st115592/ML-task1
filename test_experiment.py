"""Regression tests for identical tuning and experiment-level test isolation."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import run_analysis as analysis
from servo_model import COLUMNS, coefficients


def test_production_tuning_configuration() -> None:
    """Pin the production scheme, including the seed and one-SE denominator."""
    first, second = analysis.tuning_cv(), analysis.tuning_cv()
    assert first is not second
    assert first.cvargs == second.cvargs == {"n_splits": 5}
    assert first.n_repeats == second.n_repeats == 3
    assert first.random_state == second.random_state == 142
    assert first.get_n_splits() == second.get_n_splits() == 15
    for (a, b), (c, d) in zip(first.split(np.zeros(106)), second.split(np.zeros(106))):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)


def test_whole_experiment_ignores_test_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run main twice with real training, CV, export and diagnostics.

    Only computational budgets/grids are reduced. The orchestration, one-SE
    selection and train/test boundaries are the production code paths.
    """
    data = pd.read_csv(analysis.ROOT / "data/raw/servo.data", names=COLUMNS + ["y"])
    train, test = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=analysis.CONFIG.seed
    )
    changed = data.copy()
    changed.loc[test, "y"] = changed.loc[test, "y"] * 17 + 5
    np.testing.assert_array_equal(data.iloc[train].y, changed.iloc[train].y)
    monkeypatch.setattr(
        analysis,
        "CONFIG",
        replace(
            analysis.CONFIG,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            inner_repeats=1,
            bootstrap_repeats=2,
        ),
    )
    original_grids = analysis.grids

    def reduced_grids(family: str) -> list[dict[str, Any]]:
        result = original_grids(family)
        return [
            {
                key: value[:1] if key.endswith(("alpha", "l1_ratio")) else value
                for key, value in grid.items()
            }
            for grid in result
        ]

    monkeypatch.setattr(analysis, "grids", reduced_grids)
    monkeypatch.setattr(analysis, "write_report", lambda *args: None)
    observed_cv = []
    original_search = analysis.search

    def observed_search(
        family: str, log: bool, x: pd.DataFrame, y: np.ndarray, cv: Any
    ) -> Any:
        observed_cv.append((cv.cvargs["n_splits"], cv.n_repeats, cv.random_state))
        return original_search(family, log, x, y, cv)

    monkeypatch.setattr(analysis, "search", observed_search)
    saved = []
    for index, frame in enumerate((data, changed)):
        root = tmp_path / str(index)
        (root / "data/raw").mkdir(parents=True)
        frame.to_csv(root / "data/raw/servo.data", header=False, index=False)
        monkeypatch.setattr(analysis, "ROOT", root)
        monkeypatch.setattr(analysis, "RESULTS", root / "results")
        monkeypatch.setattr(analysis, "FIGURES", root / "results/figures")
        analysis.main()
        manifest = json.loads((root / "results/selection_frozen.json").read_text())
        model = joblib.load(root / "results/model.joblib")
        saved.append(
            (
                manifest,
                coefficients(model),
                pd.read_csv(root / "results/nested_selection.csv"),
            )
        )
    assert len(observed_cv) > 2
    assert set(observed_cv) == {(2, 1, 142)}  # every outer, final and regime search
    assert saved[0][0] == saved[1][0]
    pd.testing.assert_series_equal(saved[0][1], saved[1][1], check_exact=True)
    pd.testing.assert_frame_equal(saved[0][2], saved[1][2], check_exact=True)
