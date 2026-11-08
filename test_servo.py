"""Limit cases, dimensions, category semantics and leakage protection."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from servo_model import COLUMNS, CategoryGainInteractions, GainBasis, coefficients
from servo_model import inverse_log_target, log_target, make_model, metrics
from servo_model import validate_x, validate_y


@pytest.fixture
def inputs() -> pd.DataFrame:
    """A full-rank-ish synthetic design with all documented categories."""
    rng = np.random.default_rng(13)
    return pd.DataFrame(
        {
            "motor": rng.choice(list("ABCDE"), 100),
            "screw": rng.choice(list("ABCDE"), 100),
            "pgain": rng.uniform(3, 6, 100),
            "vgain": rng.uniform(1, 5, 100),
        }
    )[COLUMNS]


def test_reference_scales_and_reciprocal_limits() -> None:
    basis = GainBasis("physical", p_ref=2, v_ref=4)
    np.testing.assert_allclose(basis.fit_transform([[2, 4]]), [[1, 1]])
    limiting = GainBasis("physical").fit_transform([[1e12, 1], [1e-12, 1]])
    assert limiting[0, 0] == pytest.approx(1e-12)
    assert limiting[1, 0] == pytest.approx(1e12)
    np.testing.assert_allclose(GainBasis().fit_transform([[3, 4]]), [[3, 4]])


@pytest.mark.parametrize("values", [[[0, 1]], [[-1, 1]], [[1, np.inf]], [1, 2]])
def test_invalid_gain_dimensions(values: list) -> None:
    with pytest.raises(ValueError):
        GainBasis("physical").fit_transform(values)


def test_validation_and_log_roundtrip(inputs: pd.DataFrame) -> None:
    validate_x(inputs)
    validate_y(np.array([0.1, 1.0, 7.0]))
    y = np.array([0.1, 1.0, 7.0])
    np.testing.assert_allclose(inverse_log_target(log_target(y)), y)
    assert log_target(np.array([1.0]))[0] == 0
    with pytest.raises(ValueError):
        validate_y(np.array([0.0]))
    with pytest.raises(ValueError):
        validate_y(np.ones((2, 1)))
    bad = inputs.copy()
    bad.loc[0, "motor"] = "F"
    with pytest.raises(ValueError):
        validate_x(bad)


@pytest.mark.parametrize(
    "family,n_features", [("ols", 10), ("physical", 10), ("poly2", 13)]
)
def test_basis_dimensions_and_linear_formula(
    inputs: pd.DataFrame, family: str, n_features: int
) -> None:
    y = 1 + inputs.pgain.to_numpy()
    fitted = make_model(family).fit(inputs, y)
    design = fitted.regressor_[:2].transform(inputs)
    assert design.shape == (100, n_features)
    beta = coefficients(fitted)
    np.testing.assert_allclose(
        fitted.predict(inputs),
        beta.iloc[0] + design @ beta.iloc[1:].to_numpy(),
        rtol=1e-8,
        atol=1e-8,
    )


def test_interaction_and_ablation_limit() -> None:
    a = np.zeros((2, 13))
    a[:, :2] = [[3, 2], [4, 3]]
    assert CategoryGainInteractions(True).fit_transform(a).shape == (2, 29)
    np.testing.assert_allclose(
        CategoryGainInteractions(True).fit_transform(a)[:, 13:], 0
    )
    assert CategoryGainInteractions(exclude_product=True).fit_transform(a).shape == (
        2,
        12,
    )
    with pytest.raises(ValueError):
        CategoryGainInteractions(True).fit_transform(np.zeros((2, 10)))


def test_fit_only_train_and_clone(inputs: pd.DataFrame) -> None:
    model = make_model("ridge")
    first, second = clone(model), clone(model)
    first.fit(inputs.iloc[:60], np.ones(60))
    second.fit(inputs.iloc[60:], np.ones(40))
    raw_train = first.regressor_[:2].transform(inputs.iloc[:60])
    np.testing.assert_allclose(first.regressor_["scale"].mean_, raw_train.mean(axis=0))
    before = first.regressor_["scale"].mean_.copy()
    first.predict(inputs.iloc[60:])
    np.testing.assert_array_equal(before, first.regressor_["scale"].mean_)
    np.testing.assert_allclose(first.predict(inputs), 1)


def test_metrics_exact_and_mismatch() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert metrics(y, y) == {"MAE": 0.0, "RMSE": 0.0, "R2": 1.0}
    assert metrics(y, np.full(3, 2.0))["R2"] == 0
    with pytest.raises(ValueError):
        metrics(y, y.reshape(3, 1))
