"""Linear-in-coefficients bases and validation for UCI Servo.

No dimensional physical constants are inferred from undocumented gain scales.
PolynomialFeatures documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class Config:
    """Explicit numerical reference scales in dataset units, NOT SI values."""

    p_ref: float = 1.0
    v_ref: float = 1.0
    y_ref: float = 1.0
    seed: int = 42
    outer_folds: int = 5
    outer_repeats: int = 3
    inner_folds: int = 4
    bootstrap_repeats: int = 300


CONFIG = Config()
NUM = ["pgain", "vgain"]
CAT = ["motor", "screw"]
COLUMNS = CAT + NUM


def validate_x(x: pd.DataFrame) -> None:
    """Check array dimensionality and the documented categorical schema."""
    if not isinstance(x, pd.DataFrame) or list(x.columns) != COLUMNS:
        raise ValueError(f"Expected a DataFrame with columns {COLUMNS}")
    if x.empty or x.isna().any().any():
        raise ValueError("Empty or missing input")
    if not np.isfinite(x[NUM].to_numpy(dtype=float)).all():
        raise ValueError("Gains must be finite")
    if (x[NUM] <= 0).any().any():
        raise ValueError("Positive gain scales required for reciprocal basis")
    if not all(x[c].isin(list("ABCDE")).all() for c in CAT):
        raise ValueError("Unknown mechanical configuration")


def validate_y(y: np.ndarray) -> None:
    """A positive finite 1-D target is required, including for log comparison."""
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Target must be a finite positive 1-D vector")


class GainBasis(TransformerMixin, BaseEstimator):
    """Dimensionless features; physical basis is a hypothesis, not a plant law."""

    def __init__(
        self, mode: str = "raw", p_ref: float = 1.0, v_ref: float = 1.0
    ) -> None:
        self.mode = mode
        self.p_ref = p_ref
        self.v_ref = v_ref

    def fit(self, x: Any, y: Any = None) -> "GainBasis":
        self.transform(x)
        self.n_features_in_ = 2
        return self

    def transform(self, x: Any) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("Expected (n, 2) gains")
        if self.p_ref <= 0 or self.v_ref <= 0:
            raise ValueError("Reference scales must be positive")
        if not np.isfinite(a).all() or (a <= 0).any():
            raise ValueError("Gains must be positive and finite")
        p, v = a[:, 0] / self.p_ref, a[:, 1] / self.v_ref
        if self.mode == "physical":
            return np.column_stack((1.0 / p, v / p))
        if self.mode != "raw":
            raise ValueError("Unknown basis")
        return np.column_stack((p, v))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = ["1/p", "v/p"] if self.mode == "physical" else ["p", "v"]
        return np.array(names, dtype=object)


class CategoryGainInteractions(TransformerMixin, BaseEstimator):
    """Optional total-degree-2 category*gain terms; no ordered letter codes."""

    def __init__(self, enabled: bool = False, exclude_product: bool = False) -> None:
        self.enabled = enabled
        self.exclude_product = exclude_product

    def fit(self, x: Any, y: Any = None) -> "CategoryGainInteractions":
        self.n_features_in_ = np.asarray(x).shape[1]
        return self

    def transform(self, x: Any) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        if a.ndim != 2 or a.shape[1] < 2:
            raise ValueError("Expected a 2-D encoded design matrix")
        if self.exclude_product:
            if self.enabled or a.shape[1] != 13:
                raise ValueError(
                    "Product ablation requires a 13-column quadratic basis"
                )
            return np.delete(a, 3, axis=1)
        if not self.enabled:
            return a
        # First five columns: p,v,p^2,p*v,v^2; remaining eight: A-reference dummies.
        if a.shape[1] != 13:
            raise ValueError("Interactions require quadratic basis with 13 columns")
        return np.column_stack((a, a[:, 5:] * a[:, [0]], a[:, 5:] * a[:, [1]]))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        n = np.asarray(input_features, dtype=object)
        if self.exclude_product:
            return np.delete(n, 3)
        if not self.enabled:
            return n
        return np.concatenate((n, [f"{c}*p" for c in n[5:]], [f"{c}*v" for c in n[5:]]))


def log_target(y: np.ndarray) -> np.ndarray:
    """log of a dimensionless target ratio; y_ref is explicitly documented."""
    return np.log(y / CONFIG.y_ref)


def inverse_log_target(z: np.ndarray) -> np.ndarray:
    """Back-transform without claiming unbiased conditional-mean prediction."""
    return CONFIG.y_ref * np.exp(z)


def make_model(family: str, log: bool = False) -> TransformedTargetRegressor:
    """All learned preprocessing lives inside the fitted sklearn pipeline."""
    numeric = Pipeline(
        [
            ("basis", GainBasis(p_ref=CONFIG.p_ref, v_ref=CONFIG.v_ref)),
            ("poly", PolynomialFeatures(degree=1, include_bias=False)),
        ]
    )
    encoder = OneHotEncoder(
        categories=[list("ABCDE"), list("ABCDE")],
        drop="first",
        sparse_output=False,
        handle_unknown="error",
    )
    pre = ColumnTransformer(
        [("gain", numeric, NUM), ("cat", encoder, CAT)], verbose_feature_names_out=False
    )
    estimators = {
        "baseline": DummyRegressor(strategy="mean"),
        "ols": LinearRegression(),
        "physical": LinearRegression(),
        "poly2": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(max_iter=100000, tol=1e-6),
        "elastic": ElasticNet(max_iter=100000, tol=1e-6),
    }
    pipe = Pipeline(
        [
            ("pre", pre),
            ("cross", CategoryGainInteractions()),
            ("scale", StandardScaler()),
            ("reg", estimators[family]),
        ]
    )
    if family == "physical":
        pipe.set_params(pre__gain__basis__mode="physical")
    if family == "poly2":
        pipe.set_params(pre__gain__poly__degree=2)
    return TransformedTargetRegressor(
        regressor=pipe,
        func=log_target if log else None,
        inverse_func=inverse_log_target if log else None,
    )


def grids(family: str) -> list[dict[str, Any]]:
    """Finite predeclared grids; polynomial degree never exceeds two."""
    prefix = "regressor__"
    if family in ("baseline", "ols", "physical"):
        return [{}]
    if family == "poly2":
        return [{prefix + "cross__enabled": [False, True]}]
    regularization = {prefix + "reg__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    if family == "elastic":
        regularization[prefix + "reg__l1_ratio"] = [0.2, 0.8]
    return [
        dict(regularization, **{prefix + "pre__gain__poly__degree": [1, 2]}),
        dict(
            regularization,
            **{
                prefix + "pre__gain__poly__degree": [2],
                prefix + "cross__enabled": [True],
            },
        ),
        dict(regularization, **{prefix + "pre__gain__basis__mode": ["physical"]}),
    ]


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Errors have original target units; R2 is dimensionless."""
    if np.asarray(y).ndim != 1 or np.asarray(prediction).shape != np.asarray(y).shape:
        raise ValueError("Metric inputs must be aligned 1-D vectors")
    return {
        "MAE": float(mean_absolute_error(y, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
        "R2": float(r2_score(y, prediction)),
    }


def coefficients(model: TransformedTargetRegressor) -> pd.Series:
    """Undo StandardScaler to express coefficients in original basis units."""
    pipe = model.regressor_
    names = pipe[:2].get_feature_names_out()
    scale, reg = pipe["scale"], pipe["reg"]
    beta = np.asarray(reg.coef_) / scale.scale_
    intercept = float(reg.intercept_ - np.dot(beta, scale.mean_))
    return pd.Series(np.r_[intercept, beta], index=["intercept", *names])
