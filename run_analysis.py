"""Reproducible nested CV, locked tests, diagnostics and Russian report."""

import argparse
import hashlib
import json
import platform
import zipfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedKFold,
    train_test_split,
    cross_validate,
)

from servo_model import CONFIG, COLUMNS, coefficients, grids, make_model, metrics
from servo_model import validate_x, validate_y

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FAMILIES = ["baseline", "ols", "physical", "poly2", "ridge", "lasso", "elastic"]


def tuning_cv() -> RepeatedKFold:
    """Identical deterministic tuning procedure for outer-train and final fit.

    Both hyperparameter search and the one-SE rule see the same number of
    folds. Each call creates a fresh splitter; indices are local to its input.
    """
    return RepeatedKFold(
        n_splits=CONFIG.inner_folds,
        n_repeats=CONFIG.inner_repeats,
        random_state=CONFIG.seed + 100,
    )


def dump_json(path: Path, data: Any) -> None:
    """Write structured metadata without losing non-ASCII labels."""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def export_test_predictions(model: Any, x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Export unchanged locked-test predictions for configuration-level audit."""
    frame = x.copy()
    frame["y"] = y
    frame["prediction"] = model.predict(x)
    frame["residual"] = y - frame.prediction
    frame["nonpositive_time"] = frame.prediction <= 0
    frame.to_csv(RESULTS / "test_predictions.csv", index_label="row_id")
    return frame


def load_data(archive: Path | None = None) -> pd.DataFrame:
    """Extract only the known data files, keeping the original bytes intact."""
    folder = ROOT / "data" / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    if archive is not None:
        with zipfile.ZipFile(archive) as source:
            for name in ("servo.data", "servo.names", "Index"):
                (folder / name).write_bytes(source.read(name))
    data = pd.read_csv(folder / "servo.data", names=COLUMNS + ["y"])
    validate_x(data[COLUMNS])
    validate_y(data.y.to_numpy())
    if len(data) != 167 or data[COLUMNS].duplicated().any():
        raise ValueError("Expected 167 distinct Servo configurations")
    return data


def search(
    family: str, log: bool, x: pd.DataFrame, y: np.ndarray, cv: Any
) -> GridSearchCV:
    """Fit preprocessing and tune hyperparameters on training folds only."""
    fit = GridSearchCV(
        make_model(family, log),
        grids(family),
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
    )
    return fit.fit(x, y)


def choose_raw(fits: dict[str, GridSearchCV]) -> str:
    """One-standard-error heuristic favours a lower-dimensional raw basis.

    Repeated folds are dependent: this is a complexity heuristic, not a CI.
    Hyperparameters within each family are still selected by minimum MAE.
    """
    best = max(fits, key=lambda f: fits[f].best_score_)
    winner = fits[best]
    se = winner.cv_results_["std_test_score"][winner.best_index_] / np.sqrt(
        winner.n_splits_
    )
    eligible = [f for f in fits if fits[f].best_score_ >= winner.best_score_ - se]

    def dimension(family: str) -> tuple[int, float, int]:
        if family == "baseline":
            n = 0
        else:
            n = len(fits[family].best_estimator_.regressor_[:2].get_feature_names_out())
        return n, -fits[family].best_score_, FAMILIES.index(family)

    return min(eligible, key=dimension)


def paired_hypotheses(data: pd.DataFrame) -> dict[str, Any]:
    """Matched finite differences avoid confounding in the H1 direction check."""
    slopes = []
    for _, group in data.groupby(["motor", "screw", "vgain"]):
        ordered = group.sort_values("pgain")
        if len(ordered) > 1:
            slopes.extend((np.diff(ordered.y) / np.diff(ordered.pgain)).tolist())
    a = np.asarray(slopes)
    return {
        "matched_differences": len(a),
        "negative_fraction": float((a < 0).mean()),
        "zero_fraction": float(np.isclose(a, 0).mean()),
        "median_slope": float(np.median(a)),
    }


def bootstrap(
    model: Any, x: pd.DataFrame, y: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired row bootstrap, fixed basis/tuning; OLS comparator has same basis."""
    rng = np.random.default_rng(CONFIG.seed)
    regular, unregular = [], []
    for _ in range(CONFIG.bootstrap_repeats):
        idx = rng.integers(0, len(x), len(x))
        for use_ols, collection in ((False, regular), (True, unregular)):
            fitted = clone(model)
            if use_ols:
                fitted.set_params(regressor__reg=LinearRegression())
            fitted.fit(x.iloc[idx], y[idx])
            collection.append(coefficients(fitted))

    def summarize(rows: list[pd.Series]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        return pd.DataFrame(
            {
                "median": frame.median(),
                "q025": frame.quantile(0.025),
                "q975": frame.quantile(0.975),
                "positive_fraction": (frame > 1e-9).mean(),
                "negative_fraction": (frame < -1e-9).mean(),
                "zero_fraction": (frame.abs() <= 1e-9).mean(),
            }
        )

    return summarize(regular), summarize(unregular)


def main(archive: Path | None = None) -> None:
    """Run the predeclared experiment and generate all artifacts."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_data(archive)
    x, y = data[COLUMNS], data.y.to_numpy()
    train, test = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=CONFIG.seed
    )
    xt, yt = x.iloc[train], y[train]
    xv, yv = x.iloc[test], y[test]
    dump_json(
        RESULTS / "split.json",
        {"train_row_ids": train.tolist(), "test_row_ids": test.tolist()},
    )
    dump_json(
        RESULTS / "data_audit.json",
        {
            "sha256_servo_data": hashlib.sha256(
                (ROOT / "data/raw/servo.data").read_bytes()
            ).hexdigest(),
            "n": len(data),
            "missing": int(data.isna().sum().sum()),
            "duplicates": int(data.duplicated().sum()),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "train_hypotheses": paired_hypotheses(data.iloc[train]),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    )
    outer = RepeatedKFold(
        n_splits=CONFIG.outer_folds,
        n_repeats=CONFIG.outer_repeats,
        random_state=CONFIG.seed,
    )
    keys = [(f, log) for f in FAMILIES for log in (False, True)]
    oof = {key: np.full((CONFIG.outer_repeats, len(train)), np.nan) for key in keys}
    selected_oof = np.full((CONFIG.outer_repeats, len(train)), np.nan)
    fold_rows, selection_rows = [], []
    for fold, (a, b) in enumerate(outer.split(xt)):
        inner = tuning_cv()
        raw = {}
        for family, log in keys:
            fit = search(family, log, xt.iloc[a], yt[a], inner)
            prediction = fit.predict(xt.iloc[b])
            oof[(family, log)][fold // CONFIG.outer_folds, b] = prediction
            fold_rows.append(
                {
                    "fold": fold,
                    "family": family,
                    "target": "log" if log else "raw",
                    **metrics(yt[b], prediction),
                    "inner_MAE": -fit.best_score_,
                    "params": json.dumps(fit.best_params_),
                }
            )
            if not log:
                raw[family] = fit
        selected = choose_raw(raw)
        selected_prediction = raw[selected].predict(xt.iloc[b])
        selected_oof[fold // CONFIG.outer_folds, b] = selected_prediction
        selection_rows.append(
            {"fold": fold, "family": selected, **metrics(yt[b], selected_prediction)}
        )
        print(f"Outer fold {fold + 1}/15 complete; selected={selected}", flush=True)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(RESULTS / "nested_cv_folds.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(RESULTS / "nested_selection.csv", index=False)
    repeat_rows = []
    for key, array in oof.items():
        for repeat in range(CONFIG.outer_repeats):
            repeat_rows.append(
                {
                    "family": key[0],
                    "target": "log" if key[1] else "raw",
                    "repeat": repeat,
                    **metrics(yt, array[repeat]),
                }
            )
    pd.DataFrame(repeat_rows).to_csv(RESULTS / "cv_repeat_metrics.csv", index=False)
    pd.DataFrame(selected_oof.T, index=train).to_csv(
        RESULTS / "selected_oof_predictions.csv"
    )

    # No test predictions are made before ALL development searches are complete.
    final_cv = tuning_cv()
    final = {(f, log): search(f, log, xt, yt, final_cv) for f, log in keys}
    ablation_rows = []
    for label, exclude, cross in (
        ("squares_without_pv", True, False),
        ("quadratic_with_pv", False, False),
        ("quadratic_with_category_gain", False, True),
    ):
        ablated = make_model("poly2").set_params(
            regressor__cross__exclude_product=exclude, regressor__cross__enabled=cross
        )
        scores = cross_validate(
            ablated, xt, yt, cv=final_cv, scoring="neg_mean_absolute_error"
        )
        for fold, score in enumerate(scores["test_score"]):
            ablation_rows.append({"basis": label, "fold": fold, "MAE": -score})
    pd.DataFrame(ablation_rows).to_csv(RESULTS / "ablation_train_cv.csv", index=False)
    chosen = choose_raw({f: final[(f, False)] for f in FAMILIES})
    model = final[(chosen, False)].best_estimator_
    regime_train = xt.pgain.to_numpy() > 3
    regime_fits = {
        f: search(f, False, xt.loc[regime_train], yt[regime_train], final_cv)
        for f in FAMILIES
    }
    regime_family = choose_raw(regime_fits)
    regime_model = regime_fits[regime_family].best_estimator_
    dump_json(
        RESULTS / "selection_frozen.json",
        {
            "family": chosen,
            "params": final[(chosen, False)].best_params_,
            "regime_family": regime_family,
            "regime_params": regime_fits[regime_family].best_params_,
            "selection_rule": "raw only; one-SE lower dimension across family winners",
            "tuning_cv": {
                "type": "RepeatedKFold",
                "n_splits": CONFIG.inner_folds,
                "n_repeats": CONFIG.inner_repeats,
                "random_state": CONFIG.seed + 100,
            },
        },
    )
    rows = []
    for family, log in keys:
        fit = final[(family, log)]
        for label, xx, yy in (("train", xt, yt), ("test", xv, yv)):
            rows.append(
                {
                    "family": family,
                    "target": "log" if log else "raw",
                    "split": label,
                    **metrics(yy, fit.predict(xx)),
                }
            )
        subset = folds[
            (folds.family == family) & (folds.target == ("log" if log else "raw"))
        ]
        rows.append(
            {
                "family": family,
                "target": "log" if log else "raw",
                "split": "CV",
                **{m: subset[m].mean() for m in ("MAE", "RMSE", "R2")},
                **{m + "_std": subset[m].std() for m in ("MAE", "RMSE", "R2")},
            }
        )
        curve = pd.DataFrame(fit.cv_results_)
        curve[
            ["params", "mean_test_score", "std_test_score", "mean_train_score"]
        ].to_csv(RESULTS / f"grid_{family}_{'log' if log else 'raw'}.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "metrics.csv", index=False)
    regime_rows = []
    for label, mask in (
        ("interpolation_p_gt_3", xv.pgain.to_numpy() > 3),
        ("extrapolation_p_eq_3", xv.pgain.to_numpy() == 3),
    ):
        for name, fitted in (("regime_model", regime_model), ("ordinary_model", model)):
            regime_rows.append(
                {
                    "model": name,
                    "regime": label,
                    "n_train": (
                        int(regime_train.sum()) if name == "regime_model" else len(xt)
                    ),
                    "n_test": int(mask.sum()),
                    **metrics(yv[mask], fitted.predict(xv.loc[mask])),
                }
            )
    regime_table = pd.DataFrame(regime_rows)
    regime_table.to_csv(RESULTS / "regime_metrics.csv", index=False)
    joblib.dump(model, RESULTS / "model.joblib")
    export_test_predictions(model, xv, yv)
    beta = coefficients(model)
    beta.to_csv(RESULTS / "coefficients.csv", header=["coefficient"])
    stable, ols_stable = bootstrap(model, xt, yt)
    stable.to_csv(RESULTS / "bootstrap_coefficients.csv")
    ols_stable.to_csv(RESULTS / "bootstrap_ols_same_basis.csv")
    design = model.regressor_[:2].transform(xt)
    names = model.regressor_[:2].get_feature_names_out()
    correlation = pd.DataFrame(design, columns=names).corr()
    correlation.to_csv(RESULTS / "feature_correlations.csv")
    standardized = model.regressor_["scale"].transform(design)
    singular = np.linalg.svd(
        np.column_stack((np.ones(len(xt)), standardized)), compute_uv=False
    )
    diag = {
        "rank": int(np.linalg.matrix_rank(standardized)),
        "n_features": design.shape[1],
        "condition_number_with_intercept": float(singular[0] / singular[-1]),
    }

    average_oof = selected_oof.mean(axis=0)
    residual = yt - average_oof
    diagnostics = xt.copy()
    diagnostics["y"] = yt
    diagnostics["prediction"] = average_oof
    diagnostics["residual"] = residual
    diagnostics["abs_error"] = np.abs(residual)
    diagnostics.to_csv(RESULTS / "oof_diagnostics.csv", index_label="row_id")
    group_rows = []
    for p, g in diagnostics.groupby("pgain"):
        group_rows.append(
            {
                "pgain": p,
                "n": len(g),
                "mean_residual": g.residual.mean(),
                **metrics(g.y.to_numpy(), g.prediction.to_numpy()),
            }
        )
    groups = pd.DataFrame(group_rows)
    groups.to_csv(RESULTS / "oof_by_pgain.csv", index=False)
    for key, vector in (
        ("prediction", average_oof),
        ("pgain", xt.pgain),
        ("vgain", xt.vgain),
    ):
        rho, pvalue = spearmanr(residual, vector)
        diag[key] = {"rho": float(rho), "p_descriptive": float(pvalue)}
    diag["negative_test_predictions"] = int((model.predict(xv) < 0).sum())
    # Median partial derivative across observed training configurations.
    h = 1e-4
    xp, xm = xt.copy(), xt.copy()
    xp["pgain"] = xp.pgain.astype(float) + h
    xm["pgain"] = xm.pgain.astype(float) - h
    deriv = (model.predict(xp) - model.predict(xm)) / (2 * h)
    diag["negative_partial_p_fraction"] = float((deriv < 0).mean())
    dump_json(RESULTS / "diagnostics.json", diag)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    pred = model.predict(xv)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(yt, average_oof, alpha=0.55, label="Train: nested OOF (3-repeat mean)")
    ax.scatter(yv, pred, marker="x", color="crimson", label="Locked test")
    lim = [min(y.min(), pred.min(), average_oof.min()) - 0.2, y.max() + 0.3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="Измеренное / заданное y, ед.н.д.", ylabel="Прогноз, ед.н.д.")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "measured_predicted.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, values in zip(
        axes, ["Прогноз", "pgain", "vgain"], [average_oof, xt.pgain, xt.vgain]
    ):
        ax.scatter(values, residual, alpha=0.6)
        ax.axhline(0, color="black", ls="--")
        ax.set(xlabel=key, ylabel="OOF остаток y − прогноз, ед.н.д.")
    fig.tight_layout()
    fig.savefig(FIGURES / "residuals.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for family in ("ridge", "lasso", "elastic"):
        result = pd.DataFrame(final[(family, False)].cv_results_)
        # Show a fixed quadratic basis WITHOUT category*gain to isolate alpha.
        keep = result.params.apply(
            lambda d: d.get("regressor__pre__gain__poly__degree") == 2
            and not d.get("regressor__cross__enabled", False)
            and d.get("regressor__reg__l1_ratio", 0.8) == 0.8
        )
        r = result.loc[keep].copy()
        alpha = r.params.apply(lambda d: d["regressor__reg__alpha"]).to_numpy()
        axes[0].plot(alpha, -r.mean_test_score, "o-", label=family)
        axes[0].plot(alpha, -r.mean_train_score, "--", alpha=0.45)
    axes[0].set(
        xscale="log",
        xlabel="alpha",
        ylabel="MAE, ед.н.д.",
        title="CV: сплошная; train: пунктир",
    )
    axes[0].legend()
    for degree, label in ((1, "Степень 1"), (2, "Степень 2")):
        fit = search("ols" if degree == 1 else "poly2", False, xt, yt, final_cv)
        if degree == 2:
            index = next(
                i
                for i, p in enumerate(fit.cv_results_["params"])
                if p["regressor__cross__enabled"] is False
            )
        else:
            index = 0
        axes[1].bar(
            degree - 0.16,
            -fit.cv_results_["mean_train_score"][index],
            width=0.32,
            color="steelblue",
            label="train" if degree == 1 else None,
        )
        axes[1].bar(
            degree + 0.16,
            -fit.cv_results_["mean_test_score"][index],
            width=0.32,
            color="darkorange",
            label="CV" if degree == 1 else None,
        )
    axes[1].set(
        xticks=[1, 2],
        xlabel="Степень полинома (без category*gain)",
        ylabel="MAE, ед.н.д.",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "complexity.png", dpi=170)
    plt.close(fig)
    heat = diagnostics.pivot_table(
        index="vgain", columns="pgain", values="abs_error", aggfunc="mean"
    )
    counts = diagnostics.pivot_table(
        index="vgain", columns="pgain", values="abs_error", aggfunc="count"
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="YlOrRd")
    for i in range(len(heat)):
        for j in range(len(heat.columns)):
            if pd.notna(heat.iloc[i, j]):
                ax.text(
                    j,
                    i,
                    f"{heat.iloc[i, j]:.2f}\nn={int(counts.iloc[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
    ax.set(
        xticks=range(len(heat.columns)),
        xticklabels=heat.columns,
        yticks=range(len(heat.index)),
        yticklabels=heat.index,
        xlabel="pgain",
        ylabel="vgain",
        title="MAE по средним nested OOF прогнозам",
    )
    fig.colorbar(im, ax=ax, label="MAE, ед.н.д.")
    fig.tight_layout()
    fig.savefig(FIGURES / "error_map.png", dpi=170)
    plt.close(fig)
    selected_beta = (
        stable.drop(index="intercept")
        .sort_values("median", key=lambda s: s.abs())
        .tail(12)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        selected_beta["median"],
        range(len(selected_beta)),
        xerr=[
            selected_beta["median"] - selected_beta.q025,
            selected_beta.q975 - selected_beta["median"],
        ],
        fmt="o",
    )
    ax.axvline(0, color="gray", ls="--")
    ax.set(
        yticks=range(len(selected_beta)),
        yticklabels=selected_beta.index,
        xlabel="Коэффициент в исходном базисе: медиана и 2.5–97.5% bootstrap",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "coefficient_stability.png", dpi=170)
    plt.close(fig)
    write_report(
        table,
        folds,
        regime_table,
        beta,
        stable,
        ols_stable,
        groups,
        diag,
        chosen,
        regime_family,
        final[(chosen, False)].best_params_,
        selection_rows,
    )
    print(f"DONE. Final raw family={chosen}; see REPORT.md", flush=True)


def write_report(*args: Any) -> None:
    """Render the scientific account from the completed experiment artifacts."""
    from readable_report import write_readable_report

    write_readable_report(ROOT)


def refresh_report() -> None:
    """Rebuild the report from saved results, with no refitting or test tuning."""
    selection = json.loads(
        (RESULTS / "selection_frozen.json").read_text(encoding="utf-8")
    )
    diagnostic = json.loads((RESULTS / "diagnostics.json").read_text(encoding="utf-8"))
    split = json.loads((RESULTS / "split.json").read_text(encoding="utf-8"))
    test_data = load_data().iloc[split["test_row_ids"]]
    export_test_predictions(
        joblib.load(RESULTS / "model.joblib"),
        test_data[COLUMNS],
        test_data.y.to_numpy(),
    )
    write_report(
        pd.read_csv(RESULTS / "metrics.csv"),
        pd.read_csv(RESULTS / "nested_cv_folds.csv"),
        pd.read_csv(RESULTS / "regime_metrics.csv"),
        pd.read_csv(RESULTS / "coefficients.csv", index_col=0).iloc[:, 0],
        pd.read_csv(RESULTS / "bootstrap_coefficients.csv", index_col=0),
        pd.read_csv(RESULTS / "bootstrap_ols_same_basis.csv", index_col=0),
        pd.read_csv(RESULTS / "oof_by_pgain.csv"),
        diagnostic,
        selection["family"],
        selection["regime_family"],
        selection["params"],
        pd.read_csv(RESULTS / "nested_selection.csv").to_dict("records"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optional original servo.zip; data/raw is otherwise used",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild REPORT.md from saved results without training",
    )
    args = parser.parse_args()
    if args.report_only:
        refresh_report()
    else:
        main(args.archive)
