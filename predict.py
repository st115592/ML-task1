"""Command-line inference using the saved strictly linear raw-target model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from servo_model import COLUMNS, validate_x


def predict(motor: str, screw: str, pgain: float, vgain: float) -> float:
    """Validate a single configuration and return y in original dataset units."""
    x = pd.DataFrame([[motor, screw, pgain, vgain]], columns=COLUMNS)
    validate_x(x)
    if not 3 <= pgain <= 6 or not 1 <= vgain <= 5:
        raise ValueError("Outside the documented gain range; extrapolation refused")
    model = joblib.load(Path(__file__).resolve().parent / "results/model.joblib")
    return float(model.predict(x)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor", choices=list("ABCDE"), required=True)
    parser.add_argument("--screw", choices=list("ABCDE"), required=True)
    parser.add_argument("--pgain", type=float, required=True)
    parser.add_argument("--vgain", type=float, required=True)
    args = parser.parse_args()
    value = predict(args.motor, args.screw, args.pgain, args.vgain)
    print(f"Predicted rise time: {value:.6f} dataset units (SI units undocumented).")
    print(
        "Local approximation only; documented ranges contain unobserved combinations."
    )
