# analyze_run_results.py
# ============================================================
# Post-run analysis & visualization (π-aware, Excel-safe)
# ============================================================

import os
import json
import re
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ---------------------- STYLE ----------------------
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
})


# ============================================================
# Dataset loader (CSV / Excel)
# ============================================================
def load_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {ext}")


# ============================================================
# Load final equation
# ============================================================
def load_final_equation(run_dir):
    path = os.path.join(run_dir, "final_equations.json")
    with open(path, "r") as f:
        data = json.load(f)

    eq_str = data["final_physics_aware"]["algebraic"]
    eq_str = eq_str.replace("^", "**").replace("⋅", "*")
    expr = sp.sympify(eq_str)
    return expr


# ============================================================
# Load π-map
# ============================================================
def load_pi_map(run_dir):
    # search recursively (your main.py saves under 01_AA/)
    for root, _, files in os.walk(run_dir):
        if "pi_map.json" in files:
            with open(os.path.join(root, "pi_map.json"), "r") as f:
                return json.load(f)
    return None


# ============================================================
# Build π-features from raw dataset
# ============================================================
def build_pi_features(df, pi_map):
    pi_df = pd.DataFrame(index=df.index)

    for pi_name, expr in pi_map.items():
        expr = expr.replace("^", "**")
        sym_expr = sp.sympify(expr)
        symbols = list(sym_expr.free_symbols)
        f = sp.lambdify(symbols, sym_expr, "numpy")
        pi_df[pi_name] = f(*[df[str(s)].to_numpy() for s in symbols])

    return pi_df


# ============================================================
# Evaluate equation safely
# ============================================================
def evaluate_equation(expr, X):
    symbols = sorted(expr.free_symbols, key=lambda s: str(s))
    f = sp.lambdify(symbols, expr, "numpy")
    y_pred = f(*[X[str(s)].to_numpy() for s in symbols])
    return np.asarray(y_pred).flatten()


# ============================================================
# Plotting utilities
# ============================================================
def plot_observed_vs_predicted(y_true, y_pred, outdir):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    plt.figure()
    plt.scatter(y_true, y_pred, s=60, alpha=0.8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "k--", lw=2)

    plt.xlabel("Observed Phoretic Spread")
    plt.ylabel("Predicted Phoretic Spread")
    plt.title("Observed vs Predicted Phoretic Spread")

    txt = f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nMSE = {mse:.3f}"
    plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes,
             va="top", bbox=dict(boxstyle="round", alpha=0.2))

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "observed_vs_predicted.png"), dpi=300)
    plt.close()


def plot_residuals(y_true, y_pred, outdir):
    res = y_true - y_pred

    plt.figure()
    plt.scatter(y_pred, res, s=60, alpha=0.8)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Predicted Phoretic Spread")
    plt.ylabel("Residual (Observed − Predicted)")
    plt.title("Residuals vs Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "residuals_vs_predicted.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.hist(res, bins=30, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "residual_histogram.png"), dpi=300)
    plt.close()


# ============================================================
# MAIN ANALYSIS
# ============================================================
def analyze_run(run_dir, dataset_path, target_col):
    outdir = os.path.join(run_dir, "analysis_plots")
    os.makedirs(outdir, exist_ok=True)

    df = load_dataset(dataset_path)
    y_true = df[target_col].to_numpy()

    expr = load_final_equation(run_dir)

    uses_pi = any(str(s).startswith("Pi_") for s in expr.free_symbols)

    if uses_pi:
        pi_map = load_pi_map(run_dir)
        if pi_map is None:
            raise RuntimeError("Equation uses Pi_* but no pi_map.json was found.")
        X = build_pi_features(df, pi_map)
    else:
        X = df

    y_pred = evaluate_equation(expr, X)

    plot_observed_vs_predicted(y_true, y_pred, outdir)
    plot_residuals(y_true, y_pred, outdir)

    print(f"\n✅ Analysis complete. Plots saved to:\n{outdir}")


# =================== USER CONFIG ===================
if __name__ == "__main__":

    RUN_DIR = r"outputs/run_20251212_130446"
    DATASET_PATH = r"Drying_dataset_Phor.xlsx"
    TARGET_COLUMN = "Phor"

    analyze_run(
        RUN_DIR,
        DATASET_PATH,
        TARGET_COLUMN,
    )