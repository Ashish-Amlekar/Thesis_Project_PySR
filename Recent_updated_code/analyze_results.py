import os
import json
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", palette="muted")


# ============================================================
# Load raw dataset
# ============================================================
def load_raw_data():
    with open("dataset_configs/drying_dataset.json", "r") as f:
        cfg = json.load(f)

    df = pd.read_excel(cfg["data_file_path"])
    X_raw = df[cfg["raw_feature_columns"]].copy()
    y = df[cfg["target_variable"]].to_numpy()

    return X_raw, y


# ============================================================
# Build π-features if Buckingham-π was used
# ============================================================
def build_pi_features(X_raw, pi_map_path):
    """
    Reconstructs π-features using pi_map.json
    """
    with open(pi_map_path, "r") as f:
        pi_map = json.load(f)

    X_pi = pd.DataFrame(index=X_raw.index)

    for pi_name, expr in pi_map["pi_expressions"].items():
        expr = expr.replace("^", "**")
        sym_expr = sp.sympify(expr)

        symbols = sorted(sym_expr.free_symbols, key=lambda s: str(s))
        func = sp.lambdify(symbols, sym_expr, "numpy")

        args = [X_raw[str(s)].to_numpy() for s in symbols]
        X_pi[pi_name] = func(*args)

    return X_pi


# ============================================================
# Load final equation
# ============================================================
def load_final_equation(run_dir):
    with open(os.path.join(run_dir, "final_equations.json"), "r") as f:
        data = json.load(f)

    eq_str = data["final_physics_aware"]["algebraic"]
    eq_str = eq_str.replace("^", "**").replace("⋅", "*")
    expr = sp.sympify(eq_str)

    return eq_str, expr


# ============================================================
# Evaluate equation
# ============================================================
def predict_from_equation(expr, X_df):
    symbols = sorted(expr.free_symbols, key=lambda s: str(s))
    func = sp.lambdify(symbols, expr, "numpy")

    args = [X_df[str(s)].to_numpy() for s in symbols]
    y_pred = func(*args)

    return np.asarray(y_pred).flatten()


# ============================================================
# Plotting
# ============================================================
def plot_results(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=30, alpha=0.8)

    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "k--", lw=1.2)

    plt.xlabel("Observed Phoretic Spread [µm]")
    plt.ylabel("Predicted Phoretic Spread [µm]")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
def run_plots_for_run(run_dir):
    print(f"📂 Loading run directory: {run_dir}")

    X_raw, y = load_raw_data()

    # Detect π-space usage
    pi_map_path = os.path.join(
        "outputs", "01_AA", os.path.basename(run_dir), "pi_map.json"
    )

    if os.path.exists(pi_map_path):
        print("🔁 Detected Buckingham-π → rebuilding π-features")
        X = build_pi_features(X_raw, pi_map_path)
        title = "Final Physics-Aware Equation (π-space)"
    else:
        print("📐 No Buckingham-π → using raw features")
        X = X_raw
        title = "Final Physics-Aware Equation (raw space)"

    eq_str, expr = load_final_equation(run_dir)
    print("\n📘 Final Equation:")
    print(eq_str)

    y_pred = predict_from_equation(expr, X)
    plot_results(y, y_pred, title)


# ============================================================
# Manual call
# ============================================================
if __name__ == "__main__":
    run_dir = r"C:\Users\ashis\PycharmProjects\MCPLLM\outputs\run_20251212_121332"
    run_plots_for_run(run_dir)