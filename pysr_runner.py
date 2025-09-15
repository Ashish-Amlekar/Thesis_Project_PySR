import pandas as pd
from pysr import PySRRegressor
import numpy as np
from typing import Dict, Any, Tuple
import re
from sympy import sympify, simplify, Float
import sympy as sp


# -------------------------
# Custom physics-informed operators
# -------------------------

def exp_decay(x, k=1.0):
    """Exponential decay: exp(-k * x). Models decreasing Phor with Wf."""
    return np.exp(-k * x)

def seg_affinity(Sratio, Fs, A=1.0):
    """Segregation affinity: (1 - Sratio^A) * Fs."""
    return (1 - Sratio**A) * Fs

def safe_div(a, b):
    """Safe division: returns 0 when denominator is near 0."""
    b_safe = np.where(np.abs(b) < 1e-6, 1e-6, b)
    return a / b_safe

def load_and_prepare_data(excel_path: str, feature_columns: list, target_column: str):
    print(f"Loading data from Excel file: {excel_path}...")
    df = pd.read_excel(excel_path)
    all_required_cols = feature_columns + [target_column]
    for col in all_required_cols:
        if col not in df.columns:
            raise ValueError(f"Your Excel file is missing a required column: '{col}'")
    X = df[feature_columns]
    y = df[target_column]
    print("Data loaded successfully.")
    return X, y


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

#reducing LLM hyperparameters
def _safe_scale_hparams(hparams: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp LLM-suggested hyperparameters into safe runtime ranges."""
    safe = dict(hparams)  # copy

    if safe.get("niterations", 200) > 800:
        print(f"[SAFE SCALE] Reducing niterations {safe['niterations']} → 500")
        safe["niterations"] = 800

    if safe.get("population_size", 128) > 512:
        print(f"[SAFE SCALE] Reducing population_size {safe['population_size']} → 512")
        safe["population_size"] = 512

    if safe.get("populations", 20) > 15:
        print(f"[SAFE SCALE] Reducing populations {safe['populations']} → 10")
        safe["populations"] = 15

    if safe.get("precision", 64) not in [32, 64]:
        print(f"[SAFE SCALE] Invalid precision {safe['precision']}, forcing 32")
        safe["precision"] = 32

    return safe

def prettify_equation(expr, precision=2):
    """
    Simplify and round constants in a Sympy expression for readability.
    """
    if not isinstance(expr, sp.Basic):
        try:
            expr = sp.sympify(expr)
        except Exception:
            return expr  # return raw if sympify fails

    # Step 1: Simplify structure
    expr = sp.simplify(expr)

    # Step 2: Round floats
    expr = expr.xreplace({
        c: sp.Float(round(float(c), precision))
        for c in expr.atoms(sp.Float)
    })

    # Step 3: Simplify again to clean up nested signs
    expr = sp.simplify(expr)

    return expr

def run_pysr_search(
    X: pd.DataFrame,
    y: pd.Series,
    translated_params: Dict[str, Any],
) -> Tuple[PySRRegressor, Dict[str, Any]]:
    """
    Configure and run PySR using translated parameters (grammar + sanitized hparams).
    Returns the fitted model and a dict of run metadata including train R2 of best eq.
    This version ensures variable names are always prepared and available.
    """
    print("\n--- CONFIGURING AND RUNNING PYSR WITH LLM-GUIDED PARAMETERS ---")

    # Default search params (can be overridden by LLM)
    default_search = dict(
        niterations=200,
        populations=20,
        population_size=100,
        precision=64,
        model_selection="best",
        parsimony=1e-6,
        procs=4,
        elementwise_loss="L2DistLoss()",  # Julia-side loss
    )

    # Merge with sanitized LLM hyperparams and apply safe scaling
    llm_hparams = translated_params.pop("pysr_search_params", {})
    llm_hparams = _safe_scale_hparams(llm_hparams)
    search_params = {**default_search, **llm_hparams}

    # Ensure model_selection is valid
    valid_model_selection = {"best", "accuracy", "parsimonious"}
    ms = search_params.get("model_selection", "best")
    if ms not in valid_model_selection:
        print(f"WARNING: Invalid model_selection '{ms}', forcing 'best'")
        search_params["model_selection"] = "best"

    # Remove unsupported keys before passing to PySRRegressor
    translated_params.pop("extra_julia_mappings", None)

    # -------------------------
    # Inject custom operators into sympy mappings
    # -------------------------
    custom_ops = {
        "exp_decay": exp_decay,
        "seg_affinity": seg_affinity,
        "safe_div": safe_div,
    }

    sympy_mappings = translated_params.pop("extra_sympy_mappings", None)
    if sympy_mappings:
        custom_ops.update(sympy_mappings)  # merge with user-provided

    search_params["extra_sympy_mappings"] = custom_ops

    # -------------------------
    # Prepare variable names and X_local (ALWAYS)
    # -------------------------
    raw_var_names = list(X.columns)

    def _sanitize_name(n):
        return re.sub(r"[^0-9A-Za-z_]", "_", str(n))

    sanitized_names = [_sanitize_name(n) for n in raw_var_names]

    # Make a local copy of X with sanitized names (don't mutate caller)
    if sanitized_names != raw_var_names:
        X_local = X.copy()
        X_local.columns = sanitized_names
        print("[INFO] Some column names contained invalid characters; using sanitized names for PySR.")
    else:
        X_local = X.copy()  # keep a copy to be safe

    # Ensure no user-provided variable_names in translated_params can overwrite our names
    translated_params.pop("variable_names", None)

    # Tell PySR the variable names we want it to use
    search_params["variable_names"] = sanitized_names


    # Initialize PySR (variable_names will be in search_params)
    model = PySRRegressor(
        **search_params,
        **translated_params,
    )

    print("Starting PySR search (this can take a while)...")
    # Fit using the DataFrame so PySR can use column names
    model.fit(X_local, y)

    print("\n--- PYSR SEARCH COMPLETE ---")
    print(model)

    best = model.get_best()

    # Compute R2 for the best equation on training data
    try:
        best_lambda = best["lambda_format"]
        # Evaluate on the SAME DataFrame we used for fitting
        yhat_best = best_lambda(X_local)
        r2_best = _r2(y.to_numpy(), yhat_best)
    except Exception:
        r2_best = float("nan")

    # Pretty-print best symbolic equation and replace sanitized names / x# tokens
    eq_str = best.get("equation", None) or str(best)
    pretty_eq = eq_str

    # Replace sanitized names with raw names if those were changed
    for san, raw in zip(sanitized_names, raw_var_names):
        pretty_eq = re.sub(rf"\b{re.escape(san)}\b", raw, pretty_eq)

    # Replace x0, x1, ... and unicode subscripts x₀, x₁ with raw names
    for i, raw in enumerate(raw_var_names):
        pretty_eq = re.sub(rf"\bx{i}\b", raw, pretty_eq)
        # unicode subscript replacement (x₀, x₁ ...)
        subdigits = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉"}
        uni = "x" + "".join(subdigits[d] for d in str(i))
        pretty_eq = pretty_eq.replace(uni, raw)

    print("\nBest symbolic equation (pretty):")
    pretty_equation = prettify_equation(pretty_eq, precision=3)
    print(pretty_equation)

    meta = {
        "train_r2_best": r2_best,
        "search_params": search_params,
    }
    print(f"Training R2 of best equation: {r2_best:.4f}")

    return model, meta