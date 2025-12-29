import pandas as pd
from pysr import PySRRegressor
import numpy as np
from typing import Dict, Any, Tuple
import re
import sympy as sp
from sympy import sympify, simplify, Float
from config_translator import exp_decay, safe_div, inv
from pysr import PySRRegressor
import torch
import jax.numpy as jnp


try:
    import torch
except ImportError:
    torch = None

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None


# Example built-in custom ops
def safe_div(x, y):
    return np.where(y == 0, 0.0, x / y)

def inv(x):
    return 1.0 / x

def exp_decay(x):
    return np.exp(-x)

def _normalize_julia_key(params: Dict[str, Any]) -> Dict[str, Any]:
    # PySR no longer uses extra_julia_defs/extra_julia_mappings
    return params

def make_json_safe(obj):
    """Recursively convert SymPy Floats, numpy numbers, etc. into JSON-serializable types."""
    import numpy as np
    import sympy as sp

    if obj is None:
        return None

    # native types are fine
    if isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy types -> Python types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)

    # sympy Float -> Python float
    if isinstance(obj, sp.Float):
        return float(obj)

    # sympy Integer -> Python int
    if isinstance(obj, sp.Integer):
        return int(obj)

    # lists
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]

    # tuples
    if isinstance(obj, tuple):
        return tuple(make_json_safe(x) for x in obj)

    # dicts
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    # fallback: string form
    return str(obj)

# -------------------------
# Custom physics-informed operators
# -------------------------
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

def normalize_colname(name: str) -> str:
    """Ensures all Pi_?_pow names follow consistent single-underscore form."""
    name = re.sub(r"(_){2,}", "_", name)
    name = name.replace("pow__", "pow_")
    name = name.replace("pow_-", "pow__-")
    return name

def sanitize_feature_name(name: str) -> str:
    """
    Final cleanup of feature names to ensure PySR never receives malformed names.
    This eliminates:
    - double/triple underscores
    - pow___ patterns
    - pow__- patterns
    - stray spaces introduced by PySR
    - missing decimal points in exponents
    """
    # 1. Collapse any multiple underscores
    name = re.sub(r"_+", "_", name)

    # 2. Normalize pow patterns:
    #    Pi_3_pow__0_5 -> Pi_3_pow_0.5
    name = name.replace("pow__", "pow_")
    name = name.replace("pow___", "pow_")

    # 3. Convert exponent underscore style:
    #    pow_0_5 -> pow_0.5
    #    pow_-0_5 -> pow_-0.5
    #name = re.sub(r"pow_(-?\d+)_(\d+)", r"pow_\1.\2", name)
    name = re.sub(r"_+", "_", name)

    # 4. Remove spaces introduced by Julia mutation
    name = name.replace(" ", "")

    # 5. Fix corrupted negative exponent formats:
    #    pow_- 0.5 -> pow_-0.5
    name = re.sub(r"pow_-(\s*)", "pow_-", name)

    # 6. Fix weird patterns like _-_, __-__, etc.
    name = re.sub(r"_+-_+", "_-", name)

    return name

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

    # Backfill defaults if missing
    defaults = {
        "niterations": 500,
        "population_size": 100,
        "populations": 30,
        "maxsize": 15,
        "precision": 64,
        "parsimony": 1e-3,
        "procs": 4,
        "denoise": False,
        "elementwise_loss": "L2DistLoss()",
        "model_selection": "accuracy",
        "early_stop_condition": "stop_if(loss, complexity) = loss < 1e-5 && complexity < 10",
    }
    for k, v in defaults.items():
        if k not in safe:
            safe[k] = v

    if safe["niterations"] > 700:
        print(f"[SAFE SCALE] Reducing niterations {safe['niterations']} → 700")
        safe["niterations"] = 700

    if safe["population_size"] > 512:
        print(f"[SAFE SCALE] Reducing population_size {safe['population_size']} → 512")
        safe["population_size"] = 512

    if safe["populations"] > 30:
        print(f"[SAFE SCALE] Reducing populations {safe['populations']} → 30")
        safe["populations"] = 30

    if safe["maxsize"] > 15:
        print(f"[SAFE SCALE] Reducing max_size {safe['maxsize']} → 15")
        safe["maxsize"] = 15

    if safe["precision"] not in [32, 64]:
        print(f"[SAFE SCALE] Invalid precision {safe['precision']}, forcing 32")
        safe["precision"] = 32

    unsupported = {
        "variable_dimensions",
        "output_dimension",
        "dimensional_constraint_penalty",
        "dimensionless_constants_only",
        "variable_units",
        "output_unit",
        "X_units",
        "y_units",
        "x_units",
        "y_Units",
    }
    for k in list(safe.keys()):
        if k in unsupported:
            print(f"[SAFE SCALE] Dropping unsupported PySR param from LLM config: {k}")
            safe.pop(k, None)

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
        parsimony=1e-3,
        procs=4,
        denoise=False,
        elementwise_loss="L2DistLoss()",
        maxsize=15,  # Julia-side loss
        early_stop_condition="stop_if(loss, complexity) = loss < 1e-5 && complexity < 10",
    )

    # Merge with sanitized LLM hyperparams and apply safe scaling
    llm_hparams = translated_params.pop("pysr_search_params", {})
    llm_hparams = _safe_scale_hparams(llm_hparams)
    search_params = {**default_search, **llm_hparams}

    # ✅ Ensure required keys are present with defaults
    required_keys = ["niterations", "populations", "population_size", "maxsize"]
    for k in required_keys:
        if k not in search_params:
            print(f"[SAFE SCALE] Missing {k}, setting default {default_search[k]}")
            search_params[k] = default_search[k]

    # Ensure model_selection is valid
    valid_model_selection = {"best", "accuracy", "score"}
    ms = search_params.get("model_selection", "accuracy")
    if ms not in valid_model_selection:
        print(f"WARNING: Invalid model_selection '{ms}', forcing 'accuracy'")
        search_params["model_selection"] = "accuracy"

    # -------------------------------------------------------------------
    # 1. Enable dimensional analysis using JSON config
    # -------------------------------------------------------------------
    if "variables" in translated_params:
        var_dims = translated_params.pop("variables")  # ✅ remove invalid key before PySRRegressor
    else:
        var_dims = {}

    target_dim = translated_params.pop("target_dimension", None)  # ✅ also remove if present

    # --- Enable dimensional consistency constraints if dimensions are defined ---
    if var_dims and target_dim:
        print("[INFO] Enabling dimensional constraints from config.")
        print("[INFO] Detected variable dimensions, but your PySR version does not accept dimensional args directly.")
        print("[INFO] Will run without built-in dimensional penalty (handled later via physics filter).")

    #-----------------------------
    # 2 Sympy, Torch, JAX definitions for custom operators
    #-----------------------------
    python_ops_sympy = {
        "safe_div": lambda x, y: x / (y + 1e-9),
        "inv": lambda x: 1 / (x + 1e-9),
        "exp_decay": lambda x: sp.exp(-x),
    }
    python_ops_torch = {
        "safe_div": lambda x, y: x / (y + 1e-9),
        "inv": lambda x: 1 / (x + 1e-9),
        "exp_decay": lambda x: torch.exp(-x),
    }
    python_ops_jax = {
        "safe_div": "jnp.divide",
        "inv": "jnp.reciprocal",
        "exp_decay": "jnp.exp",
    }
    # -------------------------
    # Inject custom operators into sympy mappings
    # -------------------------
    # 1) Built-in Python callables (available in this runner)
        # Prepare Python callables for PySR
    #custom_ops = {
     #   "safe_div": safe_div,
      #  "inv": inv,
       # "exp_decay": exp_decay,
    #}

    # Merge extra_sympy_mappings from translator
    extra_sympy = translated_params.pop("extra_sympy_mappings", {})
    if isinstance(extra_sympy, dict):
            python_ops_sympy.update(extra_sympy)

    # Sympy mappings (Python-level lambdas, always callables)
    search_params["extra_sympy_mappings"] = python_ops_sympy

    # Torch mappings (MUST be callables, not strings)

    search_params["extra_torch_mappings"] = {
        "safe_div": lambda x, y: torch.div(x, y + 1e-9),
        "inv": lambda x: torch.reciprocal(x + 1e-9),
        "exp_decay": torch.exp,
    }

    # JAX mappings (MUST be strings, not callables)
    search_params["extra_jax_mappings"] = {
        "safe_div": "jnp.divide",
        "inv": "jnp.reciprocal",
        "exp_decay": "jnp.exp",
    }

    # -------------------------
    # Merge Julia-side operators (unary/binary)
    # -------------------------
    unary_ops = translated_params.pop("unary_operators", [])
    binary_ops = translated_params.pop("binary_operators", [])

    # Add custom Julia operators
    unary_ops.extend([
        "inv(x) = 1 / (abs(x) < 1e-12 ? 1e-12 : x)",
        "exp_decay(x) = exp(-x)",
    ])
    binary_ops.append(
        "safe_div(x, y) = x / (abs(y) < 1e-12 ? 1e-12 : y)"
    )

    search_params["unary_operators"] = unary_ops
    search_params["binary_operators"] = binary_ops

    # -------------------------------------------------------------------
    # 3. Constraints (power)
    # -------------------------------------------------------------------
    constraints = translated_params.pop("constraints", {})
    if "^" not in constraints:
        constraints["^"] = (-2, 2)
        print("[INFO] Adding default power constraint: '^': (-2, 2)")
    search_params["constraints"] = constraints

    # -------------------------
    # Prepare variable names and X_local (ALWAYS)
    # -------------------------
    # Start from the original column names
    original_colnames = list(X.columns)

    # First normalize Pi_* names (collapse multiple underscores)
    raw_var_names = [normalize_colname(str(n)) for n in original_colnames]

    # ✓ Step 1: basic cleanup (remove illegal chars)
    sanitized_names = [re.sub(r"[^0-9A-Za-z_]", "_", str(n)) for n in raw_var_names]

    # ✓ Step 2: apply robust final sanitization to eliminate pow___ etc.
    sanitized_names = [sanitize_feature_name(n) for n in sanitized_names]

    # Keep mapping for re-substitution later (so PySR equations show proper names)
    name_map = dict(zip(sanitized_names, raw_var_names))

    # Use a local copy with sanitized column names
    X_local = X.copy()
    X_local.columns = sanitized_names

    if sanitized_names != original_colnames:
        print(
            "[INFO] Some column names contained invalid characters or were normalized; using sanitized names for PySR.")

    search_params["variable_names"] = sanitized_names
    translated_params.pop("custom_operator_suggestions", None)

    # Initialize PySR (variable_names will be in search_params)
    model = PySRRegressor(
        **search_params,
        **translated_params,
    )

    print("Starting PySR search (this can take a while)...")
    # Absolute safety: drop inf/NaN again before PySR
    X_local = X_local.replace([np.inf, -np.inf], np.nan)
    if X_local.isna().sum().sum() > 0:
        print("[WARN] Found invalid entries before PySR, applying median filling (runner-level).")
        X_local = X_local.fillna(X_local.median(numeric_only=True))
    # Fit using the DataFrame so PySR can use column names
    model.fit(X_local, y, variable_names=sanitized_names)

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

    # Replace sanitized names using name_map
    for san, raw in name_map.items():
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

def run_pysr_search_phase2(X, y, translated_params):
    """
    Phase-2 wrapper:
    Calls the original run_pysr_search(), then converts output to a JSON-safe dict.
    """
    model, meta = run_pysr_search(X, y, translated_params)

    # Convert equations table to Python dicts
    eq_df = model.equations_.copy()
    equations_serializable = eq_df.to_dict(orient="records")

    # Find best equation safely
    try:
        best_idx = eq_df["loss"].idxmin()
        best_eq = eq_df.loc[best_idx, "equation"]
        best_loss = float(eq_df.loc[best_idx, "loss"])
        best_complexity = int(eq_df.loc[best_idx, "complexity"])
    except Exception:
        best_eq = None
        best_loss = None
        best_complexity = None

        # Return JSON-safe dict
        result = {
            "equations": equations_serializable,
            "best_equation": best_eq,
            "best_loss": best_loss,
            "best_complexity": best_complexity,
            "train_r2_best": float(meta.get("train_r2_best", float("nan"))),
            "search_params": meta.get("search_params", {}),
        }

        return make_json_safe(result)
