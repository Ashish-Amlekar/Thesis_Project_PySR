import json
import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import sympy as sp
from sympy import pretty
from sympy import Matrix
from pathlib import Path

from llm_handler import generate_pysr_configuration, configure_llm
from config_translator import translate_llm_config_to_pysr_params
from pysr_runner import load_and_prepare_data, run_pysr_search, _r2
from utils.knowledge_converter import build_markdown_knowledge
from dimensional_filter import DimensionalFilter
from physics_filter import (
    build_hints_from_config_or_auto,
    apply_preprocessing_hints_to_pysr_params,
    rank_and_select_physical_equations,
    PhysicsHints, remap_hints_to_pi_space,
)
from knowledge_extractor import extract_physics_hints_from_knowledge
from subexpression_discovery import discover_repeated_subexpressions, add_new_features_to_dataset
from equation_mapper import remap_equation, pretty_print_equation, latex_equation, restore_original_variables
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


CONFIG_FILE = "dataset_configs/drying_dataset.json"

# ---------------------- Nondimensionalization Helper (generic) ----------------------

def nondimensionalize_target(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    var_dims: Dict[str, str],
    target_dim: str,
    ref_var: str | None = None,
    forbid_prefixes: tuple[str, ...] = ("Pi_",),
    min_variance: float = 1e-12,
) -> tuple[pd.Series, str | None]:
    """
    Make target dimensionless: y_dimless = y / ref_var

    Generic rule:
      - ref_var must exist in 'features'
      - dim(ref_var) must equal target_dim
      - avoid forbidden prefixes by default (e.g., Pi_*)
      - if ref_var not provided: choose the "best" candidate by a stability score

    Returns:
      (scaled_target, ref_var_used)  OR  (target, None) if no valid ref found.
    """
    def _dim(v: str) -> str:
        return str(var_dims.get(v, "")).strip()

    # --- validate an explicitly provided ref_var ---
    if ref_var is not None:
        if ref_var not in features.columns:
            print(f"[WARN] preferred_ref '{ref_var}' not in features. Skipping scaling.")
            return target, None
        if any(ref_var.startswith(p) for p in forbid_prefixes):
            print(f"[WARN] preferred_ref '{ref_var}' is forbidden by prefix {forbid_prefixes}. Skipping scaling.")
            return target, None
        if _dim(ref_var) != target_dim:
            print(f"[WARN] preferred_ref '{ref_var}' has dim '{_dim(ref_var)}' but target dim is '{target_dim}'. Skipping scaling.")
            return target, None
        denom = features[ref_var].astype(float)
        return target / (denom + 1e-12), ref_var

    # --- auto-pick a ref_var with the same dimension as target ---
    candidates = []
    for c in features.columns:
        if any(str(c).startswith(p) for p in forbid_prefixes):
            continue
        if _dim(str(c)) == target_dim:
            candidates.append(str(c))

    if not candidates:
        print(f"[WARN] No feature with dimension '{target_dim}' found for nondimensionalization. Skipping scaling.")
        return target, None

    # Score candidates for stability:
    #   - non-constant (variance)
    #   - fewer near-zeros (avoid division blowups)
    #   - "typical magnitude" not too tiny
    best = None
    best_score = -np.inf

    for c in candidates:
        x = features[c].astype(float).to_numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        var = float(np.var(x))
        if not np.isfinite(var) or var < min_variance:
            continue

        frac_near_zero = float(np.mean(np.abs(x) < 1e-9))
        med_abs = float(np.median(np.abs(x)) + 1e-12)

        # Higher is better:
        #   prefer higher variance, larger typical magnitude, fewer zeros
        score = (np.log(var + 1e-12)) + (np.log(med_abs)) - (5.0 * frac_near_zero)

        if score > best_score:
            best_score = score
            best = c

    if best is None:
        print(f"[WARN] Dimension-matching refs exist but are unstable/constant. Skipping scaling.")
        return target, None

    print(f"[INFO] Auto-picked '{best}' to nondimensionalize target (dim matches {target_dim}).")
    denom = features[best].astype(float)
    return target / (denom + 1e-12), best

def get_eval_features(eq_str, features_pi, features_orig, features_enriched):
    """Return correct feature DataFrame based on equation contents."""
    import re
    has_pi = bool(re.search(r"\bPi_\d+\b", eq_str))
    has_hint = features_enriched is not None and any(k in eq_str for k in features_enriched.columns)
    has_orig = any(k in eq_str for k in features_orig.columns)

    # priority
    if has_pi:
        return features_pi
    if has_hint:
        return features_enriched
    if has_orig:
        return features_orig
    return features_pi

def physics_hints_to_dict(hints, fallback: dict | None = None) -> dict:
    """
    Convert a PhysicsHints instance (or dict) to a plain dict that
    build_engineered_features_from_hints / build_hint_seed_features expect.
    """
    if isinstance(hints, PhysicsHints):
        return {
            "monotonicity": getattr(hints, "monotonicity", {}) or {},
            "dominant_variables": getattr(hints, "dominant_variables", []) or [],
            "preferred_forms": getattr(hints, "preferred_forms", []) or [],
            "candidate_templates": getattr(hints, "candidate_templates", []) or [],
            "recommended_operators": getattr(hints, "recommended_operators", {}) or {},
            "engineered_features": getattr(hints, "engineered_features", []) or [],
        }
    elif isinstance(hints, dict):
        # already in dict form (e.g. if build_hints_from_config_or_auto changes later)
        return hints
    else:
        # last resort: use merged_hints if provided
        return fallback or {}

# ---------------------- Column safety + rounding helpers ----------------------

def dedupe_columns(df: pd.DataFrame, *, keep: str = "last") -> pd.DataFrame:
    """Remove duplicate column names. Keeps the last occurrence by default."""
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    if df.columns.is_unique:
        return df
    return df.loc[:, ~df.columns.duplicated(keep=keep)].copy()

def col_to_1d_numpy(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Robustly get a 1D numpy array from a DataFrame column name,
    even if duplicate columns exist (then take the last one).
    """
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, -1]
    return np.asarray(x.to_numpy()).reshape(-1)

def round_sig_str(x: float, sig: int = 3) -> str:
    """Format float with N significant digits."""
    try:
        return f"{float(x):.{sig}g}"
    except Exception:
        return str(x)

def round_sympy_constants(expr: sp.Expr, sig: int = 3) -> sp.Expr:
    """
    Round ONLY Float constants in a sympy expression to N significant digits.
    Keeps structure (doesn't convert rationals to floats).
    """
    if expr is None:
        return expr
    repl = {}
    for a in expr.atoms(sp.Float):
        try:
            repl[a] = sp.Float(round_sig_str(float(a), sig))
        except Exception:
            pass
    return expr.xreplace(repl) if repl else expr

# ---------------- Helper: Operator Simplifier ----------------
def simplify_equation(eq: str) -> str:
    if not isinstance(eq, str):
        eq = str(eq)

    # Collapse the double-underscore form Pi_k_pow__0_5 → Pi_k_pow_0_5
    eq = re.sub(r"Pi_(\d+)_pow__0_5", r"Pi_\1_pow_0_5", eq)
    eq = re.sub(r"Pi_(\d+)_pow__-0_5", r"Pi_\1_pow_-0_5", eq)

    # Round decimal constants to 4 sig figs for readability
    eq = re.sub(r"([-+]?\d*\.\d+)", lambda m: f"{float(m.group()):.3g}", eq)
    return eq


def sanitize_equation_string(eq: str) -> str:
    """Cleans common PySR formatting issues before sympify()."""
    if not isinstance(eq, str):
        eq = str(eq)
    # Replace tuple commas and malformed decimals (0_5 -> 0.5)
    eq = re.sub(r",\s*([A-Za-z0-9_])", r" * \1", eq)
    eq = re.sub(r"(\d)_([0-9]+)", r"\1.\2", eq)
    eq = eq.replace(",,", ",")
    eq = eq.strip().rstrip(',')
    return eq

# ---------------------- Generic Transform Helpers (domain-agnostic) ----------------------
# NOTE: generate_generic_transforms is currently NOT used in the calendering pipeline.
# It is kept here only for future experiments if you want to re-enable generic features.

def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _safe_div(a, b, eps=1e-9):
    return a / (b + eps)

def _safe_log(x, eps=1e-9):
    # keep sign but avoid log(<=0)
    return np.log(np.abs(x) + eps) * np.sign(x)

def _clip_extremes(s: pd.Series, q_low=0.001, q_high=0.999):
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    return s.clip(lo, hi)

def generate_generic_transforms(X: pd.DataFrame,
                                powers=(-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
                                add_logs=True,
                                add_one_minus=True,
                                max_new_per_col=6,
                                corr_prune_threshold=0.999) -> pd.DataFrame:
    """
    Domain-agnostic feature engineering:
      - inverse: inv_x
      - one-minus: one_minus_x
      - powers: x^p for p in powers
      - logs: log(|x|+eps)
    Automatically prunes near-duplicate columns (high correlation).

    ⚠ Currently not applied in this project to keep feature count low.
    """
    X = X.copy()
    new_cols = {}

    numeric_cols = [c for c in X.columns if _is_numeric_series(X[c])]

    for col in numeric_cols:
        col_s = X[col].astype(float)
        candidates = []

        # 1) Inverse
        inv = _safe_div(1.0, col_s)
        candidates.append((f"inv_{col}", inv))

        # 2) One minus
        if add_one_minus:
            one_minus = 1.0 - col_s
            candidates.append((f"one_minus_{col}", one_minus))

        # 3) Powers
        for p in powers:
            try:
                powv = np.sign(col_s) * (np.abs(col_s) ** p)
                candidates.append((f"{col}_pow_{p:g}", powv))
            except Exception:
                pass

        # 4) Logs
        if add_logs:
            try:
                logv = _safe_log(col_s)
                candidates.append((f"log_{col}", logv))
            except Exception:
                pass

        # Light pruning per-column by variance
        kept = []
        for name, vec in candidates:
            v = np.nanvar(vec)
            if np.isfinite(v) and v > 1e-16:
                kept.append((name, vec))

        kept = kept[:max_new_per_col]
        for name, vec in kept:
            new_cols[name] = vec

    if not new_cols:
        return X

    X_new = pd.DataFrame(new_cols, index=X.index)

    # global correlation pruning (very strict)
    try:
        corr = X_new.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_prune_threshold)]
        X_new = X_new.drop(columns=to_drop, errors="ignore")
    except Exception:
        pass

    for c in X_new.columns:
        X_new[c] = _clip_extremes(X_new[c])

    print(f"[Transforms] Added {X_new.shape[1]} generic features (inverse, one-minus, powers, logs).")
    return pd.concat([X, X_new], axis=1)

def build_hint_seed_features(X: pd.DataFrame, hints: dict, max_templates=5):
    """
    Adds columns hint_template_i built from hints['candidate_templates'][i]['expr'].

    NOTE: In the current optimized pipeline we *do not* use these as PySR inputs.
    They are kept only for potential future experiments.
    """
    import sympy as sp
    X = X.copy()
    hint_feature_map = {}

    if not hints or "candidate_templates" not in hints:
        print("[Hints] No candidate_templates available; skipping hint features.")
        return X, hint_feature_map

    templates = hints.get("candidate_templates", [])[:max_templates]
    if not templates:
        print("[Hints] Empty candidate_templates; skipping hint features.")
        return X, hint_feature_map

    present_symbols = list(X.columns)
    sym_vars = [sp.Symbol(v) for v in present_symbols]

    def _sympify_template_to_feature(expr_str: str):
        expr_str = expr_str.replace("^", "**")
        placeholder_syms = {"C": 1.0, "A": 1.0, "B": 1.0, "k": 1.0}

        try:
            expr = sp.sympify(expr_str, convert_xor=True)
        except Exception:
            cleaned = re.sub(r"[^0-9A-Za-z_+\-*/(). ]", "", expr_str)
            expr = sp.sympify(cleaned, convert_xor=True)

        subs = {}
        for s in expr.free_symbols:
            s_name = str(s)
            if s_name not in present_symbols:
                subs[s] = placeholder_syms.get(s_name, 1.0)

        return sp.simplify(expr.subs(subs))

    for i, t in enumerate(templates, start=1):
        expr_str = (t.get("expr") or "").strip()
        if not expr_str:
            continue
        try:
            sym_expr = _sympify_template_to_feature(expr_str)
            f = sp.lambdify(sym_vars, sym_expr, modules={"log": np.log, "exp": np.exp})
            vals = f(*[X[v].to_numpy() for v in present_symbols])
            vals = np.nan_to_num(vals)
            colname = f"hint_template_{i}"
            X[colname] = vals
            hint_feature_map[colname] = expr_str
            print(f"[Hints] Added hint feature from template #{i}: {expr_str}")
        except Exception as e:
            print(f"[Hints] Skipped template #{i}: {expr_str} | Reason: {e}")

    return X, hint_feature_map

def build_engineered_features_from_hints(X: pd.DataFrame, hints: dict):
    """
    Reads hints['engineered_features'] which is a list of:
      { "name": "one_minus_phi_over_phim", "expr": "1 - phi/phim" }
    Evaluates each expression on X and creates a new column using that name.

    Returns:
      X_aug, engineered_feature_map  # dict name -> expr (string)
    """
    import numpy as np
    import sympy as sp

    X = X.copy()
    engineered_map = {}

    feats = (hints or {}).get("engineered_features") or []
    if not feats:
        return X, engineered_map

    present_symbols = list(X.columns)
    sym_vars = [sp.Symbol(v) for v in present_symbols]

    for fdef in feats:
        name = (fdef.get("name") or "").strip()
        expr_str = (fdef.get("expr") or "").strip()
        if not name or not expr_str:
            continue
        try:
            expr = sp.sympify(expr_str.replace("^", "**"), convert_xor=True)
            # only keep symbols that exist in X
            syms = [str(s) for s in expr.free_symbols]
            for s in syms:
                if s not in present_symbols:
                    print(f"[WARN] Placeholder variable '{s}' not found in dataset; substituting 1.0 for '{name}'")
                    expr = expr.subs(sp.Symbol(s), 1.0)
            f = sp.lambdify([sp.Symbol(v) for v in syms], expr, modules={"log": np.log, "exp": np.exp, "sqrt": np.sqrt})
            vals = f(*[X[v].to_numpy() for v in syms])
            vals = np.array(vals, dtype=float)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            X[name] = _clip_extremes(pd.Series(vals, index=X.index))
            engineered_map[name] = expr_str
            print(f"[Hints] Added engineered feature: {name} := {expr_str}")
        except Exception as e:
            print(f"[Hints] Skipped engineered feature '{name}'  |  Reason: {e}")

    return X, engineered_map

def load_manual_engineered_features(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[INFO] No manual engineered features file found at: {path}")
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    feats = data.get("engineered_features", [])
    if not isinstance(feats, list):
        raise ValueError("manual_engineered_features.json must contain a list under 'engineered_features'")
    # basic validation
    cleaned = []
    for feat in feats:
        if not isinstance(feat, dict):
            continue
        name = (feat.get("name") or "").strip()
        expr = (feat.get("expr") or "").strip()
        if name and expr:
            cleaned.append({"name": name, "expr": expr})
    return cleaned

def merge_engineered_features_into_hints(hints: dict, manual_feats: list[dict]) -> dict:
    hints = hints or {}
    existing = hints.get("engineered_features") or []
    # de-duplicate by name (manual overrides if same name)
    merged = { (f.get("name") or "").strip(): f for f in existing if isinstance(f, dict) and f.get("name") }
    for f in manual_feats:
        merged[f["name"]] = f
    hints["engineered_features"] = list(merged.values())
    return hints

def infer_engineered_feature_dims(
    engineered_feature_map: Dict[str, str],
    base_var_dims: Dict[str, str],
    constants: Dict[str, Any] | None = None,
    *,
    default_dim: str = "-",
) -> Dict[str, str]:
    constants = constants or {}
    df = DimensionalFilter(var_dims=base_var_dims, target_dim="1", constants=constants)

    out: Dict[str, str] = {}
    for name, expr in engineered_feature_map.items():
        try:
            valid, dim, reason = df.compute_dimension(expr)
        except Exception as e:
            valid, dim, reason = False, default_dim, str(e)

        if not valid:
            print(f"[WARN] Could not infer dimension for engineered feature '{name}' from '{expr}'. Using '{default_dim}'. Reason: {reason}")
            out[name] = default_dim
        else:
            out[name] = str(dim) if not isinstance(dim, str) else dim
    return out

# ============================================================
# ⚙️ Helper: Evaluate metrics for a given equation string (robust)
# ============================================================
def evaluate_metrics(eq_str, X_df, y_true, pi_map=None, ref_scale=None):
    """
    Evaluate symbolic regression equations on both normalized (training) and
    physical scales. Returns both sets of metrics:
      → (r2, mse, mae, acc, r2_phys, mse_phys, mae_phys, acc_phys)
    """
    import re
    import numpy as np
    import sympy as sp
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    if not isinstance(eq_str, str) or not eq_str.strip():
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

    try:
        # --- Step 1: Normalize equation string ---
        eq = (
            eq_str.replace("⋅", "*")
            .replace("^", "**")
            .replace("Π", "Pi_")
            .replace("−", "-")
            .replace(" ", "")
        )
        eq = re.sub(r"_pow__(-?\d+)", r"**\1", eq)
        eq = re.sub(r"_pow_(-?\d+)_(-?\d+)", lambda m: f"**{m.group(1)}.{m.group(2)}", eq)
        eq = re.sub(r"_pow_(-?\d+\.?\d*)", r"**\1", eq)
        eq = re.sub(r"(\d)_([0-9]+)", r"\1.\2", eq)
        eq = eq.replace("inv_", "1/(").replace("one_minus_", "(1-")
        # Add small epsilon for numeric stability in denominators like (phi - phim)
        eq = re.sub(r"\(([^()]+?)-([^()]+?)\)", r"(\1-\2+1e-9)", eq)
        if "exp_decay(" in eq:
            eq = re.sub(r"exp_decay\(([^)]+)\)", r"exp(-(\1))", eq)
        diff = eq.count("(") - eq.count(")")
        if diff > 0:
            eq += ")" * diff
        eq = re.sub(r"safe_div\(([^,]+),([^)]+)\)", r"(\1)/(\2)", eq)
        eq = re.sub(r"\)(\d)", r")*\1", eq)

        # --- Step 2: Sympy parse ---
        try:
            expr = sp.sympify(eq, convert_xor=True)
        except Exception:
            eq_safe = re.sub(r"[^0-9A-Za-z_+\-*/().,]+", "", eq)
            expr = sp.sympify(eq_safe, convert_xor=True)

        # --- Step 2: Extract variable names (deduplicated) ---
        seen = set()
        vars_in_expr = []
        for s in expr.free_symbols:
            name = str(s)
            if name not in seen:
                seen.add(name)
                vars_in_expr.append(name)

        if not vars_in_expr:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # --- Step 3: Variable mapping ---
        local_dict = {}
        df_cols = list(X_df.columns)

        for v in vars_in_expr:
            if v in df_cols:
                local_dict[v] = col_to_1d_numpy(X_df, v)
            else:
                print(f"⚠ Variable '{v}' missing in X_df; setting to 0 for metric evaluation.")
                local_dict[v] = np.zeros(len(X_df))

        # --- Step 4: Evaluate numerically ---
        func = sp.lambdify(vars_in_expr, expr, "numpy")
        y_pred = func(*[local_dict[v] for v in vars_in_expr])
        y_pred = np.nan_to_num(np.array(y_pred).flatten(), nan=0.0, posinf=0.0, neginf=0.0)

        # --- Step 5: Metrics (normalized scale) ---
        if np.std(y_pred) == 0:
            return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        acc = 1 - (mae / (np.mean(np.abs(y_true)) + 1e-8))

        # --- Step 6: Physical-scale metrics (if ref_scale provided) ---
        r2_phys = mse_phys = mae_phys = acc_phys = np.nan
        if ref_scale is not None:
            try:
                scale_vec = np.asarray(ref_scale).flatten()
                if scale_vec.shape[0] == y_true.shape[0]:
                    y_true_phys = y_true * scale_vec
                    y_pred_phys = y_pred * scale_vec
                    r2_phys = r2_score(y_true_phys, y_pred_phys)
                    mse_phys = mean_squared_error(y_true_phys, y_pred_phys)
                    mae_phys = mean_absolute_error(y_true_phys, y_pred_phys)
                    acc_phys = 1 - (mae_phys / (np.mean(np.abs(y_true_phys)) + 1e-8))
            except Exception:
                pass

        return r2, mse, mae, acc, r2_phys, mse_phys, mae_phys, acc_phys

    except Exception as e:
        print(f"⚠️ Metric evaluation failed for {eq_str[:100]} → {e}")
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan


# ---------------- Buckingham-π Dimensional Analysis ----------------

def _parse_dim_string(dim: str):
    """Turn 'M^1 L^-1 T^0' into dict like {'M':1,'L':-1,'T':0,...} over the standard base set."""
    base_units = ["M", "L", "T", "I", "Θ", "N", "J"]
    out = {b: 0.0 for b in base_units}
    if dim is None or dim.strip() in ("", "-", "1"):
        return out
    dim = dim.replace("**", "^")
    for b in base_units:
        m = re.search(rf"{b}\^?(-?\d+\.?\d*)", dim)
        if m:
            out[b] = float(m.group(1))
    return out

def run_buckingham_pi(features_df: pd.DataFrame, var_dims: dict, target_dim: str):
    """
    Compute π-groups from features_df using var_dims (strings) via nullspace of dimension matrix.

    Returns:
      new_features_df: DataFrame with columns Pi_1...Pi_k (dimensionless groups)
      pi_expr_map: dict mapping 'Pi_i' -> a human-readable product in original variables (e.g., 'Sl/d')
    """
    print("\n🔬 Running Buckingham-π dimensional analysis...")

    base_units = ["M", "L", "T", "I", "Θ", "N", "J"]
    var_names = list(features_df.columns)

    # Build dimension matrix (rows: base units, cols: variables)
    D = []
    for v in var_names:
        if v not in var_dims:
            raise ValueError(f"Missing dimension for variable '{v}' in config.")
        vec = _parse_dim_string(var_dims[v])
        D.append([vec[b] for b in base_units])
    D = Matrix(D).T  # shape (n_base, n_vars)
    rank = D.rank()
    num_pi = len(var_names) - rank

    print(f"📐 Dimension matrix rank: {rank}")
    print(f"📊 Number of dimensionless groups (π-groups): {num_pi}")

    ns = D.nullspace()
    if not ns:
        print("⚠️ No nontrivial dimensionless groups found.")
        new_features_df = features_df.copy()
        pi_expr_map = {name: name for name in var_names}
        return new_features_df, pi_expr_map

    new_cols = {}
    pi_expr_map = {}
    for i, vec in enumerate(ns, start=1):
        exps = [float(e) for e in vec]
        # Numeric π values
        values = np.ones(len(features_df))
        for vname, e in zip(var_names, exps):
            if abs(e) > 1e-12:
                values *= np.power(features_df[vname].values, e)

        col = f"Pi_{i}"
        new_cols[col] = values

        pieces = []
        for vname, e in zip(var_names, exps):
            if abs(e) < 1e-12:
                continue
            if abs(e - 1.0) < 1e-9:
                pieces.append(f"{vname}")
            else:
                pieces.append(f"{vname}^{e:.2f}")
        expr_str = " * ".join(pieces) if pieces else "1"
        pi_expr_map[col] = expr_str
        print(f"   π{i} = " + " * ".join([f"{v}^{e:.2f}" for v, e in zip(var_names, exps) if abs(e) > 1e-12]))

    new_features_df = pd.DataFrame(new_cols, index=features_df.index)
    return new_features_df, pi_expr_map


# ============================= MAIN =============================
if __name__ == "__main__":
    configure_llm()
    with open(CONFIG_FILE, "r") as f:
        cfg = json.load(f)

    DATA_FILE_PATH = cfg["data_file_path"]
    RAW_FEATURE_COLUMNS = cfg["raw_feature_columns"]
    TARGET_VARIABLE_NAME = cfg["target_variable"]
    TARGET_DIMENSION = cfg["target_dimension"]
    PROBLEM = cfg["problem_description"]
    INPUT_VARIABLES = cfg["variables"]
    CUSTOM_PROMPT_FILE = cfg.get("custom_prompt_file", None)

    print("--- Phase 1: Building knowledge base ---")
    build_markdown_knowledge("raw_knowledge/", "knowledge-base/", clear_output=False)

    # -------------------------------------------------------------
    # 📘 PHASE 1A – LLM #1: Extract Physics Hints from Knowledge Base
    # -------------------------------------------------------------
    AUTO_HINTS_PATH = "configs/auto_physics_hints.json"
    EXTRACTION_PROMPT = "user_prompts/physics_extract_prompt.txt"

    if not os.path.exists(AUTO_HINTS_PATH):
        print("\n🔎 Auto-extracting physics hints from knowledge base ...")
        auto_hints = extract_physics_hints_from_knowledge(
            md_dir="knowledge-base",
            extraction_prompt_file=EXTRACTION_PROMPT,
            save_path=AUTO_HINTS_PATH
        )
    else:
        print("\n📁 Using cached auto-extracted physics hints.")
        with open(AUTO_HINTS_PATH, "r") as f:
            auto_hints = json.load(f)

    print("\n📄 Auto Physics Hints (LLM #1 output):")
    print(json.dumps(auto_hints, indent=2))

    # -------------------------------------------------------------
    # 📊 PHASE 2 – Load dataset and prepare features
    # -------------------------------------------------------------
    try:
        features, target = load_and_prepare_data(
            excel_path=DATA_FILE_PATH,
            feature_columns=RAW_FEATURE_COLUMNS,
            target_column=TARGET_VARIABLE_NAME,
        )
        target_phor = target.copy()  # ALWAYS the real Phor from Excel (end-goal units)
        original_features = features.copy()
        features_enriched = None

        # Merge any manual hints with the auto-extracted ones
        cfg_hints = (cfg.get("physics_hints") or {})
        merged_hints = {**cfg_hints, **auto_hints}
        hints = build_hints_from_config_or_auto(features, target, merged_hints)
        hints_dict = physics_hints_to_dict(hints, fallback=merged_hints)

        # ---------------------------------------------------------------------
        # 🧱 Merge manual engineered features into hints_dict (AFTER hints_dict exists)
        # ---------------------------------------------------------------------
        MANUAL_FEATURES_PATH = "configs/manual_engineered_features.json"
        manual_feats = load_manual_engineered_features(MANUAL_FEATURES_PATH)
        hints_dict = merge_engineered_features_into_hints(hints_dict, manual_feats)
        if manual_feats:
            print(f"[INFO] Loaded {len(manual_feats)} manual engineered features into hints_dict")
        else:
            print("[INFO] No manual engineered features loaded.")

        # ---------------------------------------------------------------------
        # ✅ Build engineered features as real columns and carry them through
        #    - Includes BOTH: manual_engineered_features.json + LLM suggestions
        #    - When Buckingham-π is enabled: PySR will see (π-groups + engineered)
        # ---------------------------------------------------------------------

        # Always start from raw variables (no transforms)
        features_raw = features[RAW_FEATURE_COLUMNS].copy()

        # Collect engineered definitions from hints_dict, but **skip** collisions with raw vars.
        engineered_defs_all = (hints_dict or {}).get("engineered_features", []) or []
        engineered_defs = [
            ef
            for ef in engineered_defs_all
            if isinstance(ef, dict)
            and ef.get("name")
            and ef.get("expr")
            and ef.get("name") not in set(features_raw.columns)
        ]

        # Build engineered columns on raw features (for reporting + template matching)
        engineered_feature_map: Dict[str, str] = {}
        if engineered_defs:
            try:
                original_features, engineered_feature_map = build_engineered_features_from_hints(
                    features_raw.copy(), {"engineered_features": engineered_defs}
                )
            except Exception as e:
                print(f"[WARN] Engineered feature construction failed: {e}")
                original_features = features_raw.copy()
                engineered_feature_map = {}
        else:
            original_features = features_raw.copy()
            engineered_feature_map = {}

        engineered_cols = list(engineered_feature_map.keys())
        # Dimensions: prefer cfg["variables"] (single source of truth), fall back to inference if missing.
        engineered_var_dims = {}
        for name in engineered_cols:
            if name in INPUT_VARIABLES:
                engineered_var_dims[name] = INPUT_VARIABLES[name]

        # Infer dims only for engineered cols not present in config
        missing_dim_cols = [c for c in engineered_cols if c not in engineered_var_dims]
        if missing_dim_cols:
            inferred = infer_engineered_feature_dims(
                {k: engineered_feature_map[k] for k in missing_dim_cols},
                base_var_dims=INPUT_VARIABLES,
                constants=cfg.get("constants", {}),
                default_dim="-",
            )
            engineered_var_dims.update(inferred)

        # ---------------------------------------------------------
        # ⚙️ Optional Buckingham-π dimensional analysis
        # ---------------------------------------------------------
        use_buckingham = input("\nApply Buckingham-π dimensional analysis? (y/n): ").strip().lower()
        use_dimless_target = "n"
        if use_buckingham == "y":
            use_dimless_target = input("Nondimensionalize the target variable too? (y/n): ").strip().lower()

        pi_map: Dict[str, str] = {}

        if use_buckingham == "y":
            # Run π-analysis on RAW variables only.
            features_pi, pi_map = run_buckingham_pi(features_raw, INPUT_VARIABLES, TARGET_DIMENSION)

            # Remap hints to π-space for operator pruning etc.
            # BUT keep engineered features as explicit columns (so Sratio, inv_Wf, etc remain visible).
            hints_raw = hints
            hints = remap_hints_to_pi_space(hints, pi_map)
            try:
                # Preserve any monotonicity/dominance that targets engineered feature names.
                if engineered_cols:
                    for v in (getattr(hints_raw, "dominant_variables", []) or []):
                        if v in engineered_cols and v not in (hints.dominant_variables or []):
                            hints.dominant_variables.append(v)
                    for k, vv in (getattr(hints_raw, "monotonicity", {}) or {}).items():
                        if k in engineered_cols and k not in (hints.monotonicity or {}):
                            hints.monotonicity[k] = vv
            except Exception:
                pass

            print("\n✅ Buckingham-π analysis complete. New π-features:")
            print(features_pi.head())

            # Compute engineered features on a combined dataframe (RAW + π) so expressions can use either.
            combined_for_eng = pd.concat([features_raw, features_pi], axis=1)
            if engineered_defs:
                combined_aug, engineered_feature_map = build_engineered_features_from_hints(
                    combined_for_eng, {"engineered_features": engineered_defs}
                )
                engineered_cols = list(engineered_feature_map.keys())
                engineered_var_dims = infer_engineered_feature_dims(
                    engineered_feature_map,
                    base_var_dims=INPUT_VARIABLES,
                    constants=cfg.get("constants", {}),
                    default_dim="-",
                )
                engineered_block = combined_aug[engineered_cols]
            else:
                engineered_block = pd.DataFrame(index=features_pi.index)

            # ✅ FINAL feature matrix sent to PySR: (π-groups + engineered)
            features = pd.concat([features_pi, engineered_block], axis=1)
            features = dedupe_columns(features, keep="last")

            # Keep original_features (raw + engineered) for mapping/reporting
            # IMPORTANT: avoid re-adding engineered columns that original_features already has
            if not engineered_block.empty:
                original_features = original_features.drop(columns=engineered_block.columns, errors="ignore")
                original_features = pd.concat([original_features, engineered_block], axis=1)

            original_features = dedupe_columns(original_features, keep="last")

        else:
            # ✅ FINAL feature matrix sent to PySR: (raw + engineered)
            features = original_features.copy()
            features = dedupe_columns(features, keep="last")
            original_features = dedupe_columns(original_features, keep="last")

        # Optionally nondimensionalize the target (generic + dimension-aware)
        ref_var_used = None
        ref_scale_vec_train = None

        if use_dimless_target == "y":
            # Build dims dict for any available columns (raw + engineered if present)
            scaling_dims = dict(INPUT_VARIABLES)
            scaling_dims.update(engineered_var_dims or {})

            preferred_ref = (cfg.get("target_scaling", {}) or {}).get("preferred_ref", None)

            # IMPORTANT:
            # pick ref from ORIGINAL (raw/engineered) feature space, not Pi-space,
            # because Pi_* are dimensionless by construction.
            target_scaled, ref_var_used = nondimensionalize_target(
                features=original_features,  # <--- key change
                target=target,
                var_dims=scaling_dims,
                target_dim=TARGET_DIMENSION,
                ref_var=preferred_ref,
                forbid_prefixes=("Pi_",),
            )

            if ref_var_used is None:
                # scaling failed; keep target as-is
                use_dimless_target = "n"
                ref_scale_vec_train = np.ones(len(target_phor), dtype=float)
            else:
                target = target_scaled
                ref_scale_vec_train = original_features[ref_var_used].to_numpy()
                print(f"\n✅ Target nondimensionalized using reference variable: {ref_var_used}")
        else:
            ref_scale_vec_train = np.ones(len(target_phor), dtype=float)

        # Compatibility placeholder
        hint_feature_map = {}

        print("\n[INFO] Feature matrix after Buckingham + engineered features:")
        print(features.head())
        print(f"[INFO] Final feature count sent to PySR: {features.shape[1]}")

        # ---------------------------------------------------------
        # 🧹 DATA SANITIZATION: Handle NaN / Inf before PySR
        # ---------------------------------------------------------
        print("\n[CLEANUP] Checking for NaN or inf values before PySR...")

        # Replace infinities with NaN for uniform handling
        features = features.replace([np.inf, -np.inf], np.nan)
        target = target.replace([np.inf, -np.inf], np.nan)

        # Count total missing entries
        n_missing = features.isna().sum().sum() + target.isna().sum()
        if n_missing > 0:
            print(f"[WARN] Found {int(n_missing)} missing entries; applying median imputation.")
            # Fill each numeric column’s NaNs with its median
            features = features.fillna(features.median(numeric_only=True))
            target = target.fillna(target.median(numeric_only=True))
        else:
            print("[OK] No missing vanlues detected.")

        # Drop constant columns (zero variance) to avoid PySR normalization bugs
        stds = features.std(numeric_only=True)
        const_cols = stds[stds == 0].index.tolist()
        if const_cols:
            print(f"[WARN] Dropping constant columns (no variation): {const_cols}")
            features = features.drop(columns=const_cols)

        # Drop columns with extreme magnitudes (prevent Julia/GMP crash)
        bad_cols = []
        for c in features.columns:
            col = features[c]
            if col.abs().max() > 1e12 or col.isna().sum() > 0.1 * len(col):
                bad_cols.append(c)

        if bad_cols:
            print(f"[WARN] Dropping unstable engineered features: {bad_cols}")
            features = features.drop(columns=bad_cols)

        # Optional verification
        print(f"[CLEANUP] Final shape after sanitization: {features.shape}")

        # ---------------------------------------------------------
        # 🤖 PHASE 2.5 – LLM #2 (PySR Configurator)
        # ---------------------------------------------------------
        print("\n--- Phase 2.5: Generating PySR configuration from LLM #2 ---")

        llm_json_config = generate_pysr_configuration(
            knowledge_base_path="knowledge-base/",
            problem_description=PROBLEM,
            variables=INPUT_VARIABLES,
            custom_prompt_file=CUSTOM_PROMPT_FILE,
        )

        if not llm_json_config:
            print("\n❌ FAILED to obtain configuration from LLM #2. Aborting.")
            raise SystemExit(1)

        translated = translate_llm_config_to_pysr_params(llm_json=llm_json_config)
        translated["variables"] = INPUT_VARIABLES
        translated["target_dimension"] = TARGET_DIMENSION
        translated = apply_preprocessing_hints_to_pysr_params(translated, hints)

        # -------------------------------
        # 🔧 Remove invalid early-stop expressions
        # -------------------------------
        if "pysr_search_params" in translated:
            if "early_stop_condition" in translated["pysr_search_params"]:
                print("[WARN] Removing LLM-generated early_stop_condition (unsupported by PySR).")
                translated["pysr_search_params"].pop("early_stop_condition", None)

        print("\n[INFO] LLM #2 Suggested Operators and Parameters:")
        print(f"  Unary operators:  {translated['unary_operators']}")
        print(f"  Binary operators: {translated['binary_operators']}")
        print(f"  Search parameters: {translated['pysr_search_params']}")

        # ---------------------------------------------------------
        # 🚀 PHASE 3 – Run PySR symbolic regression
        # ---------------------------------------------------------
        model, meta = run_pysr_search(
            X=features,
            y=target,
            translated_params=translated,
        )
        # -------------------------------------------
        # CREATE A RUN DIRECTORY FOR ALL OUTPUT FILES
        # -------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("outputs", f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"[INFO] Using run_dir = {run_dir}")

        equations = model.equations_.copy()
        equations["equation_raw"] = equations["equation"].astype(str)      # ✅ add this
        equations["simplified_equation"] = equations["equation_raw"].apply(simplify_equation)

        # --- NEW: backfill per-equation training R² if PySR didn't include it ---
        r2_col = "r2" if "r2" in equations.columns else "r2_train"

        if r2_col == "r2_train":
            from sklearn.metrics import r2_score

            y_np = target.to_numpy()

            def _safe_r2(row):
                try:
                    y_pred = row["lambda_format"](features.to_numpy())
                    y_pred = np.asarray(y_pred).reshape(-1)
                    if y_pred.size != y_np.size or np.std(y_pred) == 0:
                        return np.nan
                    return r2_score(y_np, y_pred)
                except Exception:
                    return np.nan

            equations[r2_col] = equations.apply(_safe_r2, axis=1)
            # -------------------------------------------
            # PHASE 4 – Inspect PySR equations
            # -------------------------------------------
            print("\n--- Simplified Equations from PySR ---")
            for i, row in equations.iterrows():
                r2_here = row.get("r2", row.get("r2_train", np.nan))
                loss_here = row.get("loss", np.nan)
                print(
                    f"[{i}] {row['simplified_equation']} "
                    f"(R²_train={r2_here if pd.notnull(r2_here) else float('nan'):.4f} "
                    f"| loss={loss_here if pd.notnull(loss_here) else float('nan'):.3e})"
                )

            # Prefer highest R² if available; otherwise lowest loss
            if equations[r2_col].notnull().any():
                best_eq_row = equations.loc[equations[r2_col].idxmax()]
            else:
                best_eq_row = equations.loc[equations["loss"].idxmin()]

            best_eq_raw = str(best_eq_row["equation_raw"])  # ✅ source of truth for parsing/eval
            best_eq_pretty = str(best_eq_row["simplified_equation"])  # ✅ only for printing
            best_eq_train_r2 = best_eq_row.get(r2_col, np.nan)
            best_eq_train_loss = best_eq_row.get("loss", np.nan)

            print(f"\n✅ Using PySR’s best equation for comparison:")
            print(f"   → {best_eq_pretty}")  # ✅ show readable version
            print(f"   (raw: {best_eq_raw})")  # optional debug
            print(f"   (train R² = {best_eq_train_r2:.4f}, loss = {best_eq_train_loss:.3e})")

            # -------------------------------------------
            # PHASE 5 – Dimensional Consistency Analysis
            # -------------------------------------------
            print("\n--- Performing Dimensional Consistency Analysis ---")

            try:
                if use_buckingham == "y":
                    # Pi_* are dimensionless. Engineered features are inferred from their expressions.
                    effective_var_dims: Dict[str, str] = {}
                    for col in features.columns:
                        if str(col).startswith("Pi_"):
                            effective_var_dims[col] = "-"
                        elif col in engineered_var_dims:
                            effective_var_dims[col] = engineered_var_dims[col]
                        elif col in INPUT_VARIABLES:
                            effective_var_dims[col] = INPUT_VARIABLES[col]
                        else:
                            effective_var_dims[col] = "-"
                else:
                    effective_var_dims = dict(INPUT_VARIABLES)
                    effective_var_dims.update(engineered_var_dims)
                effective_target_dim = "-" if (use_dimless_target == "y") else TARGET_DIMENSION

                dim_filter = DimensionalFilter(
                    var_dims=effective_var_dims,
                    target_dim=effective_target_dim,
                    constants=cfg.get("constants", {}),
                )

                results = []
                valid_equations = []
                for _, row in equations.iterrows():
                    eq = row["simplified_equation"]
                    valid, dim, reason = dim_filter.analyze_equation(eq)
                    is_valid = valid and (dim == dim_filter.target_dim)

                    new_row = row.copy()
                    new_row["dimension"] = str(dim)
                    new_row["dimensionally_valid"] = is_valid
                    new_row["reason"] = reason
                    results.append(new_row)
                    if is_valid:
                        valid_equations.append(new_row)

                equations = pd.DataFrame(results)
                run_dir_dim = f"outputs/01_AA/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(run_dir_dim, exist_ok=True)
                equations.to_csv(os.path.join(run_dir_dim, "equations_with_dimensions.csv"), index=False)
                print(f"\n📂 Saved dimensional analysis results to {run_dir_dim}/equations_with_dimensions.csv")

                # --- Save π-map and scaling metadata for separate re-runs ---
                try:
                    # Only makes sense if Buckingham-π was used
                    if (use_buckingham == "y") and isinstance(pi_map, dict) and pi_map:
                        pi_map_path = os.path.join(run_dir_dim, "pi_map.json")
                        with open(pi_map_path, "w", encoding="utf-8") as f:
                            json.dump(pi_map, f, indent=2)
                        print(f"[INFO] π-map saved → {pi_map_path}")
                    else:
                        print("[INFO] No π-map to save (Buckingham-π not used).")

                    # Save ref var used for nondimensionalizing the target (if any)
                    scaling_meta = {
                        "target_name": TARGET_VARIABLE_NAME,
                        "used_buckingham": (use_buckingham == "y"),
                        "used_dimless_target": (use_dimless_target == "y"),
                        "ref_var_used": ref_var_used,
                    }
                    meta_path = os.path.join(run_dir_dim, "scaling_meta.json")
                    with open(meta_path, "w") as f:
                        json.dump(scaling_meta, f, indent=2)
                    print(f"[INFO] scaling meta saved → {meta_path}")
                except Exception as e:
                    print(f"[WARN] Could not save π-map/scaling_meta: {e}")

                print("\n--- Dimensional Consistency Report ---")
                for _, row in equations.iterrows():
                    status = "✔ VALID" if row["dimensionally_valid"] else "❌ INVALID"
                    print(f"{status}: {row['simplified_equation']} | Dim: {row['dimension']} | Reason: {row['reason']}")

                # --- Final dimensionally valid equation (by loss) ---
                if valid_equations:
                    best_row_dim = min(valid_equations, key=lambda r: r["loss"])
                    best_eq_dim = best_row_dim["simplified_equation"]

                    try:
                        best_expr = sp.sympify(sanitize_equation_string(best_eq_dim))
                        best_expr_simplified = sp.simplify(best_expr)
                        best_eq_pretty = best_expr_simplified.evalf(3)
                        pretty_eq_str = pretty(best_eq_pretty)
                    except Exception:
                        pretty_eq_str = best_eq_dim

                    y_pred_dim = best_row_dim["lambda_format"](features.to_numpy())
                    r2_dim = _r2(target.to_numpy(), y_pred_dim)
                    mse_dim = np.mean((target.to_numpy() - y_pred_dim) ** 2)
                    mae_dim = np.mean(np.abs(target.to_numpy() - y_pred_dim))

                    print("\n--- ✅ DIMENSIONALLY VALID EQUATION ---")
                    print(f"Equation:\n{pretty_eq_str}")
                    print(f"Loss: {best_row_dim['loss']:.4e}")
                    print(f"Train R²: {r2_dim:.4f}")
                    print(f"MAE: {mae_dim:.4e}")
                    print(f"MSE: {mse_dim:.4e}")
                else:
                    print("\n⚠️ No dimensionally valid equation found.")
                    valid_equations = []
                    best_row_dim = equations.loc[equations["loss"].idxmin()]
                    best_eq_dim = best_row_dim["simplified_equation"]

                if valid_equations:
                    print("\n📊 Summary: All dimensionally valid equations sorted by loss:")
                    valid_sorted = sorted(valid_equations, key=lambda r: r["loss"])
                    for eq in valid_sorted:
                        print(f"- Loss: {eq['loss']:.2e} | Eq: {eq['simplified_equation']}")

            except KeyError as e:
                print(f"[WARN] Dimensional check skipped due to missing constant: {e}")
                valid_equations = []
                best_row_dim = best_eq_row
                best_eq_dim = best_eq_raw
            except Exception as e:
                print(f"[WARN] Dimensional check encountered unexpected error: {e}")
                valid_equations = []
                best_row_dim = best_eq_row
                best_eq_dim = best_eq_raw

            # Final summary from PySR meta
            print(f"\nPySR training R2 (best overall): {meta['train_r2_best']:.2f}")

            # -------------------------------------------
            # PHYSICS-AWARE POST FILTER & RANKING
            # -------------------------------------------
            if "dimensionally_valid" in equations.columns and equations["dimensionally_valid"].any():
                keep_only_dim_valid = True
            else:
                print("\n⚠️ No dimensionally valid equations found; physics ranking will use all equations.")
                keep_only_dim_valid = False

            try:
                ranked, chosen = rank_and_select_physical_equations(
                    equations_df=equations,
                    X_df=features,
                    y=target.to_numpy(),
                    hints=hints,
                    keep_only_dimensionally_valid=keep_only_dim_valid,
                )
            except Exception as e:
                print(f"\n⚠️ Physics ranking skipped due to error: {e}")
                ranked = equations
                chosen = None

            # ======================================================
            # Discover repeated sub-expressions (optional enrichment)
            # ======================================================
            top_equations = ranked.sort_values("physics_total_score", ascending=False).head(7)
            equation_list = [simplify_equation(eq) for eq in top_equations["simplified_equation"].dropna().tolist()]
            repeated_subs = discover_repeated_subexpressions(equation_list, min_occurrences=2, min_symbols=2)

            if repeated_subs:
                print("\n🔬 Discovered repeated sub-expressions (to be promoted as new variables):")
                for name, expr in repeated_subs.items():
                    print(f"  {name} = {expr}")
                enriched_features = add_new_features_to_dataset(features, repeated_subs)
                features_enriched = enriched_features.copy()
                if "target" not in enriched_features.columns:
                    enriched_features["target"] = target

                enriched_path = os.path.join(run_dir, "pi_dataset_enriched.csv")
                enriched_features.to_csv(enriched_path, index=False)
                print(f"[✅] Enriched dataset with target saved → {enriched_path}")
                print(f"[✅] Enriched dataset with repeated sub-expressions saved → {enriched_path}")
                print("\n💡 Next step: Re-run PySR using this enriched dataset to discover simpler equations.")
            else:
                print("\nℹ️ No repeated sub-expressions met the threshold. Skipping enrichment.")

            # Decide which equation is “physics-best”
            if chosen is not None:
                print("\n🔎 Physics-aware ranked candidates (top 5):")
                cols = ["simplified_equation", "loss", "complexity", "physics_total_score"]
                print(ranked[cols].head().to_string(index=False))

                best_row_phys = chosen
                best_eq_phys_pretty = str(best_row_phys["simplified_equation"])
                best_eq_phys_raw = str(
                    best_row_phys["equation_raw"]) if "equation_raw" in best_row_phys else best_eq_phys_pretty
            else:
                print("\n⚠️ Physics-aware ranking found no candidates; falling back to loss-only pick.")
                best_row_phys = (
                    min(valid_equations, key=lambda r: r["loss"])
                    if valid_equations else equations.loc[equations["loss"].idxmin()]
                )
                best_eq_phys_pretty = str(best_row_phys["simplified_equation"])
                best_eq_phys_raw = str(
                    best_row_phys["equation_raw"]) if "equation_raw" in best_row_phys else best_eq_phys_pretty

            # ---------------------------------------------------------
            # Build symbolic_feature_map BEFORE using it
            # ---------------------------------------------------------
            symbolic_feature_map = {}

            # Add engineered features
            if "engineered_feature_map" in locals() and engineered_feature_map:
                symbolic_feature_map.update(engineered_feature_map)

            # Hint template–based features are not used as columns in this pipeline
            # but we keep hook for future (hint_feature_map currently empty)

            # ============================================================
            # Final Comparison – Helper functions
            # ============================================================

            def _format_equation_readable(eq_str: str) -> str:
                if not isinstance(eq_str, str):
                    eq_str = str(eq_str)
                eq_str = eq_str.replace("**", "^").replace("⋅", "*")
                eq_str = eq_str.replace(" ", "")
                eq_str = re.sub(r"([+\-*/^()])", r" \1 ", eq_str)
                eq_str = re.sub(r"\s+", " ", eq_str)
                eq_str = eq_str.replace("( ", "(").replace(" )", ")")
                eq_str = eq_str.replace("+ -", "- ")
                eq_str = re.sub(r"([0-9]) \* ([A-Za-z\(])", r"\1*\2", eq_str)
                return eq_str.strip()

            print("\n" + "-" * 45)
            print("📘 FINAL EQUATION COMPARISON (Original Variables + Metrics)")
            print("-" * 45)

            results_summary = []


            # ---------------------------------------------------------
            # Metric evaluation in ORIGINAL / π / ENGINEERED SPACE
            # ---------------------------------------------------------

            # Ensure no duplicate columns before any derived-column operations
            original_features = dedupe_columns(original_features, keep="last")
            original_features = dedupe_columns(original_features, keep="last")

            # ----------------
            #-----------------

            def to_phor_units(expr_in_original_vars, use_dimless_target, ref_var_used, use_buckingham, pi_map):
                """
                If target was nondimensionalized during training (y_train = Phor/ref),
                convert the printed model back to Phor units:
                    Phor = (model_expr) * ref
                Here expr_in_original_vars is already restored to original variables (no Pi_*)
                """
                import sympy as sp

                if use_dimless_target != "y" or not ref_var_used:
                    return sp.simplify(expr_in_original_vars)

                # Build the reference expression in original variables
                ref_expr = sp.Symbol(ref_var_used, real=True)   # ✅ matches your cached symbols assumptions better

                if use_buckingham == "y" and isinstance(pi_map, dict) and pi_map:
                    # If ref is Pi_k, turn it into original-variable expression
                    # (this uses your existing restore_original_variables)
                    try:
                        ref_expr = restore_original_variables(ref_expr, pi_map)
                    except Exception:
                        pass

                try:
                    return sp.simplify(expr_in_original_vars * ref_expr)
                except Exception:
                    return expr_in_original_vars * ref_expr


            def _compute_scale_from_pi_map(pi_expr: str, df: pd.DataFrame) -> np.ndarray:
                """
                Evaluate a pi-map expression like 'd^-1.00 * Sl^1.00' on original_features.
                """
                import sympy as sp
                expr = pi_expr.replace("^", "**")
                sym = sp.sympify(expr, convert_xor=True)
                vars_needed = sorted([str(s) for s in sym.free_symbols])
                f = sp.lambdify([sp.Symbol(v) for v in vars_needed], sym, modules="numpy")
                vals = f(*[df[v].to_numpy() for v in vars_needed])
                return np.asarray(vals, dtype=float).flatten()


            def evaluate_in_orig_units(sym_expr, original_features, target_phor,
                                       features_pi=None, features_enriched=None,
                                       use_dimless_target="n",
                                       ref_var_used=None, pi_map=None):
                """
                Always returns metrics vs REAL Phor.

                If use_dimless_target == "y", sym_expr is assumed to predict (Phor/ref).
                We multiply predictions by ref (vector) to recover Phor before scoring.
                """
                import numpy as np
                import sympy as sp
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

                # ---- stable symbol order ----
                symbols = sorted(list(sym_expr.free_symbols), key=lambda s: str(s))
                names = [str(s) for s in symbols]
                if not names:
                    print("⚠ Equation has no free symbols; cannot evaluate.")
                    return np.nan, np.nan, np.nan, np.nan

                # ---- choose df that contains the variables used by the equation ----
                def _pick_df():
                    # 1) exact match in original
                    if all(v in original_features.columns for v in names):
                        return original_features
                    # 2) pi-space
                    if features_pi is not None and all(v in features_pi.columns for v in names):
                        return features_pi
                    # 3) enriched
                    if features_enriched is not None and all(v in features_enriched.columns for v in names):
                        return features_enriched
                    # last resort: original
                    return original_features

                df = _pick_df()
                missing = [v for v in names if v not in df.columns]
                if missing:
                    print(f"⚠ Cannot evaluate; missing variables: {missing}")
                    return np.nan, np.nan, np.nan, np.nan

                # ---- predict in training scale ----
                f = sp.lambdify(symbols, sym_expr, modules="numpy")
                args = [col_to_1d_numpy(df, v) for v in names]
                try:
                    y_pred_train = np.asarray(f(*args)).flatten()
                except Exception as e:
                    print(f"⚠ Evaluation failed: {e}")
                    return np.nan, np.nan, np.nan, np.nan

                if y_pred_train.size != target_phor.size:
                    print("⚠ y_pred dimension mismatch")
                    return np.nan, np.nan, np.nan, np.nan

                # ---- rescale prediction back to Phor if needed ----
                y_pred_phor = y_pred_train

                if use_dimless_target == "y":
                    if ref_var_used is None:
                        print("⚠ use_dimless_target=y but ref_var_used is None; cannot rescale to Phor.")
                        return np.nan, np.nan, np.nan, np.nan

                    # If ref_var is directly present in df, use it.
                    if ref_var_used in df.columns:
                        # Reference variable exists in the dataframe we're evaluating against
                        scale = col_to_1d_numpy(df, ref_var_used)
                    elif (original_features is not None) and (ref_var_used in original_features.columns):
                        # Common case when the equation is in π-space: ref_var_used (e.g. 'Sl') is only in original_features
                        scale = col_to_1d_numpy(original_features, ref_var_used)
                    elif ref_var_used.startswith("Pi_") and (pi_map is not None) and (ref_var_used in pi_map) and (original_features is not None):
                        # Reconstruct reference scale from π-map using original_features
                        scale = _compute_scale_from_pi_map(pi_map[ref_var_used], original_features)
                    else:
                        print(f"⚠ Cannot rescale: ref_var_used='{ref_var_used}' not available in df or original_features.")
                        return np.nan, np.nan, np.nan, np.nan

                    y_pred_phor = y_pred_train * np.asarray(scale).flatten()

                # ---- metrics vs REAL Phor ----
                r2 = r2_score(target_phor, y_pred_phor)
                mse = mean_squared_error(target_phor, y_pred_phor)
                mae = mean_absolute_error(target_phor, y_pred_phor)
                acc = 1 - mae / (np.mean(np.abs(target_phor)) + 1e-8)
                return r2, mse, mae, acc

            # ============================================================
            # 1) BEST PySR EQUATION
            # ============================================================
            print("\n---  BEST SYMBOLIC PySR EQUATION ---")
            print(f"Pretty: {best_eq_pretty}")  # optional
            best_symbolic_raw = best_eq_raw  # ✅ raw PySR string

            sym_best, raw_best = remap_equation(
                best_symbolic_raw,
                feature_map=symbolic_feature_map,
                debug=False,
            )

            sym_best_disp = round_sympy_constants(sym_best, sig=3)
            print("\nMapped form (π / engineered features space):")
            print(pretty_print_equation(sym_best_disp))
            print("Algebraic form:", str(sym_best_disp))

            # ---- Convert to original variables
            if use_buckingham == "y" and pi_map:
                sym_best_orig = restore_original_variables(sym_best, pi_map)
            else:
                sym_best_orig = sym_best

            sym_best_orig_disp = round_sympy_constants(sym_best_orig, sig=3)
            print("\n--- PySR Equation (Original Variables) ---")
            print(pretty_print_equation(sym_best_orig_disp))
            print("Algebraic:", str(sym_best_orig_disp))
            print("LaTeX:", latex_equation(sym_best_orig_disp))

            # --- Print as Phor in real units (undo target scaling if applied)
            sym_best_phor = to_phor_units(
                expr_in_original_vars=sym_best_orig,
                use_dimless_target=use_dimless_target,
                ref_var_used=ref_var_used,
                use_buckingham=use_buckingham,
                pi_map=pi_map,
            )

            sym_best_phor_disp = round_sympy_constants(sym_best_phor, sig=3)
            print("\n--- PySR Equation as Phor (REAL UNITS) ---")
            print("Phor =", pretty_print_equation(sym_best_phor_disp))
            print("Algebraic:", str(sym_best_phor_disp))
            print("LaTeX:", latex_equation(sym_best_phor_disp))

            # ---- Metrics (original space)
            r2_sym, mse_sym, mae_sym, acc_sym = evaluate_in_orig_units(
                sym_expr=sym_best,  # evaluate in the same space it was trained
                original_features=original_features,
                target_phor=target_phor,  # ALWAYS real Phor here
                features_pi=features,  # this is your Pi matrix when Buckingham used
                features_enriched=features_enriched,
                use_dimless_target=use_dimless_target,
                ref_var_used=ref_var_used,
                pi_map=pi_map
            )

            print(f"R²: {r2_sym:.4f} | MSE: {mse_sym:.4e} | MAE: {mae_sym:.4f} | Accuracy: {acc_sym:.4f}")

            results_summary.append((
                "Best PySR Equation",
                equations["loss"].min() if "loss" in equations.columns else np.nan,
                r2_sym, mse_sym, mae_sym, acc_sym
            ))

            # ============================================================
            # 2) PHYSICALLY CONSISTENT (DIMENSIONALLY VALID) EQUATION
            # ============================================================
            print("\n--- ✅ FINAL DIMENSIONALLY VALID & PHYSICS-AWARE EQUATION ---")

            if (not isinstance(best_eq_phys_raw, str)) or ("unparsed" in best_eq_phys_raw):
                print("[WARN] Physics-best equation unparseable; falling back to dimensionally-best.")
                best_eq_phys_raw = str(best_eq_dim)  # or better: dimensionally-best raw if you store it

            sym_phys, raw_phys = remap_equation(
                best_eq_phys_raw,  # ✅ use raw if available
                feature_map=symbolic_feature_map,
                debug=False,
            )

            sym_phys_disp = round_sympy_constants(sym_phys, sig=3)
            print("\nMapped form (π / engineered features space):")
            print(pretty_print_equation(sym_phys_disp))
            print("Algebraic form:", str(sym_phys_disp))

            # ---- Convert to original variables
            if use_buckingham == "y" and pi_map:
                sym_phys_orig = restore_original_variables(sym_phys, pi_map)
            else:
                sym_phys_orig = sym_phys

            sym_phys_orig_disp = round_sympy_constants(sym_phys_orig, sig=3)
            print("\n--- Physics-Aware Equation (Original Variables) ---")
            print(pretty_print_equation(sym_phys_orig_disp))
            print("Algebraic:", str(sym_phys_orig_disp))
            print("LaTeX:", latex_equation(sym_phys_orig_disp))

            # --- Print as Phor in real units (undo target scaling if applied)
            sym_phys_phor = to_phor_units(
                expr_in_original_vars=sym_phys_orig,
                use_dimless_target=use_dimless_target,
                ref_var_used=ref_var_used,
                use_buckingham=use_buckingham,
                pi_map=pi_map,
            )

            sym_phys_phor_disp = round_sympy_constants(sym_phys_phor, sig=3)
            print("\n--- Physics-Aware Equation as Phor (REAL UNITS) ---")
            print("Phor =", pretty_print_equation(sym_phys_phor_disp))
            print("Algebraic:", str(sym_phys_phor_disp))
            print("LaTeX:", latex_equation(sym_phys_phor_disp))

            # ---- Metrics (original space)
            r2_phys, mse_phys, mae_phys, acc_phys = evaluate_in_orig_units(
                sym_expr=sym_phys,
                original_features=original_features,
                target_phor=target_phor,
                features_pi=features,
                features_enriched=features_enriched,
                use_dimless_target=use_dimless_target,
                ref_var_used=ref_var_used,
                pi_map=pi_map
            )

            print(
                f"R²: {r2_phys:.4f} | MSE: {mse_phys:.4e} | MAE: {mae_phys:.4f} | Accuracy: {acc_phys:.4f} "
                f"| ΔR² vs PySR: {r2_phys - r2_sym:+.4f}"
            )

            results_summary.append((
                "Physically Consistent Eq",
                best_row_phys.get("loss", np.nan),
                r2_phys, mse_phys, mae_phys, acc_phys
            ))

            # ============================================================
            # 3) RE-DIMENSIONALIZED (FINAL PHYSICS-AWARE)
            # ============================================================
            print("\n--- 🧩 FINAL EQUATION AFTER PHYSICS (Original Variables) ---")

            expr_final = sym_phys_orig  # final = physics-aware equation
            expr_final_disp = round_sympy_constants(expr_final, sig=3)
            print(pretty_print_equation(expr_final_disp))
            print("Algebraic form:", str(expr_final_disp))
            print("LaTeX:", latex_equation(expr_final_disp))

            expr_final_phor = to_phor_units(
                expr_in_original_vars=expr_final,
                use_dimless_target=use_dimless_target,
                ref_var_used=ref_var_used,
                use_buckingham=use_buckingham,
                pi_map=pi_map,
            )

            expr_final_phor_disp= round_sympy_constants(expr_final_phor, sig=3)
            print("\n--- 🧩 FINAL EQUATION AFTER PHYSICS as Phor (REAL UNITS) ---")
            print("Phor =", pretty_print_equation(expr_final_phor_disp))
            print("Algebraic form:", str(expr_final_phor_disp))
            print("LaTeX:", latex_equation(expr_final_phor_disp))

            # ---- Metrics
            r2_final, mse_final, mae_final, acc_final = evaluate_in_orig_units(
                sym_expr=expr_final,
                original_features=original_features,
                target_phor=target_phor,
                features_pi=features,
                features_enriched=features_enriched,
                use_dimless_target=use_dimless_target,
                ref_var_used=ref_var_used,
                pi_map=pi_map
            )

            final_loss = None
            try:
                # If final equation is the same as chosen physics equation, reuse its loss.
                final_loss = best_row_phys.get("loss", np.nan) if isinstance(best_row_phys,
                                                                             (dict, pd.Series)) else np.nan
            except Exception:
                final_loss = np.nan

            results_summary.append((
                "Final Physics-Aware Eq",
                final_loss,
                r2_final, mse_final, mae_final, acc_final
            ))

            print(
                f"R²: {r2_final:.4f} | MSE: {mse_final:.4e} | MAE: {mae_final:.4f} | Accuracy: {acc_final:.4f} "
            )

            # ============================================================
            # SUMMARY TABLE + CLEAN SAVED OUTPUTS
            # ============================================================
            summary_df = pd.DataFrame(
                results_summary,
                columns=["Equation Type", "Loss", "R²", "MSE", "MAE", "Acc"],
            )

            print("\n🧮 Final Summary of All Evaluated Equations:")
            print(
                summary_df.to_string(
                    index=False,
                    justify="center",
                    float_format=lambda x: f"{x:.4e}" if abs(x) > 1 else f"{x:.4f}",
                )
            )

            # --- Save cleaner outputs ---
            def _nan_to_none(x):
                try:
                    if x is None:
                        return None
                    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                        return None
                    return float(x)
                except Exception:
                    return None

            try:
                # 1) Summary table
                summary_path = os.path.join(run_dir, "final_summary.csv")
                summary_df.to_csv(summary_path, index=False)
                print(f"\n[✅] Saved final summary table → {summary_path}")

                # 2) Final equations bundle (JSON)
                final_payload = {
                    "best_pysr": {
                        "equation_type": "best_pysr",
                        "algebraic": str(sym_best_orig_disp),
                        "latex": latex_equation(sym_best_orig_disp),
                        "loss": _nan_to_none(equations["loss"].min() if "loss" in equations.columns else np.nan),
                        "metrics": {
                            "r2": _nan_to_none(r2_sym),
                            "mse": _nan_to_none(mse_sym),
                            "mae": _nan_to_none(mae_sym),
                            "accuracy": _nan_to_none(acc_sym),
                        },
                    },
                    "physics_consistent": {
                        "equation_type": "physics_consistent",
                        "algebraic": str(sym_phys_orig_disp),
                        "latex": latex_equation(sym_phys_orig_disp),
                        "loss": _nan_to_none(best_row_phys.get("loss", np.nan)),
                        "metrics": {
                            "r2": _nan_to_none(r2_phys),
                            "mse": _nan_to_none(mse_phys),
                            "mae": _nan_to_none(mae_phys),
                            "accuracy": _nan_to_none(acc_phys),
                        },
                    },
                    "final_physics_aware": {
                        "equation_type": "final_physics_aware",
                        "algebraic": str(expr_final_disp),
                        "latex": latex_equation(expr_final_disp),
                        "loss": None,
                        "metrics": {
                            "r2": _nan_to_none(r2_final),
                            "mse": _nan_to_none(mse_final),
                            "mae": _nan_to_none(mae_final),
                            "accuracy": _nan_to_none(acc_final),
                        },
                    },
                    "meta": {
                        "train_r2_best_raw": _nan_to_none(meta.get("train_r2_best", np.nan)),
                        "used_buckingham": (use_buckingham == "y"),
                        "used_dimless_target": (use_dimless_target == "y"),
                        "ref_var_used_for_target": ref_var_used,
                    },
                }

                json_path = os.path.join(run_dir, "final_equations.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(final_payload, f, indent=2)
                print(f"[✅] Saved final equation bundle → {json_path}")
            except Exception as e:
                print(f"[WARN] Could not save final summary/equations: {e}")

            print("\nEnd of Final Comparison")
            print("=" * 90)

    except FileNotFoundError:
        print(f"\n❌ ERROR: Data file not found at '{DATA_FILE_PATH}'.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")