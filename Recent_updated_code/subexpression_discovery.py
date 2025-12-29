# subexpression_discovery.py
import sympy as sp
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import re
import os


# --- Safe helpers -----------------------------------------------------------

def safe_div(a, b):
    """Elementwise division that returns 0.0 when denominator is 0 or NaN."""
    import numpy as np

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = a_arr / b_arr
        out[~np.isfinite(out)] = 0.0
    return out


def normalize_underscores(expr: str) -> str:
    """Collapse multiple underscores to a single one, for names like Pi_3_pow__0_5."""
    return re.sub(r"(_){2,}", "_", expr)


def _extract_subexpressions(expr, min_symbols=2):
    """
    Recursively extract sub-expressions with >= min_symbols.
    Returns a list of simplified sub-expressions as strings.
    """
    subs = []

    def _recurse(e):
        if e.is_Atom:
            return
        if len(e.free_symbols) >= min_symbols:
            subs.append(sp.simplify(e))
        for arg in e.args:
            _recurse(arg)

    _recurse(expr)
    return subs


def discover_repeated_subexpressions(equations, min_occurrences=3, min_symbols=2):
    """
    Identify repeated sub-expressions across multiple symbolic equations.
    Args:
        equations (list[str]): symbolic equations (string format).
        min_occurrences (int): threshold for how many times a sub-expression must appear.
        min_symbols (int): minimum number of symbols in a sub-expression to consider.

    Returns:
        dict: { "Z1": subexpr, "Z2": subexpr, ... }
    """
    counter = Counter()

    for eq in equations:
        try:
            # 🔧 Normalize underscores in the equation itself
            eq_norm = normalize_underscores(eq)
            expr = sp.sympify(eq_norm)
        except Exception:
            continue
        subs = _extract_subexpressions(expr, min_symbols=min_symbols)
        for s in subs:
            counter[str(s)] += 1

    # Filter by frequency
    frequent_subs = [(expr, count) for expr, count in counter.items() if count >= min_occurrences]

    # Sort by frequency (descending)
    frequent_subs.sort(key=lambda x: x[1], reverse=True)

    # Assign variable names Z1, Z2, ...
    result = {}
    for i, (expr_str, count) in enumerate(frequent_subs, 1):
        result[f"Z{i}"] = expr_str

    return result


def add_new_features_to_dataset(df: pd.DataFrame, new_features: dict) -> pd.DataFrame:
    """
    Create new columns in the dataset based on repeated sub-expressions.

    Args:
        df: Original dataframe (with Pi_1, Pi_2, ... columns)
        new_features: dict of {"Z1": "Pi_4 - Pi_3/Pi_1", ...}

    Returns:
        pd.DataFrame with new feature columns added.
    """
    new_df = df.copy()

    # 🔧 Normalize column names once here to single-underscore form
    new_df.columns = [normalize_underscores(str(c)) for c in new_df.columns]
    df_cols_norm = list(new_df.columns)

    # Prepare SymPy symbols USING normalized column names
    sympy_symbols = {c: sp.Symbol(c) for c in df_cols_norm}

    # Namespace for lambdify — CRUCIAL FIX
    lambdify_namespace = {
        "safe_div": safe_div,   # <-- FIX
        "np": np,
        "numpy": np,
        "sqrt": np.sqrt,
        "log": np.log,
        "exp": np.exp
    }

    for name, expr_str in new_features.items():
        try:
            # 🔧 Normalize the expression string (Pi_3_pow__0_5 -> Pi_3_pow_0_5)
            expr_str_norm = normalize_underscores(str(expr_str))

            # Parse expression with normalized names
            expr = sp.sympify(expr_str_norm, locals=sympy_symbols)

            # Create lambda including our namespace
            f = sp.lambdify(
                tuple(sympy_symbols.values()),
                expr,
                modules=[lambdify_namespace, "numpy"]
            )

            # Evaluate on the dataframe (use normalized column order)
            new_df[name] = f(*[new_df[c].to_numpy() for c in df_cols_norm])

            print(f"[✅] Created new feature: {name} = {expr_str_norm}")

        except Exception as e:
            print(f"[⚠️] Failed to compute {name}: {e}")

    return new_df