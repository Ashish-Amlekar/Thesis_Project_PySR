"""
Universal equation_mapper.py – FIXED VERSION
Robust power parser for ANY PySR naming style.
"""

import re
import sympy as sp

# =====================================================
# GLOBAL SYMBOL CACHE  (prevents duplicate symbol bugs)
# =====================================================
_SYMBOL_CACHE = {}

def get_symbol(name: str):
    """Return the SAME Symbol object for each name."""
    if name not in _SYMBOL_CACHE:
        _SYMBOL_CACHE[name] = sp.Symbol(name, real=True)
    return _SYMBOL_CACHE[name]

# -----------------------------------------------------------
# 1. Base symbols (Pi_1 ... Pi_50)
# -----------------------------------------------------------
BASE_SYMBOLS = {f"Pi_{i}": get_symbol(f"Pi_{i}") for i in range(1, 51)}

ENGINEERED_PATTERN = re.compile(r"\bZ(\d+)\b")

SAFE_FUNCS = {
    "safe_div": lambda a, b: a / b,
    "inv": lambda x: 1 / x,
    "abs": sp.Abs,
    "exp_decay": lambda x: sp.exp(-x),
}

# -----------------------------------------------------------
# 2. CORRECT UNIVERSAL POWER PARSER  (Fixed)
# -----------------------------------------------------------
#
# Supported PySR cases:
#   Pi_3_pow_0_5
#   Pi_3_pow__0_5
#   Pi_3_pow___0_5
#   Pi_3_pow_-2
#   Pi_3_pow__-2
#   Pi_3_pow_2
#
# Also handles underscore → decimal conversion safely.
#
# -----------------------------------------------------------

POW_PATTERN = re.compile(
    r"(Pi_\d+)_pow_+([-+]?[0-9_]+(?:\.[0-9_]+)?)"
)

def replace_universal_pow(expr: str) -> str:
    """Safely convert ANY Pi_k_pow_* pattern to (Pi_k ** exponent)."""

    def repl(match):
        var = match.group(1)
        raw_exp = match.group(2)

        # Convert underscores to decimal points only inside exponents
        #
        # Examples:
        #   "0_5" → "0.5"
        #   "-0_25" → "-0.25"
        #   "2" → "2"
        #
        exp = raw_exp.replace("_", ".")

        try:
            exp_val = float(exp)
        except:
            # If conversion fails, leave original text unchanged
            return match.group(0)

        return f"({var}**({exp_val}))"

    return POW_PATTERN.sub(repl, expr)


# -----------------------------------------------------------
# 3. inv_Pi_k → 1/Pi_k
# -----------------------------------------------------------

INV_PATTERN = re.compile(r"\binv_Pi_(\d+)\b")

def replace_inv(expr: str) -> str:
    return INV_PATTERN.sub(lambda m: f"(1/Pi_{m.group(1)})", expr)


# -----------------------------------------------------------
# 4. one_minus_Pi_k → (1 - Pi_k)
# -----------------------------------------------------------

ONE_MINUS_PATTERN = re.compile(r"\bone_minus_Pi_(\d+)\b")

def replace_one_minus(expr: str) -> str:
    return ONE_MINUS_PATTERN.sub(lambda m: f"(1 - Pi_{m.group(1)})", expr)


# -----------------------------------------------------------
# 5. Replace engineered Z features
# -----------------------------------------------------------

def replace_engineered(expr: str, feature_map=None):
    if not feature_map:
        return expr

    def repl(match):
        key = f"Z{match.group(1)}"
        return f"({feature_map.get(key, key)})"

    return ENGINEERED_PATTERN.sub(repl, expr)


# -----------------------------------------------------------
# 6. Namespace builder
# -----------------------------------------------------------

def build_namespace(feature_map=None):
    ns = {}
    ns.update(BASE_SYMBOLS)

    ns.update({
        "log": sp.log,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
    })

    ns.update(SAFE_FUNCS)

    if feature_map:
        for k in feature_map:
            if k not in ns:
                ns[k] = get_symbol(k)

    return ns


# -----------------------------------------------------------
# 7. MAIN MAPPER (fixed)
# -----------------------------------------------------------

def remap_equation(pysr_eq: str, feature_map=None, debug=False):
    if pysr_eq is None:
        return None, None

    expr = pysr_eq.strip()

    # Step 1 – Normalize consecutive underscores (safe)
    expr = re.sub(r"(_){2,}", "_", expr)

    # Step 2 – SAFE universal pow replacement
    expr = replace_universal_pow(expr)

    # Step 3 – inv_Pi_k
    expr = replace_inv(expr)

    # Step 4 – one_minus_Pi_k
    expr = replace_one_minus(expr)

    # Step 5 – engineered features Zk
    expr = replace_engineered(expr, feature_map)

    # SAFE replacement of ^ → ** (parentheses enforced)
    expr = re.sub(r"(?<!\*)\^(?!\*)", "**", expr)

    expr = expr.replace("None", "0")

    ns = build_namespace(feature_map)

    try:
        # Preload all symbol names in this expr into namespace via cache
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
        for t in tokens:
            if t not in ns:
                ns[t] = get_symbol(t)

        sym = sp.sympify(expr, locals=ns)
    except Exception as e:
        if debug:
            print("[ERROR] sympify failed:", expr)
            print("Reason:", e)

        # Recovery fallback
        try:
            sym = sp.sympify(expr.replace("__", "_"), locals=ns)
        except Exception:
            # FINAL RECOVERY: use parse_expr with evaluate=False (keeps structure)
            try:
                sym = sp.parse_expr(expr, local_dict=ns, evaluate=False)
            except Exception:
                if debug:
                    print("[FATAL] Could not parse equation:", expr)
                return sp.sympify("unparsed_expr"), expr

    try:
        sym_simpl = sp.simplify(sym)
    except Exception:
        sym_simpl = sym

    return sym_simpl, expr


# -----------------------------------------------------------
# 8. Pretty & LaTeX printing
# -----------------------------------------------------------

def pretty_print_equation(sym_expr):
    if sym_expr is None:
        return "∅"
    try:
        return sp.pretty(sym_expr)
    except Exception:
        return str(sym_expr)

def latex_equation(sym_expr):
    if sym_expr is None:
        return "∅"
    try:
        return sp.latex(sym_expr)
    except Exception:
        return str(sym_expr)


# -----------------------------------------------------------
# 9. Wrapper
# -----------------------------------------------------------

def map_and_format(pysr_eq: str, feature_map=None, debug=False):
    sym, raw = remap_equation(pysr_eq, feature_map=feature_map, debug=debug)
    return {
        "sympy": sym,
        "pretty": pretty_print_equation(sym),
        "latex": latex_equation(sym),
        "raw_expanded": raw,
    }

# ================================================================
#   NEW: STRICT SYMBOLIC RESTORATION OF ORIGINAL VARIABLES
# ================================================================
import sympy as sp

def build_pi_substitution_map(pi_map, engineered_map=None):
    subs = {}

    # IMPORTANT: Pi_* symbols in the parsed equation are created via get_symbol(...),
    # so substitution keys MUST use get_symbol(...) too.
    for pi_name, expr_str in pi_map.items():
        expr_str = expr_str.replace("^", "**").replace(" ", "")
        expr_str = re.sub(r"([A-Za-z0-9_]+)\*\*(-?\d+\.?\d*)", r"\1**(\2)", expr_str)

        try:
            subs[get_symbol(pi_name)] = sp.sympify(expr_str)
        except Exception:
            print(f"[WARN] Could not parse π mapping for {pi_name}: {expr_str}")

    if engineered_map:
        for z_name, expr_str in engineered_map.items():
            try:
                subs[get_symbol(z_name)] = sp.sympify(expr_str)
            except Exception:
                print(f"[WARN] Could not parse engineered feature {z_name}")

    return subs


def restore_original_variables(expr_sympy, pi_map, engineered_map=None):
    import sympy as sp
    if expr_sympy is None:
        return None

    subs = build_pi_substitution_map(pi_map, engineered_map)

    try:
        restored = expr_sympy.xreplace(subs)
    except Exception as e:
        print(f"[WARN] xreplace failed during variable restoration: {e}")
        restored = expr_sympy

    # Make printed form nicer: (Wf**1.0)**1.279 -> Wf**1.279
    try:
        restored = sp.powdenest(restored, force=True)
    except Exception:
        pass

    try:
        restored = sp.simplify(restored)
    except Exception:
        pass

    return restored