# =============================
# config_translator.py (v2)
# =============================

import numpy as np
import sympy as sp
import re
from typing import Dict, List, Tuple, Any

# --------- Canonical operator registries (strict) ---------
# Only operators known to be supported by PySR/Julia without extra defs.
_CANONICAL_UNARY = {
    "sin", "cos", "tan", "tanh", "sinh", "cosh",
    "exp", "log", "sqrt", "abs"
}

_CANONICAL_BINARY = {
    "+", "-", "*", "/","^" #"pow"
}

# --------- Safe custom operator registry (Python + Julia) ---------
# We only allow these four custom operators (stable across code & Julia).

def safe_div(x, y):
    return x / (y + 1e-9)


def safe_log(x):
    import numpy as _np
    return _np.log(_np.abs(x) + 1e-9)


def exp_decay(x):
    import numpy as _np
    return _np.exp(-x)


def seg_affinity(x):
    return 1.0 - x ** 2.0

# We only allow these four custom operators (stable across code & Julia).
_CUSTOM_OPS = {
    "safe_div": {
        "py": safe_div,
        "julia": "safe_div(x, y) = x / (y + 1e-9)",
        "arity": 2,
    },
    "safe_log": {
        "py": safe_log,
        "julia": "safe_log(x) = log(abs(x) + 1e-9)",
        "arity": 1,
    },
    "exp_decay": {
        "py": exp_decay,
        "julia": "exp_decay(x) = exp(-x)",
        "arity": 1,
    },
    "seg_affinity": {
        "py": seg_affinity,
        "julia": "seg_affinity(x) = 1 - x^2",
        "arity": 1,
    },
}


# --------- Helpers ---------

def _strip_func_syntax(op: str) -> str:
    # "exp(x)" -> "exp"
    return re.sub(r"\(.*\)", "", op).strip()


def _canonicalize(op: str) -> str:
    """
    Map arbitrary tokens to a strict canonical set. Unknown -> "" (drop).
    This avoids Julia errors from unknown symbols. No ad-hoc aliases beyond this map.
    """
    if not isinstance(op, str):
        return ""
    s = _strip_func_syntax(op).lower()
    synonym_map = {
        "^":"^",
        "pow":"^", #: "pow",
        "power": "^", #"pow",
        #"pow": "pow",
        "plus": "+",
        "add": "+",
        "minus": "-",
        "sub": "-",
        "times": "*",
        "mult": "*",
        "mul": "*",
        "divide": "/",
        "div": "/",
    }
    s = synonym_map.get(s, s)

    # Builtins
    if s in _CANONICAL_UNARY or s in _CANONICAL_BINARY:
        return s

    # Approved custom ops
    if s in _CUSTOM_OPS:
        return s

    # Unknown -> drop
    return ""


def _sanitize_op_list(ops: List[str], allowed_set: set, desired_arity: int) -> List[str]:
    """Keep only ops that match desired arity and are allowed (built-in or custom)."""
    clean = []
    for op in ops or []:
        c = _canonicalize(op)
        if not c or c not in allowed_set:
            continue
        if c in _CANONICAL_UNARY and desired_arity == 1:
            clean.append(c)
        elif c in _CANONICAL_BINARY and desired_arity == 2:
            clean.append(c)
        elif c in _CUSTOM_OPS and _CUSTOM_OPS[c]["arity"] == desired_arity:
            clean.append(c)
    # Remove duplicates while preserving order
    seen = set()
    out = []
    for x in clean:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _sanitize_nested_constraints(nc: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Keep only constraints for known operators; clip values to {0,1}."""
    if not isinstance(nc, dict):
        return {}
    out = {}
    for parent, inner in nc.items():
        p = _canonicalize(parent)
        if not p:
            continue
        inner_clean = {}
        if isinstance(inner, dict):
            for child, val in inner.items():
                c = _canonicalize(child)
                if not c:
                    continue
                inner_clean[c] = 1 if int(val) > 0 else 0
        if inner_clean:
            out[p] = inner_clean
    # pow cannot be nested-constrained in PySR (it causes issues); drop its entry.
    out.pop("^", None) #pow
    return out


def _sanitize_complexity_map(cm: dict, allowed_ops: set) -> dict:
    if not isinstance(cm, dict):
        return {}
    out = {}
    for k, v in cm.items():
        c = _canonicalize(k)
        if not c or c not in allowed_ops:
            continue
        try:
            # Force integer conversion (round if float)
            iv = int(round(float(v)))
        except Exception:
            continue
        out[c] = iv
    return out


def _collect_custom_ops(unary_ops: List[str], binary_ops: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (extra_sympy_mappings, extra_julia_defs) for any referenced custom ops.
    - extra_sympy_mappings: safe to pass directly into PySRRegressor
    - extra_julia_defs: list of Julia operator definitions (strings) for manual registration
    """
    used = set([op for op in unary_ops + binary_ops if op in _CUSTOM_OPS])
    extra_sympy = {name: _CUSTOM_OPS[name]["py"] for name in used}
    extra_julia_defs = [_CUSTOM_OPS[name]["julia"] for name in used]
    return extra_sympy, extra_julia_defs


def _sanitize_pysr_hyperparams(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Whitelist & sanity-check PySR hyperparameters coming from the LLM."""
    if not isinstance(raw, dict):
        return {}

    allowed_keys = {
        "niterations": int,
        "populations": int,
        "population_size": int,
        "parsimony": float,
        "procs": int,
        "model_selection": str,
        "precision": int,
        "elementwise_loss": str,  # e.g., "L2DistLoss()"
    }
    sanitized = {}

    for k, typ in allowed_keys.items():
        if k not in raw:
            continue
        v = raw[k]
        try:
            if typ is int:
                v = int(v)
            elif typ is float:
                v = float(v)
            elif typ is str:
                v = str(v)
        except Exception:
            print(f"Skipping invalid hyperparameter {k}={v!r}")
            continue

        # Basic bounds
        if k in {"niterations", "populations", "population_size"}:
            if v <= 0:
                continue
        if k == "parsimony" and v < 0:
            continue
        if k == "precision" and v not in {32, 64}:
            v = 64

        sanitized[k] = v

    return sanitized


def translate_llm_config_to_pysr_params(llm_json: dict) -> dict:
    dc = llm_json.get("direct_config", {})
    ind = llm_json.get("indirect_config_suggestions", {})

    grammar = dc.get("grammar", {})

    def _extract_list(lst):
        out = []
        for item in lst or []:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and "name" in item:
                out.append(item["name"])
        return out

    unary_raw = _extract_list(grammar.get("unary_operators", []))
    binary_raw = _extract_list(grammar.get("binary_operators", []))

    unary_ops = _sanitize_op_list(unary_raw, _CANONICAL_UNARY, desired_arity=1)
    binary_ops = _sanitize_op_list(binary_raw, _CANONICAL_BINARY, desired_arity=2)

    nested_constraints = _sanitize_nested_constraints(dc.get("nested_constraints", {}))

    allowed_ops = set(unary_ops + binary_ops)
    complexity_map = _sanitize_complexity_map(dc.get("complexity_of_operators", {}), allowed_ops)

    constraints = {}
    if "^" in binary_ops: #pow
        constraints["^"] = (-1, 1) #pow

    # Collect mappings
    extra_sympy_mappings, extra_julia_defs = _collect_custom_ops(unary_ops, binary_ops)

    pysr_hparams = _sanitize_pysr_hyperparams(dc.get("pysr_search_params", {}))

    out = {
        "unary_operators": unary_ops,
        "binary_operators": binary_ops,
        "nested_constraints": nested_constraints,
        "complexity_of_operators": complexity_map,
        "constraints": constraints,
        "extra_sympy_mappings": extra_sympy_mappings,  # valid for PySR
    }
    if pysr_hparams:
        out["pysr_search_params"] = pysr_hparams

    # Attach Julia defs for later manual registration
    if extra_julia_defs:
        out["extra_julia_defs"] = extra_julia_defs

    # -------------------------------
    # Handle indirect_config_suggestions → custom_operator_suggestions
    # -------------------------------
    #custom_ops_raw = ind.get("custom_operator_suggestions", [])
    #sanitized_custom_ops = []
    #for item in custom_ops_raw:
    #    if isinstance(item, dict) and "name" in item:
            # Strip arguments from names like "exp_decay(x, k)" → "exp_decay"
    #        name = item["name"].split("(")[0].strip()
     #       sanitized_custom_ops.append(name)
      #  elif isinstance(item, str):
       #     sanitized_custom_ops.append(item.split("(")[0].strip())

    #if sanitized_custom_ops:
     #   out["custom_operator_suggestions"] = sanitized_custom_ops
    # Ensure we don’t pass empty operator lists (causes Julia Any errors)
    if not unary_ops:
        print("[WARN] No unary operators provided, defaulting to ['exp', 'log', 'sqrt']")
        unary_ops = ["exp", "log", "sqrt"]

    if not binary_ops:
        print("[WARN] No binary operators provided, defaulting to ['+', '-', '*', '/']")
        binary_ops = ["+", "-", "*", "/"]
    return out


# --------- Physics-informed penalty (for reranking) ---------
class PhysicsInformedPenalty:
    """
    Computes ONLY the physics penalty term (scale-normalized), not MSE.
    Combine externally as: total = julia_loss + alpha * physics_penalty.
    """

    def __init__(self, X: np.ndarray, feature_names: List[str], suggestions: list, target_dimension: str = "L", alpha: float = 3.0,debug: bool = True):
        self.X = X
        self.feature_index = {name: i for i, name in enumerate(feature_names)}
        self.suggestions = suggestions or []
        self.target_dimension = target_dimension
        self.alpha = alpha
        self.debug = debug

        # Define dimensions of input variables (must adjust for your problem)
        # Example: "L" = length, "-" = dimensionless
        self.dimensions = {
            "Wf": "-",  # weight fraction (dimensionless)
            "Sratio": "-",  # size ratio (dimensionless)
            "Fs": "-",  # fraction small (dimensionless)
            "Agg": "-",  # aggregation factor (dimensionless)
            "d": "L",  # droplet size (length)
            "Ss": "L",  # small particle size (length)
            "Sl": "L",  # large particle size (length)
            "Fl":"-" #fraction large (dimensionless)
        }

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, equation_str: str = None) -> float:
        # Guard against non-finite predictions
        if not np.all(np.isfinite(y_pred)):
            # Heavy penalty if model explodes/NaNs
            return 1e6

        scale = float(np.std(y_true))
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0

        n = len(y_pred)
        penalty = 0.0

        for rule in self.suggestions:
            cond = str(rule.get("condition", "")).lower()
            if not cond:
                continue

            partial = 0.0
            if "non-negative" in cond or "nonpositive" in cond or "non negative" in cond:
                neg = y_pred[y_pred < 0]
                partial += np.sum(np.abs(neg)) / scale

            if "decrease with increasing wf" in cond:
                partial += self._monotonicity(y_pred, feature="Wf", increasing=False) / scale

            if "increase with increasing sratio" in cond:
                partial += self._monotonicity(y_pred, feature="Sratio", increasing=True) / scale

            if "increase with increasing fs" in cond:
                partial += self._monotonicity(y_pred, feature="Fs", increasing=True) / scale

            if "dimension" in cond and equation_str is not None:
                partial += self._dimension_penalty(equation_str)

            # Normalize per sample and by target scale
            penalty += partial / (n * scale)

            if self.debug and partial > 0:
                print(f"[PENALTY] Rule '{cond}' -> raw={partial:.3f}, contrib={partial / (n * scale):.6f}")

        return self.alpha * float(penalty)

    def _monotonicity(self, y_pred: np.ndarray, feature: str, increasing: bool) -> float:
        idx = self.feature_index.get(feature)
        if idx is None:
            return 0.0
        x = self.X[:, idx]
        order = np.argsort(x)
        yp = y_pred[order]
        diffs = np.diff(yp)
        if increasing:
            violations = diffs[diffs < 0]
        else:
            violations = diffs[diffs > 0]
        return float(np.sum(np.abs(violations)))

    def _dimension_penalty(self, equation_str: str) -> float:
        try:
            expr = sp.sympify(equation_str.split("=")[-1].strip())
            units = {name: sp.Symbol(self.dimensions.get(name, "-")) for name in self.dimensions}

            def infer_dim(expr):
                if expr.is_Symbol:
                    return units.get(str(expr), "-")
                if expr.is_Number:
                    return "-"
                if expr.is_Add:
                    dims = [infer_dim(arg) for arg in expr.args]
                    if all(d == dims[0] for d in dims):
                        return dims[0]
                    return "?"  # mismatch
                if expr.is_Mul:
                    dims = [infer_dim(arg) for arg in expr.args]
                    return "*".join(sorted([d for d in dims if d != "-"])) or "-"
                if expr.is_Pow:
                    base, exp = expr.args
                    base_dim = infer_dim(base)
                    if exp.is_Number:
                        return f"({base_dim})^{int(exp)}"
                    return "?"
                return "?"

            final_dim = infer_dim(expr)
            if final_dim != self.target_dimension:
                print(f"[PHYSICS] Dimensional mismatch: got {final_dim}, expected {self.target_dimension}")
                # proportional penalty: more mismatches → higher penalty
                return 0.3 * (1.0 + final_dim.count("?"))
            return 0.0
        except Exception as e:
            if self.debug:
                print(f"[WARN] Dimension check failed: {e}")
            return 0.3