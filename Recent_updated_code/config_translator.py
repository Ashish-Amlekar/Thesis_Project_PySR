# =============================
# config_translator.py (v4 – Dimension Filter Only)
# =============================

import numpy as np
import sympy as sp
import re
from typing import Dict, List, Tuple, Any
from collections import Counter

# -------- Canonical operator sets --------
_CANONICAL_UNARY = {"exp", "log", "sqrt", "inv", "abs"}
_CANONICAL_BINARY = {"+", "-", "*", "/", "^", "safe_div"}

# -------- Custom operator definitions --------
def safe_div(x, y):
    return np.where(np.abs(y) < 1e-12, 0.0, x / y)

def exp_decay(x, k=1.0):
    return np.exp(-k * x)

def inv(x):
    return 1.0 / (x + 1e-9)

_CUSTOM_OPS = {
    "safe_div": {
        "py": safe_div,
        "julia": "safe_div(x, y) = ifelse(abs(y) < 1e-12, 0.0, x / y)",
        "arity": 2,
    },
    "exp_decay": {
        "py": exp_decay,
        "julia": "exp_decay(x, k) = exp(-k * x)",
        "arity": 1,
    },
    "inv": {
        "py": inv,
        "julia": "inv(x) = 1 / (x + 1e-9)",
        "arity": 1,
    },
}

# -------- Helper functions --------
def _strip_func_syntax(op: str) -> str:
    return re.sub(r"\(.*\)", "", op).strip()

def _canonicalize(op: str) -> str:
    if not isinstance(op, str):
        return ""
    s = _strip_func_syntax(op).lower()
    synonyms = {
        "^": "^", "pow": "^", "power": "^",
        "plus": "+", "add": "+",
        "minus": "-", "sub": "-",
        "times": "*", "mult": "*", "mul": "*",
        "divide": "/", "div": "/",
    }
    s = synonyms.get(s, s)
    if s in _CANONICAL_UNARY or s in _CANONICAL_BINARY or s in _CUSTOM_OPS:
        return s
    return ""

def _sanitize_op_list(ops: List[str], allowed: set, arity: int) -> List[str]:
    clean = []
    for op in ops or []:
        c = _canonicalize(op)
        if not c or c not in allowed:
            continue
        if c in _CANONICAL_UNARY and arity == 1:
            clean.append(c)
        elif c in _CANONICAL_BINARY and arity == 2:
            clean.append(c)
        elif c in _CUSTOM_OPS and _CUSTOM_OPS[c]["arity"] == arity:
            clean.append(c)
    seen, out = set(), []
    for x in clean:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _collect_custom_ops(unary: List[str], binary: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    used = set([op for op in unary + binary if op in _CUSTOM_OPS])
    extra_sympy = {name: _CUSTOM_OPS[name]["py"] for name in used}
    extra_julia_defs = [_CUSTOM_OPS[name]["julia"] for name in used]
    return extra_sympy, extra_julia_defs

def translate_llm_config_to_pysr_params(llm_json: dict) -> dict:
    if isinstance(llm_json, list):
        llm_json = llm_json[0]
    dc = llm_json.get("direct_config", {})
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

    unary_ops = _sanitize_op_list(unary_raw, _CANONICAL_UNARY, 1)
    binary_ops = _sanitize_op_list(binary_raw, _CANONICAL_BINARY, 2)
    extra_sympy, _ = _collect_custom_ops(unary_ops, binary_ops)

    return {
        "unary_operators": unary_ops or ["exp", "log", "sqrt", "inv"],
        "binary_operators": binary_ops or ["+", "-", "*", "/"],
        "extra_sympy_mappings": extra_sympy,
        "pysr_search_params": dc.get("pysr_search_params", {}),
    }