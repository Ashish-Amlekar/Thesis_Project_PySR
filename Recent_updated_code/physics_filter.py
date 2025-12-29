# physics_filter.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import sympy as sp


# ----------------------------- Data Model -----------------------------
@dataclass
class PhysicsHints:
    """
    General, domain-agnostic "soft constraints" for guiding and scoring equations.

    Can be built from config["physics_hints"], inferred automatically,
    or generated from an LLM.
    """
    # variable -> 'increasing' | 'decreasing' | 'unknown'
    monotonicity: Dict[str, str] = field(default_factory=dict)

    # global operator guidance
    allow_unary: Optional[List[str]] = None
    allow_binary: Optional[List[str]] = None
    forbid_unary: Optional[List[str]] = None
    forbid_binary: Optional[List[str]] = None

    # exponent bounds for '^' operator (min, max)
    power_bounds: Optional[Tuple[float, float]] = (-2.0, 2.0)

    # what forms to prefer; used to bias the post-ranking (e.g. ["multiplicative", "power-law", "log"])
    preferred_forms: List[str] = field(default_factory=list)

    # prioritize these variables or π-groups (e.g., ["Sratio", "Pi_4", "Fs/Wf"])
    dominant_variables: List[str] = field(default_factory=list)

    # ✅ NEW: candidate templates from knowledge base / LLM extraction
    candidate_templates: List[Dict[str, Any]] = field(default_factory=list)

    # complexity target; lower preferred
    max_nesting: int = 4

    # sampling scale for monotonicity check; fraction of std dev to perturb inputs
    perturb_fraction: float = 0.1
# ----------------------- π-group Hint Remapping -----------------------

def remap_hints_to_pi_space(hints: PhysicsHints, pi_map: dict) -> PhysicsHints:
    """
    Improved remapper: uses substring, token, and fuzzy similarity to map
    dominant variables and monotonicity hints into π-space.
    Also prints a debug mapping table for transparency.
    """

    def _normalize(s):
        return re.sub(r'[^a-z0-9]', '', s.lower())

    def _fuzzy_match(a, b, threshold=0.6):
        """Return True if fuzzy similarity > threshold."""
        return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() > threshold

    new_hints = PhysicsHints(**vars(hints))  # shallow copy
    debug_matches = []  # to store (orig_var, pi_name, expr, match_score)

    # --- Remap dominant variables ---
    if hints.dominant_variables:
        new_doms = set()
        for pi_name, expr in pi_map.items():
            expr_norm = _normalize(expr)
            for original_var in hints.dominant_variables:
                orig_norm = _normalize(original_var)
                similarity = SequenceMatcher(None, orig_norm, expr_norm).ratio()

                if (orig_norm in expr_norm) or (expr_norm in orig_norm) or similarity > 0.6:
                    new_doms.add(pi_name)
                    debug_matches.append((original_var, pi_name, expr, similarity))

        if new_doms:
            print(f"[🔄 Remap] Dominant variables remapped → {list(new_doms)}")
            new_hints.dominant_variables = list(new_doms)
        else:
            print("[ℹ️ Remap] No dominant variable matches found in π-groups (after fuzzy match).")

    # --- Remap monotonicity hints ---
    if hints.monotonicity:
        new_mono = {}
        for pi_name, expr in pi_map.items():
            expr_norm = _normalize(expr)
            for orig_var, trend in hints.monotonicity.items():
                orig_norm = _normalize(orig_var)
                similarity = SequenceMatcher(None, orig_norm, expr_norm).ratio()

                if (orig_norm in expr_norm) or (expr_norm in orig_norm) or similarity > 0.6:
                    new_mono[pi_name] = trend
                    debug_matches.append((orig_var, pi_name, expr, similarity))

        if new_mono:
            print(f"[🔄 Remap] Monotonicity remapped → {new_mono}")
            new_hints.monotonicity = new_mono
        else:
            print("[ℹ️ Remap] No monotonicity hints matched π-groups (after fuzzy match).")

    # --- Debug Table (deduplicated) ---
    if debug_matches:
        # Deduplicate by (original_var, pi_name)
        seen = set()
        unique_matches = []
        for orig_var, pi_name, expr, score in debug_matches:
            key = (orig_var, pi_name)
            if key not in seen:
                seen.add(key)
                unique_matches.append((orig_var, pi_name, expr, score))

        print("\n📊 [Remap Debug] Variable-to-π Mapping Table (deduplicated):")
        print(f"{'Original Variable':35} {'Mapped to π':10} {'π Expression':35} {'Match Score'}")
        print("-" * 95)
        for orig_var, pi_name, expr, score in unique_matches:
            print(f"{orig_var:35} {pi_name:10} {expr:35} {score:.2f}")
        print("-" * 95 + "\n")

    return new_hints


# -------------------------- Hints Construction ------------------------
# put this near the top of your file
PRINTED_HINTS_ONCE = False

def build_hints_from_config_or_auto(
    X: pd.DataFrame,
    y: pd.Series,
    cfg_hints: Optional[dict] = None,
) -> PhysicsHints:
    """
    Build PhysicsHints from config if present; otherwise infer monotonicity heuristically.
    """
    if cfg_hints:
        if not getattr(build_hints_from_config_or_auto, "_printed_once", False):
            print("--- ⚙️ Building Physics Hints from config.json ---")
            build_hints_from_config_or_auto._printed_once = True
        for k, v in cfg_hints.items():
            if k == "candidate_templates" and isinstance(v, list):
                print("  candidate_templates:")
                for i, t in enumerate(v, 1):
                    print(f"   [{i}] Expression:     {t.get('expr', 'N/A')}")
                    print(f"       Confidence:     {t.get('confidence', 'N/A')}")
                    print(f"       Source:         {t.get('source', 'N/A')}")
                    print(f"       Justification:  {t.get('justification', 'N/A')}\n")
            else:
                print(f"  {k}: {v}")
        return PhysicsHints(
            monotonicity=cfg_hints.get("monotonicity", {}),
            allow_unary=cfg_hints.get("allow_unary"),
            allow_binary=cfg_hints.get("allow_binary"),
            forbid_unary=cfg_hints.get("forbid_unary"),
            forbid_binary=cfg_hints.get("forbid_binary"),
            power_bounds=tuple(cfg_hints.get("power_bounds", (-2.0, 2.0))),
            preferred_forms=cfg_hints.get("preferred_forms", []),
            dominant_variables=cfg_hints.get("dominant_variables", []),
            candidate_templates=cfg_hints.get("candidate_templates", []),  # ✅ ADD THIS
            max_nesting=int(cfg_hints.get("max_nesting", 3)),
            perturb_fraction=float(cfg_hints.get("perturb_fraction", 0.1)),
        )

    # ---- Auto monotonicity (Spearman sign as weak prior) ----
    from scipy.stats import spearmanr

    print("\n--- ⚙️ Auto-inferring Physics Hints (no config provided) ---")
    monotonicity = {}
    for col in X.columns:
        try:
            rho, _ = spearmanr(X[col].values, y.values)
        except Exception:
            rho = 0.0
        if np.isnan(rho) or abs(rho) < 0.15:        # small |rho| => unknown
            monotonicity[col] = "unknown"
        elif rho > 0:
            monotonicity[col] = "increasing"
        else:
            monotonicity[col] = "decreasing"

    print("  → Monotonicity guesses:", monotonicity)

    return PhysicsHints(
        monotonicity=monotonicity,
        power_bounds=(-2.0, 2.0),
        preferred_forms=["multiplicative", "power-law", "logarithmic"],
        dominant_variables=[],
        max_nesting=3,
        perturb_fraction=0.1,
    )


# ----------------------- Pre-Processing (before PySR) -----------------

def apply_preprocessing_hints_to_pysr_params(
    translated_params: dict,
    hints: PhysicsHints,
) -> dict:
    """
    Improve preprocessing:
      ✅ Prune operators based on allow/forbid lists.
      ✅ Add log/power if preferred forms require it.
      ✅ Tighten power bounds if monotonic decreasing/increasing is known.
      ✅ Warn if dominant variables are not used in π-groups (debug).
    """
    out = dict(translated_params)  # shallow copy

    unary = set(out.get("unary_operators", []))
    binary = set(out.get("binary_operators", []))

    # --- 1️⃣ Prune operators explicitly disallowed ---
    if hints.allow_unary is not None:
        unary &= set(hints.allow_unary)
    if hints.allow_binary is not None:
        binary &= set(hints.allow_binary)
    if hints.forbid_unary:
        unary -= set(hints.forbid_unary)
    if hints.forbid_binary:
        binary -= set(hints.forbid_binary)

    # --- 2️⃣ Add operators if preferred forms hint at them ---
    if "logarithmic" in hints.preferred_forms:
        unary.add("log")
    if "power-law" in hints.preferred_forms:
        binary.add("^")
    if "multiplicative" in hints.preferred_forms:
        binary.add("*")

    # --- 3️⃣ Adjust power bounds based on monotonicity ---
    if hints.power_bounds is not None:
        lo, hi = hints.power_bounds
        if any(trend == "decreasing" for trend in hints.monotonicity.values()):
            # If many variables are decreasing, bias towards negative powers
            lo = min(lo, -1)
        search_params = dict(out.get("pysr_search_params", {}))
        constraints = dict(search_params.get("constraints", {}))
        if "^" in binary:
            constraints["^"] = (float(lo), float(hi))
        if constraints:
            search_params["constraints"] = constraints
        out["pysr_search_params"] = search_params

    # --- 4️⃣ Debug print summary ---
    print("\n--- 🔬 Physics-aware Preprocessing Hints Applied ---")
    print(f"✅ Unary operators pruned: {set(translated_params.get('unary_operators', [])) - unary}")
    print(f"✅ Binary operators pruned: {set(translated_params.get('binary_operators', [])) - binary}")
    print(f"🔧 Final unary operators: {sorted(unary)}")
    print(f"🔧 Final binary operators: {sorted(binary)}")
    print(f"✅ Applied power exponent bounds: {hints.power_bounds}")
    print(f"💡 Preferred forms: {', '.join(hints.preferred_forms)}")
    print(f"💡 Dominant variables: {', '.join(hints.dominant_variables) if hints.dominant_variables else 'None'}")

    out["unary_operators"] = sorted(unary)
    out["binary_operators"] = sorted(binary)
    return out


# --------------------- Post-Processing (after PySR) -------------------

def _count_nesting_depth(expr: sp.Expr) -> int:
    if expr.is_Atom:
        return 0
    return 1 + max(_count_nesting_depth(c) for c in expr.args) if expr.args else 0

def _token_complexity(expr: sp.Expr) -> int:
    return int(expr.count_ops())

def _pathology_score(expr: sp.Expr) -> float:
    s = sp.srepr(expr)
    penalties = 0.0
    penalties += 1.0 * len(re.findall(r'Log\(Log\(', s))
    penalties += 1.0 * len(re.findall(r'Exp\(Exp\(', s))
    penalties += 0.5 * len(re.findall(r'Pow\(.+?, Rational\(-1', s))
    penalties += 0.2 * len(re.findall(r'Pow\(.+?, Float\(', s))
    return penalties

def _approx_monotonicity_violations(f_lambda, X, var_names, hints, n_samples=64) -> float:
    if X.shape[0] == 0:
        return 0.0
    rng = np.random.default_rng(42)
    idxs = rng.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)
    Xs = X[idxs].copy()
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    scale = hints.perturb_fraction
    violations = 0
    total_checks = 0
    name_to_idx = {n: i for i, n in enumerate(var_names)}

    for vname, trend in hints.monotonicity.items():
        if trend not in ("increasing", "decreasing") or vname not in name_to_idx:
            continue
        j = name_to_idx[vname]
        delta = np.zeros_like(mu)
        delta[j] = scale * sd[j]
        for row in Xs:
            y0 = float(f_lambda(row.reshape(1, -1))[0])
            y_plus = float(f_lambda((row + delta).reshape(1, -1))[0])
            y_minus = float(f_lambda((row - delta).reshape(1, -1))[0])
            if trend == "increasing":
                if not (y_minus <= y0 + 1e-9 and y0 <= y_plus + 1e-9):
                    violations += 1
            else:
                if not (y_minus >= y0 - 1e-9 and y0 >= y_plus - 1e-9):
                    violations += 1
            total_checks += 1

    return (violations / total_checks) if total_checks > 0 else 0.0

def _preferred_form_score(expr, preferred_forms):
    """Score whether the expression matches preferred qualitative forms.

    Notes:
      - SymPy represents division as Mul(x, Pow(y, -1)), so we detect inverse via Pow(..., -1).
      - We also treat `inv(x)` or `safe_div(a,b)`-style strings as inverse-like.
    """
    s = sp.srepr(expr)
    expr_str = str(expr).replace(" ", "").lower()
    score = 0.0

    if "multiplicative" in preferred_forms:
        muls = s.count("Mul(")
        adds = s.count("Add(")
        if muls > 0:
            score += 0.5
        if adds == 0 and muls > 0:
            score += 0.5

    if "power-law" in preferred_forms and "Pow(" in s:
        score += 0.5

    if "logarithmic" in preferred_forms and "Log(" in s:
        score += 0.5

    # ✅ NEW: inverse preference
    if "inverse" in preferred_forms:
        inverse_hits = 0
        inverse_hits += len(re.findall(r"Pow\(.+?, Rational\(-1", s))
        inverse_hits += len(re.findall(r"Pow\(.+?, Float\(-1", s))
        if "inv(" in expr_str or "safe_div(" in expr_str:
            inverse_hits += 1
        if inverse_hits > 0:
            score += 0.5

    return score


def _dominance_score(expr, var_names, dominant_vars):
    if not dominant_vars:
        return 0.0
    symbols = {str(s) for s in expr.free_symbols}
    present = sum(int(d in symbols) for d in dominant_vars)
    return present / max(1, len(dominant_vars))

# --------------------- Improved Physics Scoring ---------------------


def _template_similarity(expr: sp.Expr, templates: List[Dict[str, Any]]) -> float:
    """
    Compute how structurally similar the given equation is to any candidate template.
    Returns a score between 0.0 and 1.0.
    """
    if not templates:
        return 0.0

    expr_str = str(expr).replace(" ", "").lower()
    best_score = 0.0

    for t in templates:
        t_expr = t.get("expr", "").replace(" ", "").lower()
        if not t_expr:
            continue

        # 1️⃣ Token-level similarity
        score = SequenceMatcher(None, expr_str, t_expr).ratio()

        # 2️⃣ Variable overlap bonus
        expr_syms = {str(s).lower() for s in expr.free_symbols}
        templ_syms = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", t_expr))
        if templ_syms:
            overlap = len(expr_syms & templ_syms) / len(templ_syms)
            score = 0.7 * score + 0.3 * overlap

        best_score = max(best_score, score)

    return best_score


def rank_and_select_physical_equations(
    equations_df: pd.DataFrame,
    X_df: pd.DataFrame,
    y: np.ndarray,
    hints: PhysicsHints,
    *,
    keep_only_dimensionally_valid: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Rank equations with a physics-aware score.

    Key changes vs your previous version:
      ✅ Accuracy score uses **validation MSE** (not training loss) when possible.
      ✅ Inverses are treated as a *valid* physics form (mildly penalized for instability, but not harshly).
      ✅ Slightly higher simplicity weight, plus an explicit pathology penalty term.
      ✅ Still returns a DataFrame with `physics_total_score` used by main.py.
    """
    if weights is None:
        # Slightly increased simplicity weight (small change) and correspondingly reduced accuracy.
        weights = {
            "acc": 0.63,
            "simplicity": 0.12,
            "monotonic": 0.08,
            "form": 0.10,
            "dominance": 0.07,
        }

    print("\n--- 🧪 Physics-aware Equation Ranking (Debug Mode) ---")

    def safe_sympify(expr_str: str) -> Optional[sp.Expr]:
        try:
            return sp.sympify(expr_str)
        except Exception:
            return None

    # --- Deterministic validation split (no extra inputs needed) ---
    X_full = X_df.to_numpy(dtype=float)
    y_full = np.asarray(y, dtype=float).reshape(-1)
    n = len(y_full)

    if n < 5:
        # too small to split meaningfully; fall back to training-based loss scaling
        idx_train = np.arange(n)
        idx_val = np.arange(n)
    else:
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        n_val = max(1, int(round(0.2 * n)))
        idx_val = perm[:n_val]
        idx_train = perm[n_val:]

    X_train, y_train = X_full[idx_train], y_full[idx_train]
    X_val, y_val = X_full[idx_val], y_full[idx_val]

    var_names = list(X_df.columns)

    # ------------------ First pass: compute raw metrics ------------------
    rows = []
    val_losses = []

    for _, row in equations_df.iterrows():
        if keep_only_dimensionally_valid and (not row.get("dimensionally_valid", True)):
            continue

        eq_str = row.get("simplified_equation", "")
        expr = safe_sympify(eq_str)
        if expr is None:
            print(f"[WARN] Skipping unparseable equation: {eq_str}")
            continue

        nesting = _count_nesting_depth(expr)
        ops = _token_complexity(expr)
        patho = _pathology_score(expr)

        # Simplicity: keep inverse penalty mild via _pathology_score; still penalize extreme nesting.
        simplicity_score = 1.0 / (
            1.0
            + 0.02 * ops
            + 0.5 * max(0, nesting - hints.max_nesting)
            + 0.35 * patho
        )

        f_lambda = row.get("lambda_format", None)
        if callable(f_lambda):
            viol = _approx_monotonicity_violations(f_lambda, X_train, var_names, hints)
            monotonic_score = 1.0 - viol

            # ✅ Validation loss in the SAME scale as `y` passed in
            try:
                pred = np.asarray(f_lambda(X_val)).reshape(-1)
                mask = np.isfinite(pred) & np.isfinite(y_val)
                if mask.sum() < max(3, int(0.25 * len(y_val))):
                    val_loss = float("inf")
                else:
                    err = pred[mask] - y_val[mask]
                    val_loss = float(np.mean(err * err))
            except Exception:
                val_loss = float("inf")
        else:
            monotonic_score = 0.5
            # If we can't evaluate, fall back to PySR's loss (training) but mark it.
            val_loss = float(row.get("loss", float("inf")))

        form_score = _preferred_form_score(expr, hints.preferred_forms)
        dom_score = _dominance_score(expr, var_names, hints.dominant_variables)
        template_score = _template_similarity(expr, hints.candidate_templates)

        new_row = row.copy()
        new_row["_nesting"] = nesting
        new_row["_ops"] = ops
        new_row["_patho"] = patho
        new_row["_val_loss"] = val_loss
        new_row["_simplicity_score"] = simplicity_score
        new_row["_monotonic_score"] = monotonic_score
        new_row["_form_score"] = form_score
        new_row["_dom_score"] = dom_score
        new_row["_template_score"] = template_score
        rows.append(new_row)

        if np.isfinite(val_loss):
            val_losses.append(val_loss)

    ranked = pd.DataFrame(rows)

    if ranked.empty:
        print("\n[WARN] All equations were unparseable or invalid for ranking.")
        print("      Falling back to PySR’s best equation (loss-minimum).")
        best_idx = equations_df["loss"].idxmin()
        fallback_row = equations_df.loc[best_idx]
        return equations_df, fallback_row

    # ------------------ Accuracy normalization using validation loss ------------------
    if len(val_losses) >= 3:
        lo, hi = np.percentile(np.asarray(val_losses, dtype=float), [5, 95])
    else:
        # fallback to training loss percentiles if validation is unusable
        losses = ranked.get("loss", pd.Series(np.nan)).astype(float).to_numpy()
        losses = losses[np.isfinite(losses)]
        if len(losses) == 0:
            lo, hi = (0.0, 1.0)
        else:
            lo, hi = np.percentile(losses, [5, 95])

    rng = max(1e-12, float(hi - lo))

    # ------------------ Second pass: acc_score + final physics score ------------------
    scored_rows = []
    for _, r in ranked.iterrows():
        eq_str = str(r.get("simplified_equation", ""))

        val_loss = float(r["_val_loss"])
        if not np.isfinite(val_loss):
            acc_score = 0.0
        else:
            acc_score = 1.0 - (np.clip(val_loss, lo, hi) - lo) / rng

        simplicity_score = float(r["_simplicity_score"])
        monotonic_score = float(r["_monotonic_score"])
        form_score = float(r["_form_score"])
        dom_score = float(r["_dom_score"])
        template_score = float(r["_template_score"])
        patho = float(r["_patho"])
        nesting = int(r["_nesting"])

        # Base weighted score
        physics_score = (
            weights["acc"] * acc_score
            + weights["simplicity"] * simplicity_score
            + weights["monotonic"] * monotonic_score
            + weights["form"] * form_score
            + weights["dominance"] * dom_score
        )

        # Template similarity bonus
        #for more accuracy consider lowering the template bonus slightly
        physics_score += 0.07 * template_score

        # Dominance coverage bonus
        #These bonuses are helpful only when candidates have similar accuracy.
        # When accuracy differs a lot, they shouldn’t flip the choice.
        if hints.dominant_variables:
            present = sum(1 for v in hints.dominant_variables if v.lower() in eq_str.lower())
            coverage_ratio = present / len(hints.dominant_variables)
            if coverage_ratio == 1.0:
                physics_score += 0.06
            elif coverage_ratio >= 0.5:
                physics_score += 0.03

        # ✅ Explicit pathology penalty (separate from simplicity)
        # This discourages fragile expressions even if they fit well.
        nesting_excess = max(0, nesting - hints.max_nesting)
        #for reducing penalty strength use
        #physics_score -= (0.04 * patho + 0.02 * nesting_excess)
        physics_score -= (0.05 * patho + 0.015 * nesting_excess)

        # Clamp for safety
        physics_score = float(np.clip(physics_score, 0.0, 1.2))

        print(f"\nEquation: {eq_str}")
        print(f"  ✔ Validation MSE:       {val_loss:.4e}" if np.isfinite(val_loss) else "  ✔ Validation MSE:       inf (eval failed)")
        print(f"  ✔ Accuracy Score:       {acc_score:.3f}")
        print(f"  ✔ Simplicity Score:     {simplicity_score:.3f}")
        print(f"  ✔ Monotonicity Score:   {monotonic_score:.3f}")
        print(f"  ✔ Preferred Form:       {form_score:.3f}")
        print(f"  ✔ Dominance Score:      {dom_score:.3f}")
        print(f"  ✔ Template Match Score: {template_score:.3f}")
        print(f"  → Final Physics Score:  {physics_score:.4f}")

        out_row = r.copy()
        out_row["physics_val_mse"] = val_loss
        out_row["physics_acc_score"] = acc_score
        out_row["physics_simplicity_score"] = simplicity_score
        out_row["physics_monotonic_score"] = monotonic_score
        out_row["physics_form_score"] = form_score
        out_row["physics_dominance_score"] = dom_score
        out_row["physics_total_score"] = physics_score
        scored_rows.append(out_row)

    scored = pd.DataFrame(scored_rows)
    scored = scored.sort_values(["physics_total_score", "physics_val_mse", "loss"], ascending=[False, True, True]).reset_index(drop=True)

    # this guarantees: you will not pick an equation whose validation MSE is more than ~10% worse than the best validating equation
    best_val = scored["physics_val_mse"].min()
    # allow at most 10% worse validation MSE than the best equation
    threshold = 1.10 * best_val

    filtered = scored[scored["physics_val_mse"] <= threshold].copy()
    if not filtered.empty:
        chosen = filtered.iloc[0]      # best physics among near-best accuracy
    else:
        chosen = scored.iloc[0]      # fallback
    return scored, chosen


#if you want accuracy to win but still be physics-aware
#weights = { "acc": 0.70,"simplicity": 0.10,"monotonic": 0.10,"form": 0.07,"dominance": 0.03}
#lower bonus : physics_score += 0.10 * template_score
# keep dominance coverage bonus but maybe halve it:
# full -> 0.08, half -> 0.04
#Add the 15% validation guardrail (best practical improvement). (commentd part before- chosen = scored.iloc[0] )
