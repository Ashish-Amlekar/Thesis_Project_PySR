# dimensional_filter.py
import sympy as sp
import re
from typing import Dict, Tuple, Optional, List

DIMENSION_SYMBOLS = ["M", "L", "T", "I", "Θ", "N", "J"]


class Dimension:
    def __init__(self, exponents=None):
        if exponents is None:
            exponents = {d: 0 for d in DIMENSION_SYMBOLS}
        self.exp = exponents

    @classmethod
    def from_string(cls, s: str):
        exponents = {d: 0 for d in DIMENSION_SYMBOLS}

        # --- PATCH: make sure s is a string ---
        if not isinstance(s, str):
            s = str(s)

        s = s.strip()

        # empty or dimensionless
        if s in ["-", "1", ""]:
            return cls(exponents)
        tokens = re.findall(r"([A-ZΘNJ])(?:\^?(-?\d+(\.\d+)?))?", s.replace("**", "^"))
        for dim, power, _ in tokens:
            exponents[dim] += float(power) if power else 1.0
        return cls(exponents)

    def __mul__(self, other):
        return Dimension({d: self.exp[d] + other.exp[d] for d in DIMENSION_SYMBOLS})

    def __truediv__(self, other):
        return Dimension({d: self.exp[d] - other.exp[d] for d in DIMENSION_SYMBOLS})

    def __pow__(self, power: float):
        return Dimension({d: self.exp[d] * power for d in DIMENSION_SYMBOLS})

    def __eq__(self, other):
        return all(abs(self.exp[d] - other.exp[d]) < 1e-6 for d in DIMENSION_SYMBOLS)

    def is_dimensionless(self):
        return all(abs(self.exp[d]) < 1e-6 for d in DIMENSION_SYMBOLS)

    def __repr__(self):
        return " ".join([f"{d}^{p:.2f}" for d, p in self.exp.items() if abs(p) > 1e-6]) or "1"


class DimensionalFilter:
    def __init__(self, var_dims: Dict[str, str], target_dim: str, constants: Optional[Dict[str, str]] = None):
        self.var_dims = {var: Dimension.from_string(dim) for var, dim in var_dims.items()}
        self.constants = {c: Dimension.from_string(dim) for c, dim in (constants or {}).items()}
        self.target_dim = Dimension.from_string(target_dim)

    def _get_symbol_dim(self, symbol: str) -> Dimension:
        # 1) Always prefer explicit dims from config (raw + engineered)
        if symbol in self.var_dims:
            return self.var_dims[symbol]
        if symbol in self.constants:
            return self.constants[symbol]

        # 2) Auto-generated features Z1, Z2... (dimensionless by default unless you add them to var_dims)
        if symbol.startswith("Z"):
            return Dimension.from_string("1")

        # 3) Unknown symbols default dimensionless (you can make this strict later if desired)
        return Dimension.from_string("1")

    def analyze_equation(self, eq: str) -> Tuple[bool, Dimension, str]:
        try:
            expr = sp.sympify(eq)
        except Exception as e:
            return False, Dimension.from_string("1"), f"Parsing error: {e}"

        valid, dim, reason = self._check_expr(expr)
        if valid and not (dim == self.target_dim):
            return False, dim, f"Output dimension {dim} does not match target {self.target_dim}"
        return valid, dim, reason

    def compute_dimension(self, eq: str) -> Tuple[bool, Dimension, str]:
        """
        Like analyze_equation, but does NOT enforce matching self.target_dim.
        Useful for inferring dims of engineered features.
        """
        try:
            expr = sp.sympify(eq)
        except Exception as e:
            return False, Dimension.from_string("1"), f"Parsing error: {e}"

        return self._check_expr(expr)

    def _contains_symbol(self, expr) -> bool:
        # True if expr contains any non-constant symbols (Wf, Ss, etc.)
        return len(expr.free_symbols) > 0

    def _check_expr(self, expr) -> Tuple[bool, Dimension, str]:
        if expr.is_Number:
            return True, Dimension.from_string("1"), ""
        if expr.is_Symbol:
            return True, self._get_symbol_dim(str(expr)), ""

        # -------- Handle your custom unary/binary ops correctly ----------
        # sympify("inv(x)") becomes a sympy Function with func name "inv"
        if hasattr(expr, "func") and expr.func == sp.Function("inv"):
            v, d, r = self._check_expr(expr.args[0])
            if not v:
                return False, d, r
            return True, d ** (-1.0), ""

        if hasattr(expr, "func") and expr.func == sp.Function("safe_div"):
            v1, d1, r1 = self._check_expr(expr.args[0])
            if not v1:
                return False, d1, r1
            v2, d2, r2 = self._check_expr(expr.args[1])
            if not v2:
                return False, d2, r2
            return True, d1 / d2, ""

        if hasattr(expr, "func") and expr.func == sp.Function("exp_decay"):
            v, d, r = self._check_expr(expr.args[0])
            if not v:
                return False, d, r
            if not d.is_dimensionless():
                return False, d, f"Function 'exp_decay' requires dimensionless arg (got {d})"
            return True, Dimension.from_string("1"), ""
        # ---------------------------------------------------------------

        # Addition/subtraction: only allow dimensionless NUMERIC constants
        # to be added to a dimensional quantity. Do NOT allow dimensionless variables (like Wf).
        if expr.func == sp.Add:
            term_dims = []
            for arg in expr.args:
                v, d, r = self._check_expr(arg)
                if not v:
                    return False, d, r
                term_dims.append((arg, d))

            non_dimless = [(a, d) for (a, d) in term_dims if not d.is_dimensionless()]
            dimless = [(a, d) for (a, d) in term_dims if d.is_dimensionless()]

            if len(non_dimless) == 0:
                return True, Dimension.from_string("1"), ""

            # If there are multiple dimensional terms, they must match
            base_dim = non_dimless[0][1]
            for (a, d) in non_dimless[1:]:
                if d != base_dim:
                    return False, base_dim, f"Addition/Subtraction terms have inconsistent dimensions: {[dd for _,dd in term_dims]}"

            # If any dimensionless term contains symbols (e.g. Wf, Fs), reject
            for (a, d) in dimless:
                if self._contains_symbol(a):
                    return False, base_dim, f"Cannot add dimensionless variable/expression '{a}' to dimensional quantity."

            return True, base_dim, ""

        # Multiplication
        if expr.func == sp.Mul:
            total = Dimension.from_string("1")
            for arg in expr.args:
                v, d, r = self._check_expr(arg)
                if not v:
                    return False, d, r
                total *= d
            return True, total, ""

        # Power
        if expr.func == sp.Pow:
            base_v, base_d, base_r = self._check_expr(expr.args[0])
            exp = expr.args[1]
            if not base_v:
                return False, base_d, base_r
            if exp.is_Number:
                return True, base_d ** float(exp), ""
            else:
                exp_v, exp_d, exp_r = self._check_expr(exp)
                if not exp_d.is_dimensionless():
                    return False, base_d, f"Exponent must be dimensionless (got {exp_d})"
                return True, base_d, ""

        # log/exp/sin/cos/tan require dimensionless args
        if expr.func in [sp.log, sp.exp, sp.sin, sp.cos, sp.tan]:
            for arg in expr.args:
                v, d, r = self._check_expr(arg)
                if not d.is_dimensionless():
                    return False, d, f"Function '{expr.func.__name__}' requires dimensionless arg (got {d})"
            return True, Dimension.from_string("1"), ""

        # Fallback: treat unknown n-ary funcs as invalid unless dimensionless-safe
        # (Safer than multiplying dims, which is wrong for most functions.)
        if isinstance(expr, sp.Function):
            # If all args are dimensionless, assume dimensionless output; else reject.
            for arg in expr.args:
                v, d, r = self._check_expr(arg)
                if not d.is_dimensionless():
                    return False, d, f"Unknown function '{expr.func}' with non-dimensionless arg (got {d})"
            return True, Dimension.from_string("1"), ""

        # Final fallback: multiply dims of args (rarely reached now)
        total = Dimension.from_string("1")
        for arg in expr.args:
            v, d, r = self._check_expr(arg)
            if not v:
                return False, d, r
            total *= d
        return True, total, ""

    def is_valid(self, eq: str) -> bool:
        valid, dim, reason = self.analyze_equation(eq)
        return valid and dim == self.target_dim