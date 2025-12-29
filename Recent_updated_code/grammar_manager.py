# grammar_manager.py
#Stores grammar/search params in a consistent JSON format.

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional


@dataclass
class GrammarState:
    """
    Lightweight container for grammar + PySR search parameters.
    This is general for any dataset.
    """
    unary_operators: List[str]
    binary_operators: List[str]
    power_bounds: Optional[Tuple[float, float]]  # (min, max) for '^', if used
    constraints: Dict[str, Any]
    pysr_search_params: Dict[str, Any]

    @classmethod
    def from_translated_params(cls, params: Dict[str, Any]) -> "GrammarState":
        """
        Build from whatever you currently pass to run_pysr_search.
        Expect params to have:
            - unary_operators
            - binary_operators
            - pysr_search_params (with optional 'constraints')
        """
        unary = list(params.get("unary_operators", []))
        binary = list(params.get("binary_operators", []))
        search_params = dict(params.get("pysr_search_params", {}))
        constraints = dict(search_params.get("constraints", {}))

        power_bounds = None
        if "^" in constraints:
            lo, hi = constraints["^"]
            power_bounds = (float(lo), float(hi))

        return cls(
            unary_operators=unary,
            binary_operators=binary,
            power_bounds=power_bounds,
            constraints=constraints,
            pysr_search_params=search_params,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # ensure tuples → lists for JSON
        if d.get("power_bounds") is not None:
            d["power_bounds"] = list(d["power_bounds"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GrammarState":
        pb = d.get("power_bounds")
        if pb is not None:
            pb = (float(pb[0]), float(pb[1]))
        return cls(
            unary_operators=list(d.get("unary_operators", [])),
            binary_operators=list(d.get("binary_operators", [])),
            power_bounds=pb,
            constraints=dict(d.get("constraints", {})),
            pysr_search_params=dict(d.get("pysr_search_params", {})),
        )

    def update_from_agent(self, update: Dict[str, Any]) -> None:
        """
        Apply updates coming from the agent.
        Allowed keys:
            - add_unary, remove_unary
            - add_binary, remove_binary
            - power_bounds
            - constraints (dict to merge)
            - pysr_search_params (dict to merge)
        """
        add_u = update.get("add_unary", [])
        rem_u = update.get("remove_unary", [])
        add_b = update.get("add_binary", [])
        rem_b = update.get("remove_binary", [])

        if add_u or rem_u:
            s = set(self.unary_operators)
            s |= set(add_u)
            s -= set(rem_u)
            self.unary_operators = sorted(s)

        if add_b or rem_b:
            s = set(self.binary_operators)
            s |= set(add_b)
            s -= set(rem_b)
            self.binary_operators = sorted(s)

        if "power_bounds" in update:
            pb = update["power_bounds"]
            if pb is None:
                self.power_bounds = None
                if "^" in self.constraints:
                    self.constraints.pop("^", None)
            else:
                lo, hi = float(pb[0]), float(pb[1])
                self.power_bounds = (lo, hi)
                self.constraints.setdefault("^", (lo, hi))

        if "constraints" in update:
            extra = update["constraints"]
            if isinstance(extra, dict):
                self.constraints.update(extra)

        if "pysr_search_params" in update:
            extra = update["pysr_search_params"]
            if isinstance(extra, dict):
                self.pysr_search_params.update(extra)