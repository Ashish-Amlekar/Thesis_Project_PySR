# llm_agent.py
#Uses existing llm_handler.call_llm_api & configure_llm to decide next action.

import json
import os
from typing import Dict, Any, List

from mcp_context_manager import MCPContextManager
from llm_handler import configure_llm, call_llm_api


class LLMAgent:
    """
    Central 'brain' for Phase 2.
    It reads context files for a given iteration and decides what to do next.
    """

    def __init__(self, context_manager: MCPContextManager):
        self.ctx = context_manager
        configure_llm()  # reuse your existing configuration

    def _build_prompt(self, iteration: int) -> str:
        folder = self.ctx.get_iteration_folder(iteration)

        # Load available JSON context files
        pysr_results = self.ctx.load_pysr_results(iteration) or {}
        data_stats = self.ctx.load_data_stats(iteration) or {}
        grammar = self.ctx.load_grammar(iteration) or {}
        system_state = self.ctx.load_system_state(iteration) or {}

        # We keep content summarized to avoid huge prompts
        pysr_summary = {
            "best_equation": pysr_results.get("best_equation"),
            "best_metrics": pysr_results.get("best_metrics"),
            "n_equations": len(pysr_results.get("equations", [])),
            "loss_range": pysr_results.get("loss_range"),
        }

        # Only head of equations for context
        eqs = pysr_results.get("equations", [])
        preview_equations = eqs[:5] if isinstance(eqs, list) else []

        prompt = f"""
You are a symbolic regression research assistant acting as a stateful agent.

You receive project state in JSON form from iteration {iteration}.

You MUST output a single JSON object with the following schema:
{{
  "action": "REFINE_GRAMMAR" | "TUNE_HYPERPARAMETERS" | "ADD_ENGINEERED_FEATURES" |
            "ENABLE_PI" | "DISABLE_PI" | "RESET_SEARCH" | "TERMINATE",
  "update": {{ ... }},  // keys depend on action
  "reason": "short natural language explanation (1-3 sentences)"
}}

Current summarized context (JSON objects):

[PySR summary]
{json.dumps(pysr_summary, indent=2)}

[Sample equations (max 5)]
{json.dumps(preview_equations, indent=2)}

[Data stats summary]
{json.dumps(data_stats, indent=2)[:3000]}

[Current grammar state]
{json.dumps(grammar, indent=2)}

[System state]
{json.dumps(system_state, indent=2)}

Guidelines:
- If best_metrics['R2'] is already very high (>=0.98) and complexity is moderate, consider "TERMINATE".
- If many equations are too complex or unstable, choose "REFINE_GRAMMAR" to simplify operators or tighten power_bounds.
- If performance is mediocre but stable, choose "TUNE_HYPERPARAMETERS" to adjust niterations, population_size, etc.
- If equations repeatedly reuse the same sub-expressions and data_stats indicate nonlinearity, choose "ADD_ENGINEERED_FEATURES".
- If dimensionless groups seem harmful (very poor R2 vs original PySR equation), consider "DISABLE_PI".
- Always ensure 'update' is a small, incremental change (do NOT radically change everything at once).

Now output ONLY the JSON object, nothing else.
"""
        return prompt

    def decide_next_action(self, iteration: int) -> Dict[str, Any]:
        prompt = self._build_prompt(iteration)
        raw = call_llm_api(prompt)
        if raw is None:
            # fallback: no change
            return {
                "action": "TUNE_HYPERPARAMETERS",
                "update": {"pysr_search_params": {"niterations": 100}},
                "reason": "LLM call failed; perform a small additional search."
            }

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract JSON substring if LLM accidentally wrapped it
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    obj = json.loads(raw[start:end + 1])
                except Exception:
                    obj = None
            else:
                obj = None

        if not isinstance(obj, dict):
            # Very conservative fallback
            return {
                "action": "TUNE_HYPERPARAMETERS",
                "update": {"pysr_search_params": {"niterations": 100}},
                "reason": "Failed to parse JSON; doing a simple hyperparameter extension."
            }

        # Ensure required keys
        obj.setdefault("action", "TUNE_HYPERPARAMETERS")
        obj.setdefault("update", {})
        obj.setdefault("reason", "No reason provided by agent.")

        return obj