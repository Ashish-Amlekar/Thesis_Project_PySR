# agent_actions.py
#Executes actions (REFINE_GRAMMAR, TUNE_HYPERPARAMETERS, etc.) on GrammarState + system_state.

from typing import Dict, Any, Tuple
from grammar_manager import GrammarState


def apply_agent_action(
    action_obj: Dict[str, Any],
    grammar: GrammarState,
    system_state: Dict[str, Any],
) -> Tuple[GrammarState, Dict[str, Any]]:
    """
    Applies the LLM agent's chosen action to the grammar and system state.

    action_obj format:
    {
      "action": "REFINE_GRAMMAR" | "TUNE_HYPERPARAMETERS" | "ADD_ENGINEERED_FEATURES" |
                "ENABLE_PI" | "DISABLE_PI" | "RESET_SEARCH" | "TERMINATE",
      "update": {...},
      "reason": "some explanation"
    }

    Returns (updated_grammar, updated_system_state).
    """
    action = action_obj.get("action", "").upper()
    update = action_obj.get("update", {}) or {}

    # Copy system_state to avoid mutation surprises
    system_state = dict(system_state)

    if action == "REFINE_GRAMMAR":
        grammar.update_from_agent(update)

    elif action == "TUNE_HYPERPARAMETERS":
        # e.g. {"pysr_search_params": {"niterations": 500, "population_size": 256}}
        grammar.update_from_agent({"pysr_search_params": update.get("pysr_search_params", {})})

    elif action == "ADD_ENGINEERED_FEATURES":
        # For now: just record a request flag in system_state.
        # A later component can read system_state["engineered_feature_requests"]
        # and actually add them in the preprocessing.
        requests = system_state.get("engineered_feature_requests", [])
        requests.append(update)
        system_state["engineered_feature_requests"] = requests

    elif action == "ENABLE_PI":
        system_state["use_buckingham_pi"] = True

    elif action == "DISABLE_PI":
        system_state["use_buckingham_pi"] = False

    elif action == "RESET_SEARCH":
        # Optionally reset iteration counters or PySR parameters
        system_state["reset_requested"] = True

    elif action == "TERMINATE":
        system_state["terminate"] = True

    else:
        # Unknown action – do nothing but log
        system_state.setdefault("warnings", []).append(f"Unknown action: {action}")

    return grammar, system_state