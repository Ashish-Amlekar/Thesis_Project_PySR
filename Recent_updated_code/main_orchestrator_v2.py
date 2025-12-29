# main_orchestrator_v2.py

import os
import json   # ✅ REQUIRED
from datetime import datetime
from typing import Dict, Any
import pandas as pd

from mcp_context_manager import MCPContextManager
from data_stats_generator import compute_data_stats
from grammar_manager import GrammarState
from agent_actions import apply_agent_action
from llm_agent import LLMAgent

# ✅ Make sure this imports YOUR pysr_runner, not PySR's package version
from pysr_runner import load_and_prepare_data, run_pysr_search

from config_translator import translate_llm_config_to_pysr_params
from llm_handler import generate_pysr_configuration
from pysr_runner import run_pysr_search_phase2, make_json_safe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_initial_llm_config() -> Dict[str, Any]:
    knowledge_base_path = os.path.join(BASE_DIR, "knowledge-base")

    cfg = generate_pysr_configuration(
        knowledge_base_path=knowledge_base_path,
        problem_description="Symbolic regression for general scientific data",
        variables={},
        custom_prompt_file=None,
    )
    if cfg is None:
        raise RuntimeError("Failed to get initial LLM configuration")

    return translate_llm_config_to_pysr_params(cfg)


def run_phase2_orchestration(max_iterations=5, data_path=None):

    print("\n=== PHASE 2 ORCHESTRATOR ===\n")

    ctx = MCPContextManager() #Initialize Context Manager + Agent
    agent = LLMAgent(ctx)

    system_state = {
        "use_buckingham_pi": False,   # PHASE2 does NOT use pi groups unless agent requests
        "start_time": datetime.now().isoformat(),
    }

    translated_params = _load_initial_llm_config()
    grammar_state = GrammarState.from_translated_params(translated_params)

    #ITERATION LOOP BEGINS (the core of Phase 2)
    for iter_idx in range(1, max_iterations + 1):
        print(f"\n[Phase2] === Iteration {iter_idx} ===")

        iteration = ctx.start_new_iteration()

        # --- Load dataset configuration ---
        with open("dataset_configs/drying_dataset.json", "r") as f:
            cfg = json.load(f)

        excel_path = data_path or cfg["data_file_path"]

        # --- Load raw dataset ONLY (no transformations here) ---
        X, y = load_and_prepare_data(
            excel_path=excel_path,
            feature_columns=cfg["raw_feature_columns"],
            target_column=cfg["target_variable"],
        )

        # Build translated params dynamically from grammar_state
        translated_params = {
            "unary_operators": grammar_state.unary_operators,
            "binary_operators": grammar_state.binary_operators,
            "constraints": grammar_state.constraints,
            "pysr_search_params": grammar_state.pysr_search_params,
        }

        # Run symbolic regression
        pysr_results = run_pysr_search_phase2(
            X=X,
            y=y,
            translated_params=translated_params,
        )

        # --- Save context ---
        ctx.save_pysr_results(iteration, make_json_safe(pysr_results))
        ctx.save_data_stats(iteration, make_json_safe(compute_data_stats(X, y)))
        ctx.save_grammar(iteration, make_json_safe(grammar_state.to_dict()))
        ctx.save_system_state(iteration, make_json_safe(system_state))

        # --- Agent decides next step ---
        action_obj = agent.decide_next_action(iteration)
        print("\n=== LLM AGENT DECISION ===")
        print(json.dumps(action_obj, indent=2))
        print("=== END AGENT DECISION ===\n")
        ctx.save_llm_decision(iteration, action_obj)

        grammar_state, system_state = apply_agent_action(
            action_obj, grammar_state, system_state
        )

        if system_state.get("terminate"):
            break

    print("\n[Phase2] Completed.\n")

if __name__ == "__main__" or __name__.endswith("main_orchestrator_v2"):
    run_phase2_orchestration(max_iterations=5, data_path=None)