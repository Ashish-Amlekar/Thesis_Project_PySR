# mcp_context_manager.py
#This manages iteration folders + reading/writing JSON.

import os
import json
from typing import Any, Dict, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_CONTEXT_ROOT = os.path.join(BASE_DIR, "llm_context")


class MCPContextManager:
    """
    Handles creation of iteration folders and saving/loading of
    PySR results, grammar state, data stats, decisions, and system state.
    """

    def __init__(self, context_root: str = LLM_CONTEXT_ROOT):
        self.context_root = context_root
        os.makedirs(self.context_root, exist_ok=True)

    # ---------- Iteration utilities ----------

    def _format_iter_folder(self, iteration: int) -> str:
        return f"iter_{iteration:03d}"

    def get_latest_iteration(self) -> Optional[int]:
        """
        Returns the latest iteration number found in llm_context,
        or None if no iterations yet.
        """
        if not os.path.exists(self.context_root):
            return None

        iters = []
        for name in os.listdir(self.context_root):
            if name.startswith("iter_"):
                try:
                    n = int(name.split("_")[1])
                    iters.append(n)
                except Exception:
                    continue

        if not iters:
            return None
        return max(iters)

    def start_new_iteration(self) -> int:
        """
        Creates a new iteration folder after the current latest.
        Returns the new iteration number.
        """
        latest = self.get_latest_iteration()
        new_iter = 1 if latest is None else latest + 1
        folder = os.path.join(self.context_root, self._format_iter_folder(new_iter))
        os.makedirs(folder, exist_ok=True)
        return new_iter

    def get_iteration_folder(self, iteration: int) -> str:
        folder = os.path.join(self.context_root, self._format_iter_folder(iteration))
        os.makedirs(folder, exist_ok=True)
        return folder

    # ---------- JSON helpers ----------

    @staticmethod
    def _save_json(path: str, obj: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def _load_json(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------- Public save/load APIs ----------

    def save_pysr_results(self, iteration: int, results: Dict[str, Any]) -> str:
        folder = self.get_iteration_folder(iteration)
        path = os.path.join(folder, "pysr_results.json")
        self._save_json(path, results)
        return path

    def load_pysr_results(self, iteration: int) -> Optional[Dict[str, Any]]:
        folder = self.get_iteration_folder(iteration)
        path = os.path.join(folder, "pysr_results.json")
        return self._load_json(path)

    def save_data_stats(self, iteration: int, stats: Dict[str, Any]) -> str:
        folder = self.get_iteration_folder(iteration)
        path = os.path.join(folder, "data_stats.json")
        self._save_json(path, stats)
        return path

    def load_data_stats(self, iteration: int) -> Optional[Dict[str, Any]]:
        folder = self.get_iteration_folder(iteration)
        path = os.path.join(folder, "data_stats.json")
        return self._load_json(path)

    def save_grammar(self, iteration: int, grammar: Dict[str, Any]) -> str:
        folder = self.get_iteration_folder(iteration)
        path = os.path.join(folder, "grammar.json")
        self._save_json(path, grammar)
        return path

    def load_grammar(self, iteration: int) -> Optional[Dict[str, Any]]:
        folder = self.get_iteration_folder(iteration)
        path = os.path.join(folder, "grammar.json")
        return self._load_json(path)

    def save_system_state(self, iteration: int, state: Dict[str, Any]) -> str:
        folder = self.get_iteration_folder(iteration)
        path = os.path.join(folder, "system_state.json")
        self._save_json(path, state)
        return path

    def load_system_state(self, iteration: int) -> Optional[Dict[str, Any]]:
        folder = self.get_iteration_folder(iteration)
        path = os.path.join(folder, "system_state.json")
        return self._load_json(path)

    def save_llm_decision(self, iteration: int, decision: Dict[str, Any]) -> str:
        folder = self.get_iteration_folder(iteration)
        path = os.path.join(folder, "llm_decision.json")
        self._save_json(path, decision)
        return path

    def load_llm_decision(self, iteration: int) -> Optional[Dict[str, Any]]:
        folder = self.get_iteration_folder(iteration)
        path = os.path.join(folder, "llm_decision.json")
        return self._load_json(path)