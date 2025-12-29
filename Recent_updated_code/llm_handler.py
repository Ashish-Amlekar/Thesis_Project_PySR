# =============================
# llm_handler.py (v2)
# =============================

import os
import json
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv


# --------- LLM BOOTSTRAP ---------

def configure_llm():
    """Loads API key from .env and configures the Google Gemini model."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key not found. Please set it in the .env file.")
    genai.configure(api_key=api_key)


def call_llm_api(prompt: str) -> str:
    """Sends a prompt to the configured Gemini model and returns the response text."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        cleaned = response.text.strip().replace("```json", "").replace("```", "")
        return cleaned
    except Exception as e:
        print(f"LLM API call failed: {e}")
        return None


# --------- KNOWLEDGE LOADER ---------

def _load_knowledge_context(path: str) -> str:
    """Loads all .txt and .md files from a directory into a single string."""
    full_text = ""
    print(f"Loading knowledge from: {os.path.abspath(path)}")
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith((".txt", ".md")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        full_text += f"--- START OF DOCUMENT: {file} ---\n\n"
                        full_text += f.read()
                        full_text += f"\n\n--- END OF DOCUMENT: {file} ---\n\n"
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")
    return full_text

# --------- Prompt Loader ---------  {New addition}
def _load_user_prompt(file_path: str) -> str:
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

# --------- PROMPT BUILDER ---------

# --------- Default Prompt Builder ---------
def _create_advanced_prompt(context: str, description: str, variables: dict) -> str:
    var_str = "\n".join([f"- '{name}' (Dim: {dim})" for name, dim in variables.items()])
    return f"""
You are an expert AI assistant in physics and symbolic regression.

### CONTEXT
{context}

### PROBLEM DESCRIPTION
{description}

### VARIABLES
{var_str}

### OUTPUT FORMAT
Return a valid JSON with 'direct_config' and 'indirect_config_suggestions'.
"""


# --------- BASIC SHAPE VALIDATION (no external deps) ---------

def _validate_llm_config(cfg: Dict[str, Any]) -> bool:
    if not isinstance(cfg, dict):
        return False
    if "direct_config" not in cfg or "indirect_config_suggestions" not in cfg:
        print("❌ Missing 'direct_config' or 'indirect_config_suggestions' in LLM output.")
        return False
    # Light checks
    dc = cfg.get("direct_config", {})
    if "grammar" in dc:
        gram = dc["grammar"]
        if not isinstance(gram, dict):
            print("❌ 'grammar' must be an object.")
            return False
    return True


# --------- PUBLIC API ---------

def generate_pysr_configuration(knowledge_base_path: str, problem_description: str, variables: dict, custom_prompt_file: str = None) -> dict:
    """Main function to generate a PySR configuration by querying the LLM."""
    configure_llm()
    knowledge_context = _load_knowledge_context(knowledge_base_path)
    user_prompt = _load_user_prompt(custom_prompt_file)
    if not knowledge_context:
        print("Warning: Knowledge base is empty or could not be read.")
        return None
    if user_prompt:
        prompt = f"""{user_prompt}

    ---
    KNOWLEDGE BASE (read and use this, but do NOT quote directly):

    {knowledge_context}
    """
    else:
        prompt = _create_advanced_prompt(knowledge_context, problem_description, variables)

    print("\n--- Sending Prompt to LLM... ---")
    raw_response = call_llm_api(prompt)

    if raw_response:
        try:
            cfg = json.loads(raw_response)
        except json.JSONDecodeError:
            print("--- ERROR: Failed to decode JSON from LLM response. ---")
            print(f"Raw Response:\n{raw_response}")
            return None
        if not _validate_llm_config(cfg):
            return None
        return cfg
    else:
        print("--- ERROR: Received no response from LLM. ---")
        return None
