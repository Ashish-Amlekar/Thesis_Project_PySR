# src/knowledge_extractor.py

import os
import json
from pathlib import Path
from typing import Dict,List
from llm_handler import call_llm_api  # 👈 reuse your existing Gemini call

def load_markdown_files(md_dir: str) -> List[dict]:
    """
    Load all Markdown files individually instead of merging them.
    Returns a list of dicts: [{"path": file_path, "content": text}, ...]
    """
    files_data = []
    for root, _, files in os.walk(md_dir):
        for f in files:
            if f.endswith(".md"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                files_data.append({"path": path, "content": content})
    return files_data


def extract_physics_hints_from_knowledge(
    md_dir: str,
    extraction_prompt_file: str,
    save_path: str = "configs/auto_physics_hints.json"
) -> Dict:
    """
    Uses the Gemini LLM to read all markdown knowledge and extract physics hints,
    while also including the source file path in the result.
    """

    if not Path(extraction_prompt_file).exists():
        raise FileNotFoundError(
            f"⚠️ Extraction prompt file not found: {extraction_prompt_file}. "
            f"Please create one (e.g., 'user_prompts/physics_extract_prompt.txt')."
        )

    md_files = load_markdown_files(md_dir)
    user_prompt = Path(extraction_prompt_file).read_text(encoding="utf-8")

    # 🧠 Build a structured multi-file prompt
    prompt_parts = [user_prompt, "\n--- KNOWLEDGE BASE START ---\n"]
    for i, fdata in enumerate(md_files, 1):
        prompt_parts.append(
            f"\n📄 FILE {i}: {fdata['path']}\n"
            f"------------------------------------\n"
            f"{fdata['content']}\n"
        )
    prompt_parts.append("\n--- KNOWLEDGE BASE END ---\n")
    prompt_parts.append(
        "⚠️ IMPORTANT: Include the `source` field for each candidate_template with the exact file path where the information came from."
    )

    final_prompt = "\n".join(prompt_parts)

    print("\n📘 [Knowledge Extractor] Sending structured knowledge base to Gemini for physics hint extraction...")
    llm_response = call_llm_api(final_prompt)

    if not llm_response:
        raise RuntimeError("⚠️ LLM did not return a valid response.")

    # Parse the JSON output from LLM
    try:
        hints = json.loads(llm_response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"❌ LLM output is not valid JSON. Check formatting. Raw output:\n{llm_response}"
        ) from e

    # ✅ Post-process: sanity-check candidate_templates and source field
    if "candidate_templates" in hints:
        for t in hints["candidate_templates"]:
            if "source" not in t or not t["source"].strip():
                t["source"] = "[UNKNOWN FILE]"

    # Save extracted hints
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_text(json.dumps(hints, indent=2), encoding="utf-8")
    print(f"✅ Auto physics hints saved to: {save_path}")

    return hints


if __name__ == "__main__":
    extract_physics_hints_from_knowledge(
        md_dir="knowledge-base",
        extraction_prompt_file="user_prompts/physics_extract_prompt.txt"
    )