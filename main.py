# =============================
# main.py (v2)
# =============================

import json
import os
import pandas as pd
import numpy as np
import joblib
from typing import List, Tuple
from llm_handler import generate_pysr_configuration
from config_translator import translate_llm_config_to_pysr_params, PhysicsInformedPenalty
from pysr_runner import load_and_prepare_data, run_pysr_search, _r2, prettify_equation
from utils.knowledge_converter import build_markdown_knowledge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

CONFIG_FILE = "dataset_configs/drying_dataset.json"

if __name__ == "__main__":
    # ---------------- Project configuration ----------------
    with open(CONFIG_FILE, "r") as f:
        cfg = json.load(f)


    DATA_FILE_PATH = cfg["data_file_path"]
    RAW_FEATURE_COLUMNS = cfg["raw_feature_columns"]
    TARGET_VARIABLE_NAME = cfg["target_variable"]
    PROBLEM = cfg["problem_description"]
    INPUT_VARIABLES = cfg["variables"]
    ALPHA_PHYSICS = cfg.get("alpha_physics_penalty",0.5)
    CUSTOM_PROMPT_FILE = cfg.get("custom_prompt_file", None)

    # ---------------- Phase 1: Ask LLM for configuration ----------------
    print("--- Phase 1: Generating Physics-Informed Configuration from LLM ---")

    RAW_KNOWLEDGE_PATH = "raw_knowledge/"  # users drop original files here
    KNOWLEDGE_BASE_PATH = "knowledge-base/"  # normalized .md files live here

    # Convert all raw files → Markdown, preserving folder structure
    build_markdown_knowledge(RAW_KNOWLEDGE_PATH,KNOWLEDGE_BASE_PATH, clear_output=False)

    llm_json_config = generate_pysr_configuration(
        knowledge_base_path=KNOWLEDGE_BASE_PATH,
        problem_description=PROBLEM,
        variables=INPUT_VARIABLES,
        custom_prompt_file=CUSTOM_PROMPT_FILE
    )

    if not llm_json_config:
        print("\n\n--- FAILED to get a valid configuration from the LLM. Aborting. ---")
        raise SystemExit(1)

    print("\n\n SUCCESSFULLY PARSED CONFIGURATION FROM LLM ")
    print(json.dumps(llm_json_config, indent=2))

    try:
        # ---------------- Phase 2: Load data ----------------
        features, target = load_and_prepare_data(
            excel_path=DATA_FILE_PATH,
            feature_columns=RAW_FEATURE_COLUMNS,
            target_column=TARGET_VARIABLE_NAME,
        )

        # Derive Fs if not present
        if "Fl" in features.columns and "Fs" not in features.columns:
            features["Fs"] = 1.0 - features["Fl"]

        FINAL_FEATURE_COLUMNS: List[str] = list(features.columns)

        # ---------------- Phase 3: Translate LLM config ----------------
        print("\n--- Translating LLM guidance into executable PySR parameters ---")
        translated = translate_llm_config_to_pysr_params(llm_json=llm_json_config)

        # ---------------- Phase 4: Run PySR with dynamic hyperparams ---
        model, meta = run_pysr_search(
            X=features,
            y=target,
            translated_params=translated,
        )

        # ---------------- Phase 5: Rerank with physics penalty ---------
        loss_suggestions = (
            llm_json_config.get("indirect_config_suggestions", {}).get("custom_loss_suggestions")
        )
        if loss_suggestions:
            print("\n--- Applying Physics-Informed Penalty for reranking ---")
            penalty_fn = PhysicsInformedPenalty(
                X=features.to_numpy(),
                feature_names=FINAL_FEATURE_COLUMNS,
                suggestions=loss_suggestions,
            )

            equations = model.equations_.copy()
            results: List[Tuple[str, float, float, float, float,float,float]] = []
            # (equation_str, julia_loss, physics_penalty, total_score, train_r2, mse,mae)

            #alpha = 0.05  # weight of physics penalty in reranking

            for _, row in equations.iterrows():
                try:
                    func = row["lambda_format"]  # compiled callable from PySR
                    y_hat = func(features.to_numpy())
                    # compute metrics
                    physics_pen = penalty_fn(target.to_numpy(), y_hat)
                    julia_loss = float(row["loss"])  # Julia elementwise loss aggregated
                    train_r2 = _r2(target.to_numpy(), y_hat)
                    mae = mean_absolute_error(target.to_numpy(), y_hat)
                    mse = mean_squared_error(target.to_numpy(), y_hat)
                    total = julia_loss + ALPHA_PHYSICS * physics_pen
                    results.append((str(row["equation"]), julia_loss, physics_pen, total, train_r2,mae,mse))
                except Exception as e:
                    print(f"Skipping equation due to error: {e}")

            if results:
                # Explicitly filter out constant equations (no feature names inside)
                #def is_nonconstant(eq: str) -> bool:
                #    return any(var in eq for var in FINAL_FEATURE_COLUMNS)

                # prefer non-trivial equations
                #non_constant = [r for r in results if is_nonconstant(r[0])]
                #candidates = non_constant if non_constant else results
                # sort by total score
                #results_sorted = sorted(candidates, key=lambda t: t[3])
                results_sorted = sorted(results, key=lambda t: t[3])
                #pick best
                best_eq = results_sorted[0]

                print("\nBest physics-informed equation (after reranking):")
                pretty_equation = prettify_equation(best_eq[0], precision=2)
                print(f"Equation: {pretty_equation}")
                print(f"Julia Loss: {best_eq[1]:.6g}")
                print(f"Physics Penalty (scaled): {best_eq[2]:.6g}")
                print(f"Total Score: {best_eq[3]:.6g}")
                print(f"Train R2: {best_eq[4]:.4f}")
                print(f"MAE: {best_eq[5]:.6g}")
                print(f"MSE: {best_eq[6]:.6g}")

        # Also report R2 of PySR's own best equation (already printed in meta)
        print(f"\nPySR (model_selection='best') training R2: {meta['train_r2_best']:.4f}")


    except FileNotFoundError:
        print(f"\n\n❌--- ERROR: Data file not found at '{DATA_FILE_PATH}'. ---❌")

    except Exception as e:
        print(f"\n\n❌--- AN UNEXPECTED ERROR OCCURRED ---❌\n{e}")

    # Create a timestamped run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"C:/Users/ashis/PycharmProjects/MCPLLM/outputs/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    if 'model' in locals():
        # Save model and data inside this run's folder
        joblib.dump(model, os.path.join(run_dir, "best_model.pkl"))
        np.save(os.path.join(run_dir, "features.npy"), features.to_numpy())
        np.save(os.path.join(run_dir, "target.npy"), target.to_numpy())
        print(f"\n✅ Run results saved to: {run_dir}")
    else:
        print("\n⚠️ Skipping save step because model was not created.")
