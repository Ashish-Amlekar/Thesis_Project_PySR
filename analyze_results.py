import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

# Apply a modern theme
sns.set_theme(style="whitegrid", context="talk", palette="muted")

def plot_results(model, X, y, title_suffix=""):
    """
    Plot actual vs predicted values and residuals for a fitted model.
    """
    # Predict
    y_pred = model.predict(X)

    # -------- Graph 1: Actual vs Predicted --------
    plt.figure(figsize=(7, 7))
    plt.scatter(y, y_pred, alpha=0.7, edgecolor="k", s=80, cmap="viridis")
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y=x)")
    plt.xlabel("Actual", fontsize=14)
    plt.ylabel("Predicted", fontsize=14)
    plt.title(f"Actual vs Predicted {title_suffix}", fontsize=16, weight="bold")
    plt.legend(frameon=True, fontsize=12)
    plt.tight_layout()
    plt.show()

    # -------- Graph 2: Residuals histogram --------
    residuals = y - y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, bins=30, kde=True, color="skyblue", edgecolor="black")
    plt.xlabel("Residual (Actual - Predicted)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(f"Residual Distribution {title_suffix}", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()

    # -------- Graph 3: Residuals vs Actual --------
    plt.figure(figsize=(7, 5))
    plt.scatter(y, residuals, alpha=0.6, edgecolor="k", s=70, color="orange")
    plt.axhline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Actual", fontsize=14)
    plt.ylabel("Residual", fontsize=14)
    plt.title(f"Residuals vs Actual {title_suffix}", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()

def plot_error_analysis(model, X, y):
    """Extra error plots: Absolute Error and Relative Error."""
    y_pred = model.predict(X)
    abs_error = np.abs(y - y_pred)
    rel_error = abs_error / np.maximum(np.abs(y), 1e-8)

    # -------- Graph 4: Absolute Error --------
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=np.arange(len(abs_error)), y=abs_error, s=70, color="teal", edgecolor="k")
    plt.xlabel("Sample index", fontsize=14)
    plt.ylabel("Absolute Error", fontsize=14)
    plt.title("Absolute Error per Sample", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()

    # -------- Graph 5: Relative Error --------
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=np.arange(len(rel_error)), y=rel_error, s=70, color="purple", edgecolor="k")
    plt.xlabel("Sample index", fontsize=14)
    plt.ylabel("Relative Error", fontsize=14)
    plt.title("Relative Error per Sample", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load saved model and data from main.py
    run_folder = "C:/Users/ashis/PycharmProjects/LLM2/outputs/run_20250910_200302"

    model = joblib.load(os.path.join(run_folder, "best_model.pkl"))
    X = np.load(os.path.join(run_folder, "features.npy"))
    y = np.load(os.path.join(run_folder, "target.npy"))

    # Run plots
    plot_results(model, X, y, title_suffix="(Best Equation)")
    plot_error_analysis(model, X, y)