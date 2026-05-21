"""
demo.py - Demonstration of SimpleSLPClassifier and SimpleSLPRegressor

Runs experiments on three datasets:
  1. Diabetes      (sklearn) — Regression
  2. Breast Cancer (sklearn) — Binary classification
  3. Abalone       (UCI)     — Regression + Multi-class classification

Generates comparison plots saved as PNG files, and prints a summary table.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo

from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor

# ── Config ────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
IMG_DIR = "report_images"
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def prepare_data(X, y, test_size=0.2):
    """Split into train/test and standardise features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def plot_loss_curves(losses: dict, title: str, ax):
    """Plot one training loss curve per model."""
    for label, curve in losses.items():
        ax.plot(curve, label=label, linewidth=2)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()


def print_scores(scores: dict, metric: str):
    """Pretty-print a score comparison."""
    print(f"\n  {metric}")
    print("  " + "─" * 40)
    for model, score in scores.items():
        print(f"  {model:<30} {score:.4f}")


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── 1 · Diabetes — Regression ─────────────────────────────────────────────────

def run_diabetes():
    print_section("1 · Diabetes — Regression")

    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = prepare_data(diabetes.data, diabetes.target)

    print(f"  Samples : {diabetes.data.shape[0]}  |  Features : {diabetes.data.shape[1]}")
    print(f"  Target  : disease progression score")
    print(f"  Train / Test : {len(y_train)} / {len(y_test)}")

    # Our model
    our = SimpleSLPRegressor(
        hidden_layer_size=64, activation="relu", learning_rate=0.001,
        max_iter=500, random_state=RANDOM_STATE, n_iter_no_change=20, tol=1e-4,
    )
    our.fit(X_train, y_train)

    # sklearn baseline
    skl = MLPRegressor(
        hidden_layer_sizes=(64,), activation="relu", learning_rate_init=0.001,
        max_iter=500, random_state=RANDOM_STATE,
    )
    skl.fit(X_train, y_train)

    scores = {
        "SimpleSLPRegressor (ours)": our.score(X_test, y_test),
        "sklearn MLPRegressor"     : skl.score(X_test, y_test),
    }
    print_scores(scores, "R² score")
    print(f"  Early stopping : iteration {our.n_iter_} / 500")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_loss_curves(
        {"Ours": our.loss_curve_, "sklearn": skl.loss_curve_},
        "Diabetes — Training loss", axes[0]
    )
    y_pred = our.predict(X_test)
    lims = [y_test.min(), y_test.max()]
    axes[1].scatter(y_test, y_pred, alpha=0.5, s=25, label="Ours")
    axes[1].scatter(y_test, skl.predict(X_test), alpha=0.5, s=25, marker="x", label="sklearn")
    axes[1].plot(lims, lims, "k--", linewidth=1, label="Perfect")
    axes[1].set_xlabel("True value"); axes[1].set_ylabel("Predicted value")
    axes[1].set_title("Diabetes — Predicted vs Actual", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "diabetes.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")

    return scores


# ── 2 · Breast Cancer — Binary Classification ─────────────────────────────────

def run_breast_cancer():
    print_section("2 · Breast Cancer — Binary Classification")

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = prepare_data(cancer.data, cancer.target)

    print(f"  Samples : {cancer.data.shape[0]}  |  Features : {cancer.data.shape[1]}")
    print(f"  Classes : {list(cancer.target_names)}  {np.bincount(cancer.target).tolist()}")
    print(f"  Train / Test : {len(y_train)} / {len(y_test)}")

    # Our model
    our = SimpleSLPClassifier(
        hidden_layer_size=64, activation="relu", learning_rate=0.001,
        max_iter=500, random_state=RANDOM_STATE, n_iter_no_change=20, tol=1e-4,
    )
    our.fit(X_train, y_train)

    # sklearn baseline
    skl = MLPClassifier(
        hidden_layer_sizes=(64,), activation="relu", learning_rate_init=0.001,
        max_iter=500, random_state=RANDOM_STATE,
    )
    skl.fit(X_train, y_train)

    scores = {
        "SimpleSLPClassifier (ours)": our.score(X_test, y_test),
        "sklearn MLPClassifier"     : skl.score(X_test, y_test),
    }
    print_scores(scores, "Accuracy")
    print(f"  Early stopping : iteration {our.n_iter_} / 500")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_loss_curves(
        {"Ours": our.loss_curve_, "sklearn": skl.loss_curve_},
        "Breast Cancer — Training loss", axes[0]
    )
    cm = confusion_matrix(y_test, our.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=cancer.target_names).plot(
        ax=axes[1], colorbar=False, cmap="Blues"
    )
    axes[1].set_title("Breast Cancer — Confusion Matrix (ours)", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "breast_cancer.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")

    return scores


# ── 3 · Abalone — Regression + Multi-class ────────────────────────────────────

def run_abalone():
    print_section("3 · Abalone — Regression & Multi-class Classification")

    # Load
    print("  Fetching Abalone dataset from UCI...")
    abalone = fetch_ucirepo(id=1)
    X_ab = abalone.data.features.copy()
    y_rings = abalone.data.targets.values.ravel()

    # Encode Sex (M / F / I) as one-hot dummy columns
    X_ab = X_ab.join(X_ab["Sex"].str.get_dummies()).drop(columns="Sex")
    X_ab = X_ab.values.astype(float)

    print(f"  Samples : {X_ab.shape[0]}  |  Features (after encoding) : {X_ab.shape[1]}")
    print(f"  Rings   : min={y_rings.min()}, max={y_rings.max()}, mean={y_rings.mean():.1f}")

    scores = {}

    # ── 3a · Regression ───────────────────────────────────────────────────────
    print("\n  3a · Regression (predict ring count)")
    X_train, X_test, y_train, y_test = prepare_data(X_ab, y_rings)

    our_reg = SimpleSLPRegressor(
        hidden_layer_size=64, activation="relu", learning_rate=0.001,
        max_iter=600, random_state=RANDOM_STATE, n_iter_no_change=20, tol=1e-4,
    )
    our_reg.fit(X_train, y_train)

    skl_reg = MLPRegressor(
        hidden_layer_sizes=(64,), activation="relu", learning_rate_init=0.001,
        max_iter=600, random_state=RANDOM_STATE,
    )
    skl_reg.fit(X_train, y_train)

    reg_scores = {
        "SimpleSLPRegressor (ours)": our_reg.score(X_test, y_test),
        "sklearn MLPRegressor"     : skl_reg.score(X_test, y_test),
    }
    print_scores(reg_scores, "R² score")
    print(f"  Early stopping : iteration {our_reg.n_iter_} / 600")
    scores["Abalone regression"] = reg_scores

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_loss_curves(
        {"Ours": our_reg.loss_curve_, "sklearn": skl_reg.loss_curve_},
        "Abalone (regression) — Training loss", axes[0]
    )
    y_pred = our_reg.predict(X_test)
    lims = [y_test.min(), y_test.max()]
    axes[1].scatter(y_test, y_pred, alpha=0.4, s=15, label="Ours")
    axes[1].scatter(y_test, skl_reg.predict(X_test), alpha=0.4, s=15, marker="x", label="sklearn")
    axes[1].plot(lims, lims, "k--", linewidth=1, label="Perfect")
    axes[1].set_xlabel("True rings"); axes[1].set_ylabel("Predicted rings")
    axes[1].set_title("Abalone — Predicted vs Actual rings", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "abalone_regression.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")

    # ── 3b · Multi-class Classification ───────────────────────────────────────
    print("\n  3b · Multi-class classification (age groups)")
    group_names = ["Young (≤8)", "Adult (9–11)", "Old (≥12)"]
    y_groups = np.where(y_rings <= 8, 0, np.where(y_rings <= 11, 1, 2))
    counts = np.bincount(y_groups)
    for name, count in zip(group_names, counts):
        print(f"    {name:<18} : {count} samples ({100*count/len(y_groups):.1f}%)")

    X_train, X_test, y_train, y_test = prepare_data(X_ab, y_groups)

    our_clf = SimpleSLPClassifier(
        hidden_layer_size=64, activation="relu", learning_rate=0.001,
        max_iter=600, random_state=RANDOM_STATE, n_iter_no_change=20, tol=1e-4,
    )
    our_clf.fit(X_train, y_train)

    skl_clf = MLPClassifier(
        hidden_layer_sizes=(64,), activation="relu", learning_rate_init=0.001,
        max_iter=600, random_state=RANDOM_STATE,
    )
    skl_clf.fit(X_train, y_train)

    clf_scores = {
        "SimpleSLPClassifier (ours)": our_clf.score(X_test, y_test),
        "sklearn MLPClassifier"     : skl_clf.score(X_test, y_test),
    }
    print_scores(clf_scores, "Accuracy")
    print(f"  Early stopping : iteration {our_clf.n_iter_} / 600")
    scores["Abalone classification"] = clf_scores

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_loss_curves(
        {"Ours": our_clf.loss_curve_, "sklearn": skl_clf.loss_curve_},
        "Abalone (classification) — Training loss", axes[0]
    )
    cm = confusion_matrix(y_test, our_clf.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=group_names).plot(
        ax=axes[1], colorbar=False, cmap="Blues"
    )
    axes[1].set_title("Abalone — Confusion Matrix (ours)", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "abalone_classification.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")

    return scores


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_scores: dict):
    print_section("Summary — All experiments")
    header = f"  {'Dataset':<30} {'Metric':<12} {'Ours':>8} {'sklearn':>8}"
    print(header)
    print("  " + "─" * 62)

    rows = [
        ("Diabetes",               "R²",       all_scores["diabetes"]),
        ("Breast Cancer",          "Accuracy",  all_scores["breast_cancer"]),
        ("Abalone (regression)",   "R²",        all_scores["abalone"]["Abalone regression"]),
        ("Abalone (classif.)",     "Accuracy",  all_scores["abalone"]["Abalone classification"]),
    ]
    for dataset, metric, scores in rows:
        ours = list(scores.values())[0]
        skl  = list(scores.values())[1]
        print(f"  {dataset:<30} {metric:<12} {ours:>8.4f} {skl:>8.4f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_scores = {}
    all_scores["diabetes"]     = run_diabetes()
    all_scores["breast_cancer"] = run_breast_cancer()
    all_scores["abalone"]      = run_abalone()
    print_summary(all_scores)
    print(f"All plots saved in ./{IMG_DIR}/\n")
