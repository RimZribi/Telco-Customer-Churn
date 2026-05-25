"""
classification.py — Prediction du churn (classification binaire).

Modeles : Logistic Regression, KNN, SVM, Random Forest,
          Gradient Boosting, XGBoost, LightGBM (optionnel), MLP PyTorch.
Inclut  : GridSearchCV (RF + XGB), SMOTE, threshold tuning,
          courbes ROC, SHAP, matrice de confusion.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import shap

from project import PALETTE, savefig

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[classification] LightGBM absent — ignore.")


# ══════════════════════════════════════════════════════════════════════
# 1.  GRIDSEARCHCV
# ══════════════════════════════════════════════════════════════════════

def optimize_rf_clf(X_train, y_train):
    """GridSearchCV sur Random Forest Classifier."""
    param_grid = {
        "n_estimators":     [100, 200],
        "max_depth":        [8, 12, None],
        "min_samples_leaf": [3, 5],
    }
    gs = GridSearchCV(
        RandomForestClassifier(class_weight="balanced",
                                random_state=42, n_jobs=-1),
        param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    print(f"  [GridSearch RF]  params={gs.best_params_}  "
          f"AUC={gs.best_score_:.4f}")
    return gs.best_estimator_


def optimize_xgb_clf(X_train, y_train):
    """GridSearchCV sur XGBoost Classifier."""
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    param_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }
    gs = GridSearchCV(
        XGBClassifier(scale_pos_weight=scale_pos, eval_metric="logloss",
                        random_state=42, n_jobs=-1),
        param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    print(f"  [GridSearch XGB] params={gs.best_params_}  "
          f"AUC={gs.best_score_:.4f}")
    return gs.best_estimator_


# ══════════════════════════════════════════════════════════════════════
# 2.  MLP PyTorch
# ══════════════════════════════════════════════════════════════════════

class _MLP(nn.Module):
    """MLP binaire : BN -> 128 -> Dropout(0.3) -> 64 -> 32 -> 1 (logit)."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(X_train_sm, y_train_sm, X_test, y_test, epochs=60):
    """
    Entraine le MLP et retourne (metriques, modele, y_pred, y_proba).
    """
    X_tr = torch.tensor(X_train_sm.values, dtype=torch.float32)
    X_te = torch.tensor(X_test.values,     dtype=torch.float32)
    y_tr = torch.tensor(y_train_sm.values, dtype=torch.float32)

    pos_w = torch.tensor(
        [(y_train_sm == 0).sum() / max((y_train_sm == 1).sum(), 1)],
        dtype=torch.float32
    )
    loader = DataLoader(TensorDataset(X_tr, y_tr),
                         batch_size=256, shuffle=True)

    model = _MLP(input_dim=X_tr.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    tr_losses, val_losses = [], []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        tr_losses.append(ep_loss / len(X_tr))

        model.eval()
        with torch.no_grad():
            val_l = crit(
                model(X_te),
                torch.tensor(y_test.values, dtype=torch.float32)
            ).item()
        val_losses.append(val_l)

        if (ep + 1) % 20 == 0:
            print(f"    [MLP] Epoch {ep+1}/{epochs}  "
                  f"Train={tr_losses[-1]:.4f}  Val={val_losses[-1]:.4f}")

    # Courbe d'apprentissage
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tr_losses,  color=PALETTE[0], label="Train BCE")
    ax.plot(val_losses, color=PALETTE[1], linestyle="--", label="Val BCE")
    ax.set_title("MLP — Courbe d'apprentissage", fontweight="bold")
    ax.set_xlabel("Epoque"); ax.set_ylabel("Loss BCE"); ax.legend()
    plt.tight_layout(); savefig("mlp_clf_learning_curve.png"); plt.close()

    model.eval()
    with torch.no_grad():
        y_proba = torch.sigmoid(model(X_te)).numpy()
    y_pred = (y_proba >= 0.5).astype(int)

    return _mlp_metrics("MLP PyTorch", y_test.values, y_pred, y_proba), \
           model, y_pred, y_proba


def _mlp_metrics(name, y_true, y_pred, y_proba):
    return {
        "Modele":      name,
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Precision":   precision_score(y_true, y_pred,  zero_division=0),
        "Recall":      recall_score(y_true, y_pred,     zero_division=0),
        "F1-Score":    f1_score(y_true, y_pred,          zero_division=0),
        "ROC-AUC":     roc_auc_score(y_true, y_proba),
        "CV-F1 (mean)": float("nan"),
        "CV-F1 (std)":  float("nan"),
        "Threshold":   0.5,
    }


# ══════════════════════════════════════════════════════════════════════
# 3.  DEFINITION ET EVALUATION DES MODELES ML
# ══════════════════════════════════════════════════════════════════════

def get_base_models():
    """Retourne les modeles ML avec hyperparametres par defaut raisonnables."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.5,
            class_weight="balanced", random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7, n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, C=1.0,
            class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        ),
    }
    if HAS_LGBM:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1
        )
    return models


def evaluate_model(model, X_train_sm, y_train_sm,
                    X_test, y_test, name, threshold=0.5):
    """Entraine + evalue un modele sklearn."""
    model.fit(X_train_sm, y_train_sm)
    y_proba = model.predict_proba(X_test)[:, 1] \
              if hasattr(model, "predict_proba") else None
    y_pred = (y_proba >= threshold).astype(int) \
             if y_proba is not None else model.predict(X_test)

    cv = cross_val_score(model, X_train_sm, y_train_sm,
                          cv=5, scoring="f1", n_jobs=-1)
    return {
        "Modele":       name,
        "Accuracy":     accuracy_score(y_test, y_pred),
        "Precision":    precision_score(y_test, y_pred, zero_division=0),
        "Recall":       recall_score(y_test, y_pred,    zero_division=0),
        "F1-Score":     f1_score(y_test, y_pred,         zero_division=0),
        "ROC-AUC":      roc_auc_score(y_test, y_proba)
                        if y_proba is not None else float("nan"),
        "CV-F1 (mean)": cv.mean(),
        "CV-F1 (std)":  cv.std(),
        "Threshold":    threshold,
    }, model, y_pred, y_proba


# ══════════════════════════════════════════════════════════════════════
# 4.  THRESHOLD TUNING
# ══════════════════════════════════════════════════════════════════════

def tune_threshold(model, X_test, y_test, name):
    """Cherche le seuil maximisant le F1-score."""
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.10, 0.90, 0.01)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_test, yp, zero_division=0))
        precs.append(precision_score(y_test, yp, zero_division=0))
        recs.append(recall_score(y_test, yp, zero_division=0))

    best_t  = thresholds[int(np.argmax(f1s))]
    best_f1 = max(f1s)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1s,   color=PALETTE[0], lw=2, label="F1-Score")
    ax.plot(thresholds, precs, color=PALETTE[2], lw=2, ls="--", label="Precision")
    ax.plot(thresholds, recs,  color=PALETTE[1], lw=2, ls=":",  label="Recall")
    ax.axvline(best_t, color="black", ls="-.", lw=1.5,
               label=f"Seuil optimal={best_t:.2f} (F1={best_f1:.4f})")
    ax.set_title(f"Threshold Tuning — {name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Seuil"); ax.set_ylabel("Score"); ax.legend()
    plt.tight_layout()
    safe = name.replace(" ", "_")
    savefig(f"threshold_tuning_{safe}.png"); plt.close()

    print(f"[classification] Seuil optimal ({name}) : "
          f"{best_t:.2f}  F1={best_f1:.4f}")
    return float(best_t)


# ══════════════════════════════════════════════════════════════════════
# 5.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════

def plot_roc_curves(probas_dict, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, proba) in enumerate(probas_dict.items()):
        if proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatoire (0.500)")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Courbes ROC — Comparaison des Modeles",
                  fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); savefig("roc_curves.png"); plt.close()


def plot_metrics_comparison(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(results_df)); w = 0.15
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, results_df[m], w,
               label=m, color=PALETTE[i], edgecolor="white", lw=0.8)
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(results_df["Modele"], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
    ax.set_title("Comparaison des Modeles — Metriques",
                  fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout(); savefig("model_comparison.png"); plt.close()


def plot_confusion_matrix(y_test, y_pred, name):
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, sfx in zip(
        axes, [cm, cm_pct], ["d", ".1f"], ["effectifs", "%"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=["Non-Churn", "Churn"],
                    yticklabels=["Non-Churn", "Churn"], linewidths=0.5)
        ax.set_title(f"Confusion Matrix — {name} ({sfx})", fontweight="bold")
        ax.set_xlabel("Predit"); ax.set_ylabel("Reel")
    plt.tight_layout()
    safe = name.replace(" ", "_")
    savefig(f"confusion_matrix_{safe}.png"); plt.close()


def plot_shap(model, X_test, feature_names, model_name="XGBoost"):
    """SHAP TreeExplainer + summary + bar."""
    print(f"[classification] Calcul SHAP pour {model_name}...")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception as e:
        print(f"[classification] SHAP echec : {e}")
        return None, None

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                       max_display=15, show=False)
    plt.title(f"SHAP Summary — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe = model_name.replace(" ", "_")
    savefig(f"shap_summary_{safe}.png"); plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                       max_display=15, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {model_name}",
               fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig(f"shap_bar_{safe}.png"); plt.close()

    mean_shap = pd.DataFrame({
        "Feature":       feature_names,
        "SHAP_mean_abs": np.abs(shap_values).mean(axis=0),
    }).sort_values("SHAP_mean_abs", ascending=False).head(10)

    print("\n[classification] TOP 10 DRIVERS DU CHURN (SHAP)")
    for _, row in mean_shap.iterrows():
        print(f"  {row['Feature']:42s}  SHAP={row['SHAP_mean_abs']:.4f}")

    return shap_values, mean_shap


# ══════════════════════════════════════════════════════════════════════
# 6.  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def run_classification(X_train, X_test, y_train, y_test,
                        X_train_sm, y_train_sm, feature_names):
    """
    Pipeline complet de classification.
    Retourne : results_df, trained_models, probas_dict,
               best_model_name, mean_shap.
    """
    print("\n" + "="*55)
    print("  CLASSIFICATION — Prediction du Churn")
    print("="*55)

    # Optimisation RF et XGBoost
    print("\n[classification] Optimisation GridSearchCV...")
    rf_opt  = optimize_rf_clf(X_train_sm, y_train_sm)
    xgb_opt = optimize_xgb_clf(X_train_sm, y_train_sm)

    base_models = get_base_models()
    base_models["Random Forest"] = rf_opt
    base_models["XGBoost"]       = xgb_opt

    all_metrics, trained_models, probas_dict = [], {}, {}

    print("\n[classification] Evaluation des modeles ML :")
    for name, model in base_models.items():
        print(f"  Entrainement : {name}...")
        metrics, fitted, y_pred, y_proba = evaluate_model(
            model, X_train_sm, y_train_sm, X_test, y_test, name
        )
        all_metrics.append(metrics)
        trained_models[name] = (fitted, y_pred, y_proba)
        probas_dict[name]    = y_proba
        print(f"    F1={metrics['F1-Score']:.4f}  AUC={metrics['ROC-AUC']:.4f}")

    # MLP PyTorch
    print("\n[classification] Entrainement MLP PyTorch...")
    m_mlp, mlp_model, y_pred_mlp, y_proba_mlp = train_mlp(
        X_train_sm, y_train_sm, X_test, y_test, epochs=60
    )
    all_metrics.append(m_mlp)
    trained_models["MLP PyTorch"] = (mlp_model, y_pred_mlp, y_proba_mlp)
    probas_dict["MLP PyTorch"]    = y_proba_mlp

    results_df = pd.DataFrame(all_metrics).sort_values("ROC-AUC",
                                                         ascending=False)

    # Threshold tuning sur le meilleur modele sklearn (pas MLP)
    best_sklearn = results_df[results_df["Modele"] != "MLP PyTorch"].iloc[0]["Modele"]
    best_model_obj = trained_models[best_sklearn][0]
    print(f"\n[classification] Threshold tuning : {best_sklearn}")
    best_t = tune_threshold(best_model_obj, X_test, y_test, best_sklearn)

    # Reevaluation avec seuil optimal
    _, _, y_pred_opt, _ = evaluate_model(
        best_model_obj, X_train_sm, y_train_sm,
        X_test, y_test, best_sklearn, threshold=best_t
    )
    print(f"\n[classification] {best_sklearn} (seuil={best_t:.2f}) :")
    print(classification_report(y_test, y_pred_opt,
                                  target_names=["Non-Churn", "Churn"]))
    plot_confusion_matrix(y_test, y_pred_opt, best_sklearn)

    # Visualisations globales
    plot_roc_curves(probas_dict, y_test)
    plot_metrics_comparison(results_df)

    # SHAP sur XGBoost
    shap_target = "XGBoost" if "XGBoost" in trained_models else best_sklearn
    shap_vals, mean_shap = plot_shap(
        trained_models[shap_target][0], X_test,
        feature_names, model_name=shap_target
    )

    print("\n[classification] Tableau comparatif final :")
    print(results_df.round(4).to_string(index=False))

    best_name = results_df.iloc[0]["Modele"]
    return results_df, trained_models, probas_dict, best_name, mean_shap
