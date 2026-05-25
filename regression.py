"""
regression.py — Prediction de MonthlyCharges (regression).

Modeles ML  : Linear Regression, Ridge, Lasso, Random Forest, XGBoost
Modele DL   : MLP PyTorch (BatchNorm + Dropout + Adam)
Optimisation: GridSearchCV sur RF et XGBoost
Metriques   : RMSE, MAE, R2, CV-R2
Visualisations : comparaison, residus, feature importance, learning curve DL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from project import PALETTE, savefig


# ══════════════════════════════════════════════════════════════════════
# 1.  PREPARATION DES DONNEES
# ══════════════════════════════════════════════════════════════════════

def prepare_regression_data(df_encoded, target="MonthlyCharges",
                              test_size=0.2, random_state=42):
    """
    Prepare X / y pour la regression sur MonthlyCharges.
    Exclut TotalCharges (colineaire) et Churn.
    """
    exclude = [target, "TotalCharges", "Churn"]
    X = df_encoded.drop(columns=[c for c in exclude if c in df_encoded.columns])
    X = X.select_dtypes(include=[np.number])
    y = df_encoded[target]
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    print(f"[regression] Cible={target} | Features={len(feature_names)} | "
          f"Train={len(X_train)} | Test={len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_names, scaler


# ══════════════════════════════════════════════════════════════════════
# 2.  EVALUATION MODELES ML
# ══════════════════════════════════════════════════════════════════════

def evaluate_regressor(model, X_train, y_train, X_test, y_test, name):
    """Entraine, evalue (RMSE / MAE / R2 / CV-R2) et retourne les metriques."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train, y_train, cv=5,
                            scoring="r2", n_jobs=-1)
    print(f"  {name:30s}  RMSE={rmse:.3f}  MAE={mae:.3f}  "
          f"R2={r2:.4f}  CV-R2={cv.mean():.4f}+/-{cv.std():.4f}")
    return {
        "Modele": name, "RMSE": rmse, "MAE": mae,
        "R2": r2, "CV-R2 (mean)": cv.mean(), "CV-R2 (std)": cv.std()
    }, model, y_pred


# ══════════════════════════════════════════════════════════════════════
# 3.  GRIDSEARCHCV
# ══════════════════════════════════════════════════════════════════════

def optimize_rf(X_train, y_train):
    """GridSearchCV sur Random Forest Regressor."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [8, 12, None],
        "min_samples_leaf": [3, 5],
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid, scoring="r2", cv=3, n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    print(f"  [GridSearch RF]    meilleurs params : {gs.best_params_}  "
          f"CV-R2={gs.best_score_:.4f}")
    return gs.best_estimator_


def optimize_xgb(X_train, y_train):
    """GridSearchCV sur XGBoost Regressor."""
    param_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }
    gs = GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        param_grid, scoring="r2", cv=3, n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    print(f"  [GridSearch XGB]   meilleurs params : {gs.best_params_}  "
          f"CV-R2={gs.best_score_:.4f}")
    return gs.best_estimator_


# ══════════════════════════════════════════════════════════════════════
# 4.  MLP PyTorch  (Deep Learning)
# ══════════════════════════════════════════════════════════════════════

class _MLPRegr(nn.Module):
    """
    MLP pour la regression.
    Architecture : input -> BN -> 128 -> Dropout(0.3) -> 64 -> 32 -> 1
    """
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


def run_mlp_regressor(X_train, y_train, X_test, y_test,
                       epochs=80, lr=1e-3, batch_size=256):
    """
    Entraine le MLP PyTorch et retourne (metriques_dict, modele, y_pred).
    Affiche la courbe d'apprentissage.
    """
    # Conversion tenseurs
    X_tr = torch.tensor(X_train.values, dtype=torch.float32)
    X_te = torch.tensor(X_test.values,  dtype=torch.float32)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32)
    y_te_np = y_test.values

    loader = DataLoader(TensorDataset(X_tr, y_tr),
                         batch_size=batch_size, shuffle=True)

    model = _MLPRegr(input_dim=X_tr.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Scheduler : reduit le LR si la val loss stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=8
    )
    crit = nn.MSELoss()

    tr_losses, val_losses = [], []

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        tr_losses.append(ep_loss / len(X_tr))

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_te), torch.tensor(y_te_np, dtype=torch.float32)).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (ep + 1) % 20 == 0:
            print(f"    [MLP] Epoch {ep+1}/{epochs}  "
                  f"Train={tr_losses[-1]:.4f}  Val={val_losses[-1]:.4f}")

    # Courbe d'apprentissage
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tr_losses,  color=PALETTE[0], label="Train MSE")
    ax.plot(val_losses, color=PALETTE[1], linestyle="--", label="Val MSE")
    ax.set_title("MLP Regressor — Courbe d'apprentissage", fontweight="bold")
    ax.set_xlabel("Epoque"); ax.set_ylabel("MSE Loss"); ax.legend()
    plt.tight_layout()
    savefig("mlp_regressor_learning_curve.png")
    plt.close()

    # Predictions finales
    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).numpy()

    rmse = np.sqrt(mean_squared_error(y_te_np, y_pred))
    mae  = mean_absolute_error(y_te_np, y_pred)
    r2   = r2_score(y_te_np, y_pred)
    print(f"  {'MLP PyTorch':30s}  RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")

    return {"Modele": "MLP PyTorch", "RMSE": rmse, "MAE": mae,
            "R2": r2, "CV-R2 (mean)": float("nan"), "CV-R2 (std)": float("nan")
            }, model, y_pred


# ══════════════════════════════════════════════════════════════════════
# 5.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════

def plot_regression_comparison(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, color, asc in zip(
        axes,
        ["RMSE", "MAE", "R2"],
        [PALETTE[1], PALETTE[2], PALETTE[3]],
        [True, True, False]
    ):
        sd = results_df.sort_values(metric, ascending=asc)
        bars = ax.barh(sd["Modele"], sd[metric], color=color, edgecolor="white")
        ax.set_title(f"Comparaison — {metric}", fontweight="bold")
        ax.set_xlabel(metric)
        for bar, val in zip(bars, sd[metric]):
            ax.text(bar.get_width() * 1.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    plt.suptitle("Comparaison des Regresseurs — MonthlyCharges",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("regression_comparison.png")
    plt.close()


def plot_residuals(model, X_test, y_test, name, is_torch=False):
    """Graphique residus : Predits vs Reels, residus vs predits, distribution."""
    if is_torch:
        model.eval()
        with torch.no_grad():
            y_pred = model(
                torch.tensor(X_test.values, dtype=torch.float32)
            ).numpy()
        y_true = y_test.values
    else:
        y_pred = model.predict(X_test)
        y_true = y_test.values

    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Predits vs Reels
    lim = [min(y_true.min(), y_pred.min()) - 5,
            max(y_true.max(), y_pred.max()) + 5]
    axes[0].scatter(y_pred, y_true, alpha=0.3, s=15, color=PALETTE[0])
    axes[0].plot(lim, lim, color=PALETTE[1], lw=2, linestyle="--", label="y=x")
    axes[0].set_xlabel("Predits ($/mois)"); axes[0].set_ylabel("Reels ($/mois)")
    axes[0].set_title("Predits vs Reels", fontweight="bold"); axes[0].legend()

    # Residus vs Predits
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=15, color=PALETTE[2])
    axes[1].axhline(0, color=PALETTE[1], lw=1.5, linestyle="--")
    axes[1].set_xlabel("Valeurs predites"); axes[1].set_ylabel("Residus")
    axes[1].set_title("Residus vs Predits", fontweight="bold")

    # Distribution residus
    axes[2].hist(residuals, bins=40, color=PALETTE[3],
                  edgecolor="white", alpha=0.85)
    axes[2].axvline(0, color=PALETTE[1], lw=2, linestyle="--")
    axes[2].set_xlabel("Residu ($/mois)"); axes[2].set_ylabel("Frequence")
    axes[2].set_title("Distribution des Residus", fontweight="bold")

    plt.suptitle(f"Analyse des Residus — {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").replace("/", "")
    savefig(f"residuals_{safe_name}.png")
    plt.close()
    print(f"[regression] Residus {name} — moy={residuals.mean():.4f}  "
          f"std={residuals.std():.3f}")


def plot_feature_importance_reg(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        suffix = "(Importance)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        suffix = "(|Coefficients|)"
    else:
        return  # MLP : pas de feature importance directe

    indices = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in indices],
             importances[indices],
             color=PALETTE[4], edgecolor="white")
    ax.set_title(f"Top 15 Features — {model_name} {suffix}",
                  fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    safe = model_name.replace(" ", "_").replace("/", "")
    savefig(f"feature_importance_reg_{safe}.png")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 6.  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def run_regression(df_encoded):
    """
    Pipeline complet de regression.
    Retourne : results_df, best_model, feature_names.
    """
    print("\n" + "="*55)
    print("  REGRESSION — Prediction de MonthlyCharges")
    print("="*55)

    X_train, X_test, y_train, y_test, feature_names, scaler = \
        prepare_regression_data(df_encoded)

    print("\n[regression] Optimisation des hyperparametres (GridSearchCV)...")
    rf_opt  = optimize_rf(X_train, y_train)
    xgb_opt = optimize_xgb(X_train, y_train)

    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(alpha=1.0),
        "Lasso":             Lasso(alpha=0.05, max_iter=3000),
        "Random Forest":     rf_opt,
        "XGBoost":           xgb_opt,
    }

    print("\n[regression] Evaluation des modeles ML :")
    all_metrics, trained = [], {}
    for name, model in regressors.items():
        m, fitted, y_pred = evaluate_regressor(
            model, X_train, y_train, X_test, y_test, name
        )
        all_metrics.append(m)
        trained[name] = (fitted, y_pred)

    # MLP PyTorch
    print("\n[regression] Entrainement MLP PyTorch...")
    m_mlp, mlp_model, y_pred_mlp = run_mlp_regressor(
        X_train, y_train, X_test, y_test, epochs=80
    )
    all_metrics.append(m_mlp)
    trained["MLP PyTorch"] = (mlp_model, y_pred_mlp)

    results_df = pd.DataFrame(all_metrics).sort_values("R2", ascending=False)
    print("\n[regression] Tableau comparatif :")
    print(results_df.round(4).to_string(index=False))

    plot_regression_comparison(results_df)

    # Analyse approfondie du meilleur modele
    best_name  = results_df.iloc[0]["Modele"]
    is_torch   = best_name == "MLP PyTorch"
    best_model = trained[best_name][0]
    print(f"\n[regression] Meilleur modele : {best_name}")
    plot_residuals(best_model, X_test, y_test, best_name, is_torch=is_torch)
    plot_feature_importance_reg(best_model, feature_names, best_name)

    # Toujours afficher residus pour XGBoost (interpretable)
    if best_name != "XGBoost":
        plot_residuals(trained["XGBoost"][0], X_test, y_test, "XGBoost")
        plot_feature_importance_reg(trained["XGBoost"][0], feature_names, "XGBoost")

    return results_df, best_model, feature_names
