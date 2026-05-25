"""
project.py — Fonctions partagées : chargement, prétraitement, helpers visuels.

Corrections v2 :
  - Encodage UTF-8 forcé partout (plus de problèmes d'accents sur Windows/Mac)
  - savefig() sauvegarde dans graphs/ automatiquement
  - Prétraitement simplifié et robuste
"""

import os
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend sans fenêtre → évite les conflits d'encodage
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[project] imbalanced-learn absent — SMOTE desactive.")

warnings.filterwarnings("ignore")

# ── Dossier graphiques ─────────────────────────────────────────────────────
GRAPHS_DIR = "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ── Palette (sans caractères spéciaux) ────────────────────────────────────
PALETTE = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
RISK_COLORS = {
    "Faible":   "#2ECC71",
    "Modere":   "#F39C12",
    "Eleve":    "#E67E22",
    "Critique": "#E74C3C",
}


# ── Sauvegarde figure ──────────────────────────────────────────────────────
def savefig(name, dpi=150):
    """
    Sauvegarde dans graphs/<name> en UTF-8.
    Utilise uniquement des caractères ASCII dans le nom de fichier.
    """
    path = os.path.join(GRAPHS_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[project] Figure -> {path}")


# ── Chargement ────────────────────────────────────────────────────────────
def load_data(filename="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Charge le CSV Telco (télécharge si absent).
    Lecture forcée en UTF-8 avec fallback latin-1.
    """
    mirrors = [
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "https://raw.githubusercontent.com/carlosfab/dsnp2/master/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]
    if not os.path.exists(filename):
        for url in mirrors:
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"[project] Dataset telecharge : {url}")
                break
            except Exception as e:
                print(f"[project] Echec miroir : {e}")

    # Lecture robuste : UTF-8 puis latin-1 en fallback
    try:
        df = pd.read_csv(filename, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filename, encoding="latin-1")

    print(f"[project] Dataset charge : {df.shape[0]} clients, {df.shape[1]} colonnes")
    return df


# ── Prétraitement ─────────────────────────────────────────────────────────
def preprocess(df, target="Churn"):
    """
    Nettoyage + encodage. Retourne X, y, feature_names, df_encoded.

    Etapes :
      1. Suppression customerID
      2. TotalCharges -> float, NaN -> médiane
      3. Cible Churn -> 0/1
      4. Variables binaires Yes/No -> 0/1
      5. One-Hot Encoding des catégorielles restantes
    """
    df = df.copy()

    # 1. Identifiant inutile
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    # 2. TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # 3. Cible
    df[target] = (df[target] == "Yes").astype(int)

    # 4. Genre
    df["gender"] = (df["gender"] == "Male").astype(int)

    # 5. Colonnes binaires Yes/No
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col] == "Yes").astype(int)

    # 6. One-Hot sur les restantes
    ohe_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # Convertir les colonnes bool créées par get_dummies en int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    y = df[target]
    X = df.drop(columns=[target])
    feature_names = X.columns.tolist()

    print(f"[project] Pretraitement OK — {X.shape[1]} features, "
          f"taux churn={y.mean()*100:.1f}%")
    return X, y, feature_names, df


# ── Split + Scaler + SMOTE ────────────────────────────────────────────────
def scale_and_split(X, y, test_size=0.2, random_state=42, apply_smote=False):
    """
    StandardScaler + train/test split stratifie + SMOTE optionnel.

    Returns : X_train, X_test, y_train, y_test,
              X_train_sm, y_train_sm, scaler
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    X_train_sm, y_train_sm = X_train.copy(), y_train.copy()

    if apply_smote and HAS_SMOTE:
        sm = SMOTE(random_state=random_state)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        before = dict(y_train.value_counts())
        after  = dict(pd.Series(y_train_sm).value_counts())
        print(f"[project] SMOTE : {before} -> {after}")
    elif apply_smote and not HAS_SMOTE:
        print("[project] SMOTE demande mais imbalanced-learn absent — ignore.")

    return X_train, X_test, y_train, y_test, X_train_sm, y_train_sm, scaler
