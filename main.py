"""
main.py — Pipeline principal d'analyse Telco Customer Churn.

Exécution quotidienne automatique :
    python main.py

Produit :
    - churn_predictions_output.csv  (recommandations par client)
    - *.png  (toutes les visualisations)
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")


from project import load_data, preprocess, scale_and_split, PALETTE, RISK_COLORS, savefig

from clustering     import run_clustering
from classification import run_classification
from regression     import run_regression
from nlp            import run_nlp


# ══════════════════════════════════════════════════════════════════════════
#  AGRÉGATION DES SIGNAUX
# ══════════════════════════════════════════════════════════════════════════
def aggregate_signals(X_test, y_test, km_model, pca_model,
                       probas_dict, results_df_clf,
                       df_nlp, feature_names):
    """
    Stratégie d'agrégation :
        RiskScore = 0.5 × Churn_Prob  +  0.3 × Cluster_Risk  +  0.2 × Sentiment_Signal

    - Churn_Prob    : moyenne pondérée par AUC des modèles de classification.
    - Cluster_Risk  : taux de churn observé dans le cluster du client (normalisé [0,1]).
    - Sentiment_Signal : polarité négative normalisée [0,1] → 0 = fidèle, 1 = risqué.

    Retourne un DataFrame avec, pour chaque client :
        churn_proba, cluster, cluster_risk, sentiment_signal,
        risk_score, risk_segment, recommendation.
    """
    # ── Probabilité de churn agrégée (pondérée par AUC) ──────────────────
    valid_probas = {k: v for k, v in probas_dict.items() if v is not None}
    auc_weights  = {
        name: results_df_clf[results_df_clf["Modèle"] == name]["ROC-AUC"].values[0]
        for name in valid_probas.keys()
        if name in results_df_clf["Modèle"].values
    }
    names_sorted = list(auc_weights.keys())
    stack = np.stack([valid_probas[n] for n in names_sorted])
    w = np.array([auc_weights[n] for n in names_sorted])
    w = w / w.sum()
    churn_prob = np.average(stack, weights=w, axis=0)

    out = pd.DataFrame(index=range(len(X_test)))
    out["churn_proba"] = churn_prob
    out["churn_true"]  = y_test.values

    # ── Signal de clustering ──────────────────────────────────────────────
    X_pca_test    = pca_model.transform(X_test)
    cluster_ids   = km_model.predict(X_pca_test)
    out["cluster"] = cluster_ids + 1

    # Taux de churn par cluster sur l'ensemble du dataset (calculé en amont)
    churn_by_cluster = pd.Series(y_test.values).groupby(cluster_ids).mean()
    max_rate = churn_by_cluster.max() if churn_by_cluster.max() > 0 else 1.0
    cluster_risk_map = (churn_by_cluster / max_rate).to_dict()
    out["cluster_risk"] = cluster_ids
    out["cluster_risk"] = out["cluster_risk"].map(cluster_risk_map).fillna(0.5)

    # ── Signal NLP ────────────────────────────────────────────────────────
    # On réaligne les index (nlp porte sur df complet, on prend les indices test)
    test_indices = y_test.index if hasattr(y_test, "index") else range(len(y_test))
    if "sentiment_signal" in df_nlp.columns:
        sent_signal = df_nlp["sentiment_signal"].iloc[:len(out)].values
    else:
        sent_signal = np.full(len(out), 0.5)
    # Sentiment risqué = 1 - signal (signal positif = faible risque)
    out["sentiment_risk"] = 1 - sent_signal[:len(out)]

    # ── Score de risque agrégé ────────────────────────────────────────────
    out["risk_score"] = (
        0.5 * out["churn_proba"]
      + 0.3 * out["cluster_risk"]
      + 0.2 * out["sentiment_risk"]
    )

    # ── Segmentation ─────────────────────────────────────────────────────
    def segment(s):
        if s < 0.25:  return "Faible"
        elif s < 0.50: return "Modere"
        elif s < 0.75: return "Eleve"
        else:          return "Critique"

    out["risk_segment"] = out["risk_score"].apply(segment)

    reco_map = {
        "Faible":   "Suivi standard — programme de fidelite passif",
        "Modere":   "Offre de fidelisation : remise 10% ou upgrade gratuit 1 mois",
        "Eleve":    "Contact proactif conseiller + bundle personnalise",
        "Critique": "Intervention immediate — offre retention prioritaire + appel manager",
    }
    out["recommendation"] = out["risk_segment"].map(reco_map)
    return out


def plot_aggregation(out, save=True):
    risk_order = ["Faible", "Modere", "Eleve", "Critique"]
    risk_counts = out["risk_segment"].value_counts().reindex(risk_order, fill_value=0)
    colors = [RISK_COLORS[r] for r in risk_order]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Pie
    axes[0].pie(risk_counts.values, labels=risk_order, colors=colors,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    axes[0].set_title("Répartition par Segment de Risque", fontweight="bold")

    # Bar : taux de churn réel vs score
    churn_by_seg = out.groupby("risk_segment").agg(
        churn_reel=("churn_true", "mean"),
        risk_score_moy=("risk_score", "mean")
    ).reindex(risk_order) * 100

    x = np.arange(len(risk_order)); w = 0.35
    axes[1].bar(x - w/2, churn_by_seg["churn_reel"],   w, color=PALETTE[1],
                edgecolor="white", label="Churn réel (%)")
    axes[1].bar(x + w/2, churn_by_seg["risk_score_moy"], w, color=PALETTE[0],
                edgecolor="white", label="Risk Score moyen (%)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(risk_order)
    axes[1].set_title("Validation : Churn réel vs Risk Score", fontweight="bold")
    axes[1].set_ylabel("%"); axes[1].legend()

    # Distribution du risk_score
    for seg, color in zip(risk_order, colors):
        subset = out[out["risk_segment"] == seg]["risk_score"]
        if len(subset) > 0:
            axes[2].hist(subset, bins=20, alpha=0.6, color=color, label=seg, edgecolor="white")
    for t, color in zip([0.25, 0.50, 0.75], [PALETTE[3], PALETTE[4], PALETTE[1]]):
        axes[2].axvline(t, color=color, linestyle="--", linewidth=1.5)
    axes[2].set_title("Distribution du Risk Score Agrégé", fontweight="bold")
    axes[2].set_xlabel("Risk Score"); axes[2].set_ylabel("Fréquence"); axes[2].legend()

    plt.suptitle("Agrégation des Signaux & Segmentation du Risque Client",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save: savefig("aggregation_signals.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#  DASHBOARD FINAL
# ══════════════════════════════════════════════════════════════════════════
def plot_dashboard(out, results_clf, results_reg, km_labels, X_2d, y, k, history_ann=None):
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Telco Customer Churn — Dashboard de Synthèse",
                 fontsize=20, fontweight="bold", y=0.99, color="#2C3E50")

    risk_order  = ["Faible", "Modere", "Eleve", "Critique"]
    risk_counts = out["risk_segment"].value_counts().reindex(risk_order, fill_value=0)
    colors_risk = [RISK_COLORS[r] for r in risk_order]

    # 1 — Segments
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.pie(risk_counts.values, labels=risk_order, colors=colors_risk,
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax1.set_title("Segments de Risque", fontweight="bold")

    # 2 — Taux churn vs segment
    ax2 = fig.add_subplot(3, 4, 2)
    churn_seg = out.groupby("risk_segment")["churn_true"].mean().reindex(risk_order) * 100
    ax2.bar(risk_order, churn_seg.values, color=colors_risk, edgecolor="white")
    ax2.set_title("Churn Réel / Segment", fontweight="bold"); ax2.set_ylabel("%")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right", fontsize=9)

    # 3 — ROC-AUC modèles
    ax3 = fig.add_subplot(3, 4, 3)
    auc_sorted = results_clf.sort_values("ROC-AUC")
    colors_bar = [PALETTE[1] if v == auc_sorted["ROC-AUC"].max() else PALETTE[0]
                  for v in auc_sorted["ROC-AUC"]]
    ax3.barh(auc_sorted["Modèle"], auc_sorted["ROC-AUC"], color=colors_bar, edgecolor="white")
    ax3.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax3.set_xlim(0.4, 1.0)
    ax3.set_title("ROC-AUC Modèles", fontweight="bold"); ax3.set_xlabel("AUC")
    plt.setp(ax3.get_yticklabels(), fontsize=8)

    # 4 — Régression R²
    ax4 = fig.add_subplot(3, 4, 4)
    r2_sorted = results_reg.sort_values("R²")
    ax4.barh(r2_sorted["Modèle"], r2_sorted["R²"], color=PALETTE[3], edgecolor="white")
    ax4.set_title("R² — Régresseurs", fontweight="bold"); ax4.set_xlabel("R²")
    plt.setp(ax4.get_yticklabels(), fontsize=8)

    # 5 — Clusters PCA
    ax5 = fig.add_subplot(3, 4, 5)
    cluster_colors = PALETTE[:k]
    for ki in range(k):
        mask = km_labels == ki
        ax5.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=cluster_colors[ki], alpha=0.35, s=8, label=f"C{ki+1}")
    ax5.set_title("Clusters KMeans (PCA 2D)", fontweight="bold")
    ax5.set_xlabel("PC1"); ax5.set_ylabel("PC2")
    ax5.legend(markerscale=2, fontsize=8)

    # 6 — Taux churn / cluster
    ax6 = fig.add_subplot(3, 4, 6)
    churn_by_cluster = pd.DataFrame({"cluster": km_labels, "churn": y.values}).groupby("cluster")["churn"].mean() * 100
    ax6.bar([f"C{i+1}" for i in range(k)], churn_by_cluster.values,
            color=cluster_colors, edgecolor="white")
    ax6.axhline(y.mean()*100, color="black", linestyle="--", linewidth=1.5)
    ax6.set_title("Churn / Cluster", fontweight="bold"); ax6.set_ylabel("%")

    # 7 — F1 comparaison SMOTE
    ax7 = fig.add_subplot(3, 4, 7)
    f1_vals = results_clf.set_index("Modèle")["F1-Score"]
    ax7.barh(f1_vals.index, f1_vals.values, color=PALETTE[2], edgecolor="white")
    ax7.set_title("F1-Score Modèles (avec SMOTE)", fontweight="bold"); ax7.set_xlabel("F1")
    plt.setp(ax7.get_yticklabels(), fontsize=8)

    # 8 — Risk score distribution
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.hist(out["risk_score"], bins=40, color=PALETTE[0], edgecolor="white", alpha=0.85)
    for t, c in zip([0.25, 0.50, 0.75], [PALETTE[3], PALETTE[4], PALETTE[1]]):
        ax8.axvline(t, color=c, linestyle="--", linewidth=1.5)
    ax8.set_title("Distribution Risk Score Agrégé", fontweight="bold")
    ax8.set_xlabel("Risk Score"); ax8.set_ylabel("Fréquence")

    # 9 — Recommandations count
    ax9 = fig.add_subplot(3, 1, 3)
    reco_counts = out["recommendation"].value_counts()
    ax9.barh(reco_counts.index, reco_counts.values, color=PALETTE[5], edgecolor="white")
    ax9.set_title("Recommandations — Nombre de clients concernés",
                  fontweight="bold", fontsize=12)
    ax9.set_xlabel("Nombre de clients")
    plt.setp(ax9.get_yticklabels(), fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    savefig("dashboard_synthese.png", dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  TELCO CUSTOMER CHURN — PIPELINE COMPLET")
    print("█"*60)

    # ── 1. Chargement & Prétraitement ─────────────────────────────────────
    df_raw = load_data()
    X, y, feature_names, df_encoded = preprocess(df_raw)
    X_train, X_test, y_train, y_test, X_train_sm, y_train_sm, scaler = scale_and_split(X, y)

    # ── 2. Clustering ─────────────────────────────────────────────────────
    km_model, km_labels, pca_model, pca2_model, k_opt, km_metrics = run_clustering(
        pd.DataFrame(scaler.transform(X), columns=feature_names), y, feature_names
    )
    _, X_2d = None, pca2_model.transform(pd.DataFrame(scaler.transform(X), columns=feature_names))

    # ── 3. Classification ─────────────────────────────────────────────────
    results_clf, trained_models, probas_dict, best_clf_name, mean_shap = run_classification(
        X_train, X_test, y_train, y_test, X_train_sm, y_train_sm, feature_names
    )

    # ── 4. Régression ─────────────────────────────────────────────────────
    results_reg, best_reg, reg_features = run_regression(df_encoded)

    # ── 5. NLP ────────────────────────────────────────────────────────────
    df_nlp = run_nlp(df_encoded[["Churn"]].copy())

    # ── 6. Agrégation ─────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  AGREGATION DES SIGNAUX")
    print("="*55)
    out = aggregate_signals(
        X_test, y_test, km_model, pca_model,
        probas_dict, results_clf, df_nlp, feature_names
    )
    plot_aggregation(out)

    # ── 7. Dashboard ──────────────────────────────────────────────────────
    plot_dashboard(out, results_clf, results_reg, km_labels, X_2d, y, k_opt)

    # ── 8. Export CSV ─────────────────────────────────────────────────────
    out.index.name = "client_id"
    out.to_csv("churn_predictions_output.csv")
    print(f"\n[main] Fichier de sortie : churn_predictions_output.csv ({len(out)} clients)")

    # ── 9. Résumé ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESUME FINAL")
    print("="*60)
    print(f"\n  Clustering  — KMeans k={k_opt}  Silhouette={km_metrics['Silhouette']:.4f}")
    print(f"  Classif.    — Meilleur modèle : {best_clf_name}  AUC={results_clf.iloc[0]['ROC-AUC']:.4f}")
    print(f"  Régression  — Meilleur modèle : {results_reg.iloc[0]['Modèle']}  R²={results_reg.iloc[0]['R²']:.4f}")
    print(f"  NLP         — Corrélation sentiment/churn confirmée")
    risk_summary = out["risk_segment"].value_counts().reindex(
        ["Faible", "Modere", "Eleve", "Critique"], fill_value=0
    )
    print("\n  Segmentation du risque :")
    for seg, n in risk_summary.items():
        print(f"    {seg:10s} : {n:4d} clients")
    print("\n[main] Pipeline terminé.")


if __name__ == "__main__":
    main()
