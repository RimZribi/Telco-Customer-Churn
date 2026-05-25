"""
clustering.py — Segmentation non supervisee des clients.

Methodes : KMeans (+ Elbow + Silhouette), Agglomerative + Dendrogramme, DBSCAN.
Visualisations : PCA 2D, t-SNE 2D, taux de churn par cluster.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

from project import PALETTE, savefig


# ══════════════════════════════════════════════════════════════════════
# 1.  REDUCTION DE DIMENSION
# ══════════════════════════════════════════════════════════════════════

def fit_pca(X_scaled, n_components=0.95, random_state=42):
    """PCA conservant n_components% de variance (defaut 95%)."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    print(f"[clustering] PCA : {X_pca.shape[1]} composantes "
          f"({n_components*100:.0f}% variance)")
    return pca, X_pca


def fit_pca_2d(X_scaled, random_state=42):
    """PCA 2D pour visualisation."""
    pca2 = PCA(n_components=2, random_state=random_state)
    return pca2, pca2.fit_transform(X_scaled)


def fit_tsne(X_pca, random_state=42):
    """
    t-SNE 2D sur les composantes PCA (plus rapide que sur X brut).
    Utilise n=50 composantes max.
    """
    n_pre = min(50, X_pca.shape[1])
    X_pre = X_pca[:, :n_pre]
    tsne  = TSNE(n_components=2, perplexity=40, n_iter=1000,
                  random_state=random_state, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_pre)
    print("[clustering] t-SNE calcule.")
    return X_tsne


# ══════════════════════════════════════════════════════════════════════
# 2.  SELECTION DU NOMBRE DE CLUSTERS
# ══════════════════════════════════════════════════════════════════════

def select_k(X_pca, k_range=range(2, 10), random_state=42):
    """Elbow + Silhouette pour choisir k optimal."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(
            silhouette_score(X_pca, labels, sample_size=2000,
                              random_state=random_state)
        )

    best_k = list(k_range)[int(np.argmax(silhouettes))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(list(k_range), inertias, color=PALETTE[0],
                  lw=2, marker="o", ms=6)
    axes[0].set_title("Methode du Coude (Elbow)", fontweight="bold")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertie")

    axes[1].plot(list(k_range), silhouettes, color=PALETTE[2],
                  lw=2, marker="s", ms=6)
    axes[1].axvline(best_k, color=PALETTE[1], ls="--", lw=1.8,
                     label=f"k={best_k} optimal")
    axes[1].set_title("Score de Silhouette", fontweight="bold")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
    axes[1].legend()

    plt.suptitle("Selection du nombre de clusters — KMeans",
                  fontsize=14, fontweight="bold")
    plt.tight_layout(); savefig("kmeans_selection.png"); plt.close()
    print(f"[clustering] k optimal={best_k}  silhouette={max(silhouettes):.4f}")
    return best_k


# ══════════════════════════════════════════════════════════════════════
# 3.  KMEANS
# ══════════════════════════════════════════════════════════════════════

def run_kmeans(X_pca, k, random_state=42):
    """KMeans + metriques."""
    km = KMeans(n_clusters=k, random_state=random_state,
                 n_init=20, max_iter=500)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    db  = davies_bouldin_score(X_pca, labels)
    ch  = calinski_harabasz_score(X_pca, labels)
    print(f"[clustering] KMeans(k={k}) — Silhouette={sil:.4f}  "
          f"Davies-Bouldin={db:.4f}  Calinski-Harabasz={ch:.1f}")
    return km, labels, {"Silhouette": sil, "Davies-Bouldin": db,
                         "Calinski-Harabasz": ch}


def plot_clusters_2d(X_2d, labels, y, pca2, k,
                      title_suffix="PCA 2D", file_suffix="pca"):
    """Visualisation 2D (PCA ou t-SNE) + taux de churn par cluster."""
    colors = PALETTE[:k]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ki in range(k):
        mask = labels == ki
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                         c=colors[ki], alpha=0.45, s=12,
                         label=f"Cluster {ki+1}")
    axes[0].set_title(f"KMeans (k={k}) — {title_suffix}", fontweight="bold")
    if hasattr(pca2, "explained_variance_ratio_"):
        axes[0].set_xlabel(
            f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} var.)"
        )
        axes[0].set_ylabel(
            f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} var.)"
        )
    axes[0].legend(markerscale=2)

    churn_rate = (pd.DataFrame({"cluster": labels, "churn": y.values})
                  .groupby("cluster")["churn"].mean() * 100)
    bars = axes[1].bar(
        [f"Cluster {i+1}" for i in range(k)],
        churn_rate.values, color=colors, edgecolor="white", lw=1.5
    )
    axes[1].axhline(y.mean() * 100, color="black", ls="--", lw=1.5,
                     label=f"Taux global ({y.mean()*100:.1f}%)")
    axes[1].set_title("Taux de Churn par Cluster", fontweight="bold")
    axes[1].set_ylabel("Taux de churn (%)"); axes[1].legend()
    for bar, val in zip(bars, churn_rate.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.5,
                      f"{val:.1f}%", ha="center",
                      fontweight="bold", fontsize=11)

    plt.suptitle("Segmentation KMeans des Clients",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig(f"kmeans_clusters_{file_suffix}.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════
# 4.  AGGLOMERATIF + DENDROGRAMME
# ══════════════════════════════════════════════════════════════════════

def run_agglomerative(X_pca, k):
    """Clustering hierachique Ward + dendrogramme sur sous-echantillon."""
    np.random.seed(42)
    idx    = np.random.choice(len(X_pca), min(300, len(X_pca)), replace=False)
    Z      = linkage(X_pca[idx], method="ward")

    fig, ax = plt.subplots(figsize=(13, 4))
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=5,
               color_threshold=0.7 * max(Z[:, 2]),
               above_threshold_color="#95A5A6")
    ax.set_title("Dendrogramme — Clustering Hierarchique (Ward)",
                  fontweight="bold")
    ax.set_xlabel("Clients (echantillon)"); ax.set_ylabel("Distance de Ward")
    plt.tight_layout(); savefig("dendrogram.png"); plt.close()

    agg    = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(X_pca)
    sil    = silhouette_score(X_pca, labels, sample_size=2000, random_state=42)
    print(f"[clustering] Agglomerative(k={k}) — Silhouette={sil:.4f}")
    return labels, {"Silhouette": sil}


# ══════════════════════════════════════════════════════════════════════
# 5.  DBSCAN
# ══════════════════════════════════════════════════════════════════════

def run_dbscan(X_pca, eps=1.5, min_samples=20):
    """DBSCAN + k-distance graph pour visualiser eps."""
    nbrs  = NearestNeighbors(n_neighbors=5).fit(X_pca)
    dists, _ = nbrs.kneighbors(X_pca)
    dists_sorted = np.sort(dists[:, 4])[::-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dists_sorted, color=PALETTE[0], lw=2)
    ax.set_title("k-Distance Graph — Estimation epsilon (DBSCAN)",
                  fontweight="bold")
    ax.set_xlabel("Points tries"); ax.set_ylabel("5e plus proche voisin")
    plt.tight_layout(); savefig("dbscan_epsilon.png"); plt.close()

    db     = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_pca)
    n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"[clustering] DBSCAN — {n_cl} clusters, "
          f"{n_noise} points bruit ({n_noise/len(labels):.1%})")

    sil = None
    if n_cl > 1:
        mask = labels != -1
        sil  = silhouette_score(X_pca[mask], labels[mask],
                                  sample_size=2000, random_state=42)
        print(f"[clustering] DBSCAN Silhouette (sans bruit) = {sil:.4f}")
    return labels, n_cl, {"Silhouette": sil}


# ══════════════════════════════════════════════════════════════════════
# 6.  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def run_clustering(X_scaled, y, feature_names=None):
    """
    Pipeline complet de clustering.
    Retourne : km_model, km_labels, pca_model, pca2_model, k_opt, km_metrics.
    """
    print("\n" + "="*55)
    print("  CLUSTERING — Segmentation Non Supervisee")
    print("="*55)

    pca_model,  X_pca = fit_pca(X_scaled)
    pca2_model, X_2d  = fit_pca_2d(X_scaled)

    # t-SNE (optionnel mais instructif)
    print("[clustering] Calcul t-SNE (peut prendre ~30 s)...")
    X_tsne = fit_tsne(X_pca)

    # Selection de k
    k_opt = select_k(X_pca)

    # KMeans
    km_model, km_labels, km_metrics = run_kmeans(X_pca, k_opt)
    plot_clusters_2d(X_2d,   km_labels, y, pca2_model, k_opt,
                      title_suffix="PCA 2D", file_suffix="pca")
    plot_clusters_2d(X_tsne, km_labels, y, pca2_model, k_opt,
                      title_suffix="t-SNE",  file_suffix="tsne")

    # Agglomeratif + Dendrogramme
    agg_labels, agg_metrics = run_agglomerative(X_pca, k_opt)

    # DBSCAN
    db_labels, n_db, db_metrics = run_dbscan(X_pca)

    # Comparaison
    comp = pd.DataFrame({
        "Methode":        ["KMeans", "Agglomerative", "DBSCAN"],
        "N Clusters":     [k_opt, k_opt, n_db],
        "Silhouette":     [km_metrics["Silhouette"],
                           agg_metrics["Silhouette"],
                           db_metrics["Silhouette"] or float("nan")],
        "Davies-Bouldin": [km_metrics["Davies-Bouldin"],
                           float("nan"), float("nan")],
    })
    print("\n[clustering] Comparaison des methodes :")
    print(comp.round(4).to_string(index=False))

    return km_model, km_labels, pca_model, pca2_model, k_opt, km_metrics
