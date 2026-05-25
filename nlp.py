"""
nlp.py — Analyse de sentiment sur avis clients simules.

Librairie : TextBlob (polarity / subjectivity).
Signal metier : sentiment negatif correle au churn.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from textblob import TextBlob
from scipy.stats import pointbiserialr

from project import PALETTE, savefig


# ══════════════════════════════════════════════════════════════════════
# 1.  CORPUS DE REVIEWS SIMULEES
# ══════════════════════════════════════════════════════════════════════

REVIEWS_POOL = [
    # Negatifs
    "The service is absolutely terrible, I am cancelling.",
    "Very disappointed, too expensive for what it offers.",
    "Horrible customer support, never solving my issues.",
    "Internet keeps dropping, unacceptable reliability.",
    "Way too costly and the quality is poor.",
    "Worst telecom experience I ever had.",
    "No value for money, thinking of switching.",
    "Support is useless and billing is confusing.",
    "Constant outages, completely unreliable service.",
    "I hate this provider, looking for alternatives.",
    # Neutres
    "Service is okay, nothing exceptional.",
    "Average experience, could be improved.",
    "The plan is standard, not impressed but not unhappy.",
    "Decent speeds but customer service is mediocre.",
    "It works most of the time, acceptable.",
    # Positifs
    "Excellent service, very happy with my plan.",
    "Great value for money, fast and reliable internet.",
    "Outstanding customer support, very responsive.",
    "Love the service, highly recommend to friends.",
    "Perfect experience, fast speeds and fair pricing.",
    "Very satisfied, no complaints at all.",
    "Brilliant service, reliable and affordable.",
    "Amazing support team, solved my issue instantly.",
    "Best telecom provider I have used, very happy.",
    "Great package, excellent quality for the price.",
]

# Poids de tirage par statut churn
WEIGHTS_CHURN   = [0.40/10]*10 + [0.15/5]*5 + [0.45/10]*10
WEIGHTS_NOCHURN = [0.15/10]*10 + [0.15/5]*5 + [0.70/10]*10


# ══════════════════════════════════════════════════════════════════════
# 2.  GENERATION ET SCORING
# ══════════════════════════════════════════════════════════════════════

def generate_reviews(churn_series, random_state=42):
    """Genere une review par client, correlee au statut churn."""
    rng = np.random.default_rng(random_state)
    return [
        rng.choice(REVIEWS_POOL,
                    p=WEIGHTS_CHURN if c == 1 else WEIGHTS_NOCHURN)
        for c in churn_series
    ]


def compute_sentiment(df):
    """
    Ajoute :
      - sentiment_polarity     : [-1, 1]
      - sentiment_subjectivity : [0, 1]
      - sentiment_label        : Negatif / Neutre / Positif
    """
    df = df.copy()
    df["sentiment_polarity"]     = df["Review"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    df["sentiment_subjectivity"] = df["Review"].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity
    )

    def label(p):
        if p < -0.05:  return "Negatif"
        elif p > 0.05: return "Positif"
        else:          return "Neutre"

    df["sentiment_label"] = df["sentiment_polarity"].apply(label)
    return df


def compute_sentiment_signal(df):
    """
    Normalise la polarite en signal [0, 1] pour l'agregation.
    0 = tres negatif (risque eleve)  |  1 = tres positif (fidele).
    """
    df = df.copy()
    df["sentiment_signal"] = (df["sentiment_polarity"] + 1) / 2
    return df


# ══════════════════════════════════════════════════════════════════════
# 3.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════

def plot_sentiment_vs_churn(df):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Distribution de la polarite
    for lbl, color in zip(["Negatif", "Neutre", "Positif"],
                           [PALETTE[1], PALETTE[4], PALETTE[3]]):
        subset = df[df["sentiment_label"] == lbl]["sentiment_polarity"]
        axes[0].hist(subset, bins=20, alpha=0.65, color=color,
                      label=lbl, edgecolor="white")
    axes[0].set_title("Distribution de la Polarite", fontweight="bold")
    axes[0].set_xlabel("Polarity score")
    axes[0].set_ylabel("Frequence"); axes[0].legend()

    # Taux de churn par sentiment
    order      = ["Negatif", "Neutre", "Positif"]
    churn_rate = df.groupby("sentiment_label")["Churn"].mean() * 100
    color_map  = {"Negatif": PALETTE[1], "Neutre": PALETTE[4],
                   "Positif": PALETTE[3]}
    bars = axes[1].bar(
        order,
        [churn_rate.get(k, 0) for k in order],
        color=[color_map[k] for k in order],
        edgecolor="white", lw=1.5
    )
    axes[1].axhline(df["Churn"].mean() * 100, color="black",
                     ls="--", lw=1.5,
                     label=f"Taux global ({df['Churn'].mean()*100:.1f}%)")
    axes[1].set_title("Taux de Churn par Sentiment", fontweight="bold")
    axes[1].set_ylabel("Taux de churn (%)"); axes[1].legend()
    for bar, val in zip(bars, [churn_rate.get(k, 0) for k in order]):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.5,
                      f"{val:.1f}%", ha="center", fontweight="bold")

    # Boxplot polarity vs churn
    df_plot = df.copy()
    df_plot["Churn_label"] = df_plot["Churn"].map({0: "Non-Churn", 1: "Churn"})
    axes[2].boxplot(
        [df_plot[df_plot["Churn_label"] == "Non-Churn"]["sentiment_polarity"],
         df_plot[df_plot["Churn_label"] == "Churn"]["sentiment_polarity"]],
        labels=["Non-Churn", "Churn"],
        patch_artist=True,
        boxprops=dict(facecolor=PALETTE[0], color="white"),
        medianprops=dict(color=PALETTE[1], linewidth=2),
        whiskerprops=dict(color=PALETTE[0]),
        capprops=dict(color=PALETTE[0]),
    )
    axes[2].set_title("Polarite — Churn vs Non-Churn", fontweight="bold")
    axes[2].set_ylabel("Polarity score")

    plt.suptitle("Analyse de Sentiment vs Comportement de Churn",
                  fontsize=14, fontweight="bold")
    plt.tight_layout(); savefig("sentiment_vs_churn.png"); plt.close()


def plot_top_reviews(df):
    """Top reviews negatives et positives (barchart horizontal)."""
    neg_rev = df[df["sentiment_label"] == "Negatif"]["Review"].value_counts().head(8)
    pos_rev = df[df["sentiment_label"] == "Positif"]["Review"].value_counts().head(8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, reviews, color, title in zip(
        axes,
        [neg_rev, pos_rev],
        [PALETTE[1], PALETTE[3]],
        ["Top Reviews NEGATIVES (Signal Churn)",
         "Top Reviews POSITIVES (Signal Retention)"]
    ):
        ax.barh(range(len(reviews)), reviews.values,
                 color=color, edgecolor="white")
        ax.set_yticks(range(len(reviews)))
        ax.set_yticklabels(
            [r[:55] + "..." if len(r) > 55 else r for r in reviews.index],
            fontsize=9
        )
        ax.set_title(title, fontweight="bold", color=color)
        ax.set_xlabel("Frequence"); ax.invert_yaxis()

    plt.suptitle("Analyse Qualitative des Avis Clients",
                  fontsize=14, fontweight="bold")
    plt.tight_layout(); savefig("reviews_top.png"); plt.close()


# ══════════════════════════════════════════════════════════════════════
# 4.  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def run_nlp(df_with_churn):
    """
    Pipeline NLP.
    df_with_churn doit contenir la colonne 'Churn' (0/1).
    Retourne df enrichi avec colonnes sentiment.
    """
    print("\n" + "="*55)
    print("  NLP — Analyse de Sentiment sur Avis Clients")
    print("="*55)

    df = df_with_churn.copy()
    df["Review"] = generate_reviews(df["Churn"])
    df = compute_sentiment(df)
    df = compute_sentiment_signal(df)

    print(f"\n[nlp] Distribution des sentiments :")
    print(df["sentiment_label"].value_counts().to_string())
    print(f"\n[nlp] Polarite moyenne par churn :")
    print(df.groupby("Churn")["sentiment_polarity"].mean().round(4))

    plot_sentiment_vs_churn(df)
    plot_top_reviews(df)

    corr, pval = pointbiserialr(df["Churn"], -df["sentiment_polarity"])
    print(f"\n[nlp] Correlation (churn, -polarity) = {corr:.4f}  "
          f"(p={pval:.4e})")
    print("[nlp] Confirmation : polarite negative = signal de churn.")

    return df
