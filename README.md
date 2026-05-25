# Telco Customer Churn — Pipeline Multi-Signal

**Projet Final — Pratique de la Data Science**  
M2 ISF — Université Paris-Dauphine | PSL  


Groupe : KEROUAD Lamyae · ZRIBI Rim · SBAA Nour · LIU Zhaoyi

---

## Objectif

Ce projet met en place un pipeline d'agrégation de signaux pour prédire le risque de churn client dans le secteur des télécommunications (dataset IBM Telco, 7 043 clients). Quatre modules sont combinés pour produire un score de risque composite et des recommandations de rétention personnalisées.

---

## Structure

```
├── main.py            # Point d'entrée — exécute l'ensemble du pipeline
├── project.py         # Chargement des données, prétraitement, utilitaires partagés
├── clustering.py      # Segmentation non supervisée (KMeans, Agglomératif, DBSCAN)
├── classification.py  # Prédiction du churn (7 modèles ML + MLP PyTorch + SHAP)
├── regression.py      # Prédiction de MonthlyCharges (proxy valeur client)
├── nlp.py             # Analyse de sentiment sur avis clients (TextBlob)
└── requirements.txt   # Dépendances
```

---

## En cas de problème de compilation

En cas d'erreur liée à l'encodage, lancer :

```bash
set PYTHONUTF8=1
python -X utf8 main.py
```
