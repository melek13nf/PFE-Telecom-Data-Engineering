"""
=============================================================
  train_models_cv.py — Tunisie Telecom PFE
  Entraînement avec Validation Croisée Stratifiée (K-Fold)

  Fichier source : CIBLE_ECH_DEC_2025_6.xlsx
  Modèle 1 : Activation Forfait DATA  (TARGET      : 0/1)
  Modèle 2 : Prédiction Churn         (FLAG_CHURN  : 0/1)

  Méthode : StratifiedKFold (5 folds)
    → Fold 1 : train sur [2,3,4,5] — test sur [1]
    → Fold 2 : train sur [1,3,4,5] — test sur [2]
    → ...
    → Fold 5 : train sur [1,2,3,4] — test sur [5]
  Sortie   : model_forfait_cv.joblib + model_churn_cv.joblib
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, average_precision_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── CONFIGURATION ────────────────────────────────────────
import glob

# Recherche automatique du fichier CIBLE dans le dossier data/
# Accepte tous les noms contenant "CIBLE" ou "ECH" avec extension xlsx
def find_cible_file():
    patterns = [
        "data/CIBLE*.xlsx",
        "data/cible*.xlsx",
        "data/ECH*.xlsx",
        "data/ech*.xlsx",
        "data/CIBLE_ECH*.xlsx",
    ]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    # Si rien trouvé, lister ce qui existe dans data/
    all_xlsx = glob.glob("data/*.xlsx")
    if all_xlsx:
        print(f"  Fichiers xlsx disponibles dans data/ : {all_xlsx}")
        print(f"  Utilisation de : {all_xlsx[0]}")
        return all_xlsx[0]
    return None

DATA_FILE = find_cible_file() or "data/CIBLE_ECH_DEC_2025_6.xlsx"
N_FOLDS      = 5        # Nombre de folds pour la CV
MODEL_FORFAIT= "model_forfait_cv.joblib"
MODEL_CHURN  = "model_churn_cv.joblib"
REPORT_DIR   = "outputs/model_report/"

# ─── 1. CHARGEMENT ────────────────────────────────────────
def load_data():
    print(f"\n{'='*58}")
    print("  CHARGEMENT DES DONNÉES")
    print(f"{'='*58}")

    if DATA_FILE is None or not os.path.exists(DATA_FILE):
        # Lister tous les xlsx disponibles pour aider l utilisateur
        import glob
        all_xlsx = glob.glob("data/*.xlsx")
        msg = (
            f"Aucun fichier CIBLE trouvé dans data/\n"
            f"Fichiers disponibles : {all_xlsx}\n"
            f"Renommez votre fichier en CIBLE_ECH_DEC_2025_6.xlsx"
            f" et placez-le dans le dossier data/"
        )
        raise FileNotFoundError(msg)

    df = pd.read_excel(DATA_FILE)
    df.columns = df.columns.str.strip().str.upper()
    print(f"  Fichier chargé : {df.shape[0]:,} clients x {df.shape[1]} colonnes")

    # Afficher la distribution des cibles
    print(f"\n  Modèle 1 — TARGET (Activation Forfait) :")
    print(f"    TARGET=0 (pas de forfait) : {(df['TARGET']==0).sum():,} "
          f"({(df['TARGET']==0).mean()*100:.1f}%)")
    print(f"    TARGET=1 (forfait activé) : {(df['TARGET']==1).sum():,} "
          f"({(df['TARGET']==1).mean()*100:.1f}%)")

    print(f"\n  Modèle 2 — FLAG_CHURN (Attrition) :")
    print(f"    CHURN=0 (client stable)   : {(df['FLAG_CHURN']==0).sum():,} "
          f"({(df['FLAG_CHURN']==0).mean()*100:.1f}%)")
    print(f"    CHURN=1 (client à risque) : {(df['FLAG_CHURN']==1).sum():,} "
          f"({(df['FLAG_CHURN']==1).mean()*100:.1f}%)")

    return df

# ─── 2. PRÉPARATION DES FEATURES ──────────────────────────
def prepare_features(df):
    print(f"\n{'─'*58}")
    print("  PRÉPARATION DES FEATURES")
    print(f"{'─'*58}")

    df = df.copy()

    # Nettoyage valeurs manquantes
    df['HANDSET']      = df['HANDSET'].fillna('Inconnu').astype(str)
    df['CLASSE_CANAL'] = df['CLASSE_CANAL'].fillna('Inconnu').astype(str)
    df['CODE_REGION']  = df['CODE_REGION'].fillna(df['CODE_REGION'].median())

    # Encodage LabelEncoder des variables catégorielles
    encoders = {}
    cat_cols  = ['HANDSET', 'STATUT', 'CLASSE_CANAL']
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_ENC'] = le.fit_transform(df[col].astype(str))
        encoders[col]    = le
        classes = list(le.classes_)
        print(f"  {col:15s} → {len(classes)} modalités : {classes}")

    # Features numériques
    NUM_FEATS = ['ANC_M', 'ANC_J', 'ID_OFFRE', 'CODE_REGION']
    CAT_FEATS = [c + '_ENC' for c in cat_cols]
    ALL_FEATS = NUM_FEATS + CAT_FEATS

    X = df[ALL_FEATS].apply(pd.to_numeric, errors='coerce').fillna(0)

    print(f"\n  Features retenues ({len(ALL_FEATS)}) :")
    print(f"    Numériques   : {NUM_FEATS}")
    print(f"    Catégorielles: {CAT_FEATS}")

    return X, df, ALL_FEATS, encoders

# ─── 3. CROSS-VALIDATION + ENTRAÎNEMENT ───────────────────
def train_with_cv(X, y, model_name, n_folds=5):
    """
    Validation Croisée Stratifiée (StratifiedKFold).

    Principe :
    - Les données sont divisées en N folds (partitions) de taille égale
    - Chaque fold est utilisé 1 fois comme jeu de test
    - Les N-1 folds restants servent à l'entraînement
    - On calcule les métriques sur chaque fold puis on fait la moyenne

    Avantage vs simple train/test split :
    - Évalue la STABILITÉ du modèle (variance des AUC entre folds)
    - Utilise 100% des données pour l'évaluation (pas de données perdues)
    - Plus fiable pour les datasets déséquilibrés (stratified = proportions conservées)
    """
    print(f"\n{'─'*58}")
    print(f"  CROSS-VALIDATION {n_folds}-FOLD — {model_name}")
    print(f"{'─'*58}")

    # Ratio déséquilibre
    ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"  Déséquilibre : ratio négatifs/positifs = {ratio:.2f}")
    print(f"  scale_pos_weight = {ratio:.2f}")

    # Création du modèle XGBoost
    model = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = ratio,
        eval_metric      = 'auc',
        random_state     = 42,
        verbosity        = 0,
    )

    # Validation croisée stratifiée
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_aucs   = []
    fold_aps    = []
    fold_accs   = []

    print(f"\n  {'Fold':>5} | {'AUC':>8} | {'AP Score':>10} | {'Accuracy':>10} | {'Statut'}")
    print(f"  {'-'*55}")

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Entraînement sur ce fold
        model.fit(X_train, y_train, verbose=False)

        # Prédictions
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Métriques
        auc = roc_auc_score(y_test, y_proba)
        ap  = average_precision_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        fold_aucs.append(auc)
        fold_aps.append(ap)
        fold_accs.append(acc)

        statut = "✅ Bon" if auc >= 0.70 else "⚠️  Moyen"
        print(f"  {fold_num:>5} | {auc:>8.4f} | {ap:>10.4f} | {acc:>10.4f} | {statut}")

    print(f"  {'-'*55}")
    print(f"  {'Moy.':>5} | {np.mean(fold_aucs):>8.4f} | {np.mean(fold_aps):>10.4f} | "
          f"{np.mean(fold_accs):>10.4f} |")
    print(f"  {'Std':>5} | {np.std(fold_aucs):>8.4f} | {np.std(fold_aps):>10.4f} | "
          f"{np.std(fold_accs):>10.4f} |")

    print(f"\n  AUC Moyenne  : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  AP Moyenne   : {np.mean(fold_aps):.4f}  ± {np.std(fold_aps):.4f}")
    print(f"  Acc Moyenne  : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")

    # Entraînement final sur TOUTES les données
    print(f"\n  Entraînement final sur 100% des données...")
    model.fit(X, y, verbose=False)
    print(f"  Modèle final entraîné.")

    metrics = {
        'auc_cv_mean'  : round(float(np.mean(fold_aucs)), 4),
        'auc_cv_std'   : round(float(np.std(fold_aucs)),  4),
        'ap_cv_mean'   : round(float(np.mean(fold_aps)),  4),
        'acc_cv_mean'  : round(float(np.mean(fold_accs)), 4),
        'n_folds'      : n_folds,
        'fold_aucs'    : [round(float(a), 4) for a in fold_aucs],
    }

    return model, metrics, fold_aucs

# ─── 4. GRAPHIQUES ────────────────────────────────────────
def plot_cv_results(fold_aucs_forfait, fold_aucs_churn,
                    metrics_forfait, metrics_churn):

    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Résultats de la Validation Croisée 5-Fold — Tunisie Telecom PFE",
        fontsize=14, fontweight='bold', y=1.02
    )

    for ax, fold_aucs, metrics, title, color in [
        (axes[0], fold_aucs_forfait, metrics_forfait,
         "Modèle 1 — Activation Forfait DATA", "#004a99"),
        (axes[1], fold_aucs_churn,  metrics_churn,
         "Modèle 2 — Prédiction Churn",        "#ef4444"),
    ]:
        folds = [f"Fold {i+1}" for i in range(len(fold_aucs))]
        bar_colors = [color if a >= np.mean(fold_aucs) else "#93c5fd"
                      for a in fold_aucs]
        bars = ax.bar(folds, fold_aucs, color=bar_colors,
                      edgecolor='white', width=0.6)

        # Valeur sur chaque barre
        for bar, val in zip(bars, fold_aucs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        # Ligne moyenne
        ax.axhline(np.mean(fold_aucs), color='black',
                   linestyle='--', lw=2,
                   label=f"Moy = {np.mean(fold_aucs):.4f}")

        # Bande ±1 std
        ax.fill_between(range(len(folds)),
                        np.mean(fold_aucs) - np.std(fold_aucs),
                        np.mean(fold_aucs) + np.std(fold_aucs),
                        alpha=0.15, color=color,
                        label=f"±1σ = {np.std(fold_aucs):.4f}")

        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel("AUC ROC", fontsize=11)
        ax.set_ylim(max(0, min(fold_aucs) - 0.05), 1.0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "rapport_cv_models.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Graphique sauvegardé : {path}")

# ─── 5. SAUVEGARDE ────────────────────────────────────────
def save_model(model, encoders, features, metrics, path, label):
    artifact = {
        'model'        : model,
        'encoders'     : encoders,
        'feature_names': features,
        'metrics'      : metrics,
        'label'        : label,
    }
    joblib.dump(artifact, path)
    print(f"  Modèle sauvegardé : {path}")
    print(f"    AUC CV : {metrics['auc_cv_mean']:.4f} ± {metrics['auc_cv_std']:.4f}")

# ─── 6. RAPPORT CLASSIFICATION FINAL ─────────────────────
def final_report(model, X, y, label):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    print(f"\n  Rapport final — {label} :")
    print(classification_report(
        y, y_pred,
        target_names=['Classe 0 (Négatif)', 'Classe 1 (Positif)'],
        digits=4
    ))
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Matrice de confusion :")
    print(f"    Vrais Négatifs (TN) : {tn:,}")
    print(f"    Faux Positifs  (FP) : {fp:,}")
    print(f"    Faux Négatifs  (FN) : {fn:,}")
    print(f"    Vrais Positifs (TP) : {tp:,}")

# ─── PIPELINE PRINCIPAL ───────────────────────────────────
def run():
    print(f"\n{'='*58}")
    print("  ENTRAÎNEMENT DES MODÈLES — Tunisie Telecom PFE")
    print("  Méthode : Cross-Validation Stratifiée (5-Fold)")
    print(f"{'='*58}")

    # 1. Charger les données
    df = load_data()

    # 2. Préparer les features
    X, df_clean, features, encoders = prepare_features(df)

    # Cibles
    y_forfait = df_clean['TARGET']
    y_churn   = df_clean['FLAG_CHURN']

    # ── MODÈLE 1 : Activation Forfait DATA ────────────────
    print(f"\n{'═'*58}")
    print("  MODÈLE 1 — ACTIVATION FORFAIT DATA")
    print(f"{'═'*58}")
    model_forfait, metrics_forfait, aucs_forfait = train_with_cv(
        X, y_forfait, "Activation Forfait DATA", N_FOLDS
    )
    final_report(model_forfait, X, y_forfait, "Activation Forfait")

    # ── MODÈLE 2 : Prédiction Churn ───────────────────────
    print(f"\n{'═'*58}")
    print("  MODÈLE 2 — PRÉDICTION CHURN")
    print(f"{'═'*58}")
    model_churn, metrics_churn, aucs_churn = train_with_cv(
        X, y_churn, "Prédiction Churn", N_FOLDS
    )
    final_report(model_churn, X, y_churn, "Prédiction Churn")

    # ── Graphiques CV ─────────────────────────────────────
    plot_cv_results(aucs_forfait, aucs_churn, metrics_forfait, metrics_churn)

    # ── Sauvegarde des modèles ────────────────────────────
    print(f"\n{'─'*58}")
    print("  SAUVEGARDE DES MODÈLES")
    print(f"{'─'*58}")
    save_model(model_forfait, encoders, features, metrics_forfait,
               MODEL_FORFAIT, "Activation Forfait DATA")
    save_model(model_churn,   encoders, features, metrics_churn,
               MODEL_CHURN,   "Prédiction Churn")

    # ── Résumé final ──────────────────────────────────────
    print(f"\n{'='*58}")
    print("  RÉSUMÉ FINAL")
    print(f"{'='*58}")
    print(f"  Modèle 1 — Activation Forfait DATA")
    print(f"    AUC CV  : {metrics_forfait['auc_cv_mean']:.4f} "
          f"± {metrics_forfait['auc_cv_std']:.4f}")
    print(f"    AP  CV  : {metrics_forfait['ap_cv_mean']:.4f}")
    print(f"    Acc CV  : {metrics_forfait['acc_cv_mean']:.4f}")
    print(f"    Fichier : {MODEL_FORFAIT}")
    print()
    print(f"  Modèle 2 — Prédiction Churn")
    print(f"    AUC CV  : {metrics_churn['auc_cv_mean']:.4f} "
          f"± {metrics_churn['auc_cv_std']:.4f}")
    print(f"    AP  CV  : {metrics_churn['ap_cv_mean']:.4f}")
    print(f"    Acc CV  : {metrics_churn['acc_cv_mean']:.4f}")
    print(f"    Fichier : {MODEL_CHURN}")
    print()
    print(f"  Graphique : {REPORT_DIR}rapport_cv_models.png")
    print(f"{'='*58}")
    print("  ENTRAÎNEMENT TERMINÉ AVEC SUCCES")
    print(f"{'='*58}\n")

if __name__ == "__main__":
    run()