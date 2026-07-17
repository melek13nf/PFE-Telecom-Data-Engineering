"""
=============================================================
  train_model.py — Tunisie Telecom PFE
  Modèle XGBoost d'appétence DATA
  Entrée  : ABT_Final_ML.csv (produit par preprocessing_pfe_fixed.py)
  Sortie  : model_appetence_tt.joblib + rapport d'évaluation complet
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
ABT_PATH   = "ABT_Final_ML.csv"
MODEL_PATH = "model_appetence_tt.joblib"
REPORT_DIR = "outputs/model_report/"

# ─── FEATURES SÉLECTIONNÉES ───────────────────────────────────────────────────
# ⚠️  RÈGLE ANTI-LEAKAGE :
#   TARGET_IA = (MNT_FORFAIT_DATA_TOT > 0)
#   EXCLUES (leakage direct) :
#     ❌ MNT_FORFAIT_TOT      → contient MNT_FORFAIT_DATA_TOT
#     ❌ MNT_FORFAIT_DATA_TOT → EST la cible
#     ❌ NB_FORFAIT_DATA_TOT  → expose directement la cible
#   INCLUSES : comportement voix/recharge uniquement.
FEATURES_NUM = [
    'ANC_M',              # Ancienneté client (mois)
    'MNT_RECH_TOT',       # Montant total rechargé
    'MNT_RECH_MOY',       # Recharge mensuelle moyenne
    'NB_RECH_TOT',        # Nombre total de recharges
    'NB_MOIS_ACTIF_R',    # Nb de mois avec activité recharge
    'REVENU_CDR_TOT',     # Revenu voix/SMS total
    'DUREE_APPEL_TOT',    # Durée voix sortante totale (min)
    'DUREE_OFFNET_TOT',   # Durée appels vers autres opérateurs
    'NB_APPEL_TOT',       # Nombre d'appels sortants
    'NB_SMS_TOT',         # Nombre de SMS sortants
    'REVENU_INTER_TOT',   # Revenu appels internationaux
    'DUREE_APPEL_IN_TOT', # Durée voix entrante totale
    'NB_SMS_IN_TOT',      # Nombre SMS entrants
    # MNT_FORFAIT_TOT     ← RETIRÉ : leakage (contient la cible)
]

FEATURES_CAT = [
    'HANDSET',    # Type réseau : 2G, 3G, 4G, 5G
    'STATUT',     # Statut client : A (actif), S (suspendu)
    'OFFRE_CAT',  # Catégorie commerciale de l'offre
]

TARGET = 'TARGET_IA'

# ─── 1. CHARGEMENT & NETTOYAGE ────────────────────────────────────────────────
def load_abt(path):
    print(f"\n{'='*60}")
    print("  CHARGEMENT DE L'ABT")
    print(f"{'='*60}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ '{path}' introuvable.\n"
            "   → Lancez d'abord preprocessing_pfe_fixed.py"
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()
    print(f"  ✅ ABT chargée : {df.shape[0]:,} clients × {df.shape[1]} colonnes")
    print(f"  🎯 Cible TARGET_IA : {df[TARGET].sum():,} positifs "
          f"({df[TARGET].mean()*100:.1f}%) / "
          f"{(df[TARGET]==0).sum():,} négatifs ({(df[TARGET]==0).mean()*100:.1f}%)")
    return df

# ─── 2. PRÉPARATION DES FEATURES ──────────────────────────────────────────────
def prepare_features(df):
    print(f"\n{'─'*60}")
    print("  PRÉPARATION DES FEATURES")
    print(f"{'─'*60}")

    df = df.copy()

    # Nettoyage valeurs aberrantes (2 clients avec recharge négative)
    df['MNT_RECH_TOT'] = df['MNT_RECH_TOT'].clip(lower=0)
    df['MNT_RECH_MOY'] = df['MNT_RECH_MOY'].clip(lower=0)

    # Normalisation HANDSET : remplacer 0 (NaN d'origine) par 'Inconnu'
    df['HANDSET'] = df['HANDSET'].replace(0, 'Inconnu').astype(str)
    df['STATUT']  = df['STATUT'].astype(str)
    df['OFFRE_CAT'] = df['OFFRE_CAT'].astype(str)

    # Encodage des variables catégorielles (Label Encoding pour XGBoost)
    encoders = {}
    for col in FEATURES_CAT:
        le = LabelEncoder()
        df[col + '_ENC'] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  🔤 {col} encodé → {len(le.classes_)} modalités : {list(le.classes_)}")

    # Liste finale des features
    features_enc = [c + '_ENC' for c in FEATURES_CAT]
    all_features  = FEATURES_NUM + features_enc

    # Vérification que toutes les features existent
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        raise ValueError(f"❌ Features manquantes dans l'ABT : {missing}")

    X = df[all_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df[TARGET].astype(int)

    print(f"\n  ✅ {len(all_features)} features prêtes pour le modèle")
    print(f"  📋 Features numériques ({len(FEATURES_NUM)}) : {FEATURES_NUM}")
    print(f"  📋 Features catégorielles encodées ({len(features_enc)}) : {features_enc}")

    return X, y, all_features, encoders

# ─── 3. ENTRAÎNEMENT ──────────────────────────────────────────────────────────
def train(X, y):
    print(f"\n{'─'*60}")
    print("  ENTRAÎNEMENT XGBoost")
    print(f"{'─'*60}")

    # Calcul du ratio pour scale_pos_weight (gestion déséquilibre 85%/15%)
    ratio = (y == 0).sum() / (y == 1).sum()
    print(f"  ⚖️  Déséquilibre détecté : ratio négatifs/positifs = {ratio:.2f}")
    print(f"      → scale_pos_weight = {ratio:.2f} appliqué automatiquement")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  📊 Split : {len(X_train):,} train | {len(X_test):,} test (stratifié)")

    model = XGBClassifier(
        n_estimators      = 200,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = ratio,   # Compense le déséquilibre des classes
        eval_metric       = 'auc',
        early_stopping_rounds = 20,
        random_state      = 42,
        verbosity         = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    best_iter = model.best_iteration
    print(f"  🛑 Early stopping : meilleur arbre à l'itération {best_iter}")

    return model, X_train, X_test, y_train, y_test

# ─── 4. ÉVALUATION COMPLÈTE ───────────────────────────────────────────────────
def evaluate(model, X_train, X_test, y_train, y_test, feature_names):
    print(f"\n{'─'*60}")
    print("  ÉVALUATION DU MODÈLE")
    print(f"{'─'*60}")

    y_pred       = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_proba_train= model.predict_proba(X_train)[:, 1]

    auc_test  = roc_auc_score(y_test,  y_proba_test)
    auc_train = roc_auc_score(y_train, y_proba_train)
    ap_score  = average_precision_score(y_test, y_proba_test)

    print(f"\n  📈 AUC  Train : {auc_train:.4f}")
    print(f"  📈 AUC  Test  : {auc_test:.4f}  {'✅ Bon' if auc_test > 0.75 else '⚠️ À améliorer'}")
    print(f"  📈 AP Score   : {ap_score:.4f}  (Average Precision — utile si déséquilibre)")

    gap = auc_train - auc_test
    if gap > 0.05:
        print(f"  ⚠️  Écart Train/Test : {gap:.4f} → léger sur-apprentissage détecté")
    else:
        print(f"  ✅ Écart Train/Test : {gap:.4f} → bonne généralisation")

    # Validation croisée (5 folds) pour robustesse
    print(f"\n  🔁 Validation croisée 5-Fold (sur train) ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Reconstruire le modèle sans early stopping pour cv
    model_cv = XGBClassifier(
        n_estimators     = model.best_iteration,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = (y_train==0).sum()/(y_train==1).sum(),
        eval_metric      = 'auc',
        random_state     = 42,
        verbosity        = 0,
    )
    cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"  AUC CV : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Scores : {[f'{s:.4f}' for s in cv_scores]}")

    # Rapport de classification
    print(f"\n  📋 Rapport de Classification :")
    print(classification_report(y_test, y_pred,
                                 target_names=['Non-Data (0)', 'Data (1)'],
                                 digits=4))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Matrice de Confusion :")
    print(f"    {'':20s} Prédit Non-Data  Prédit Data")
    print(f"    Réel Non-Data  :  {tn:>10,}      {fp:>10,}")
    print(f"    Réel Data      :  {fn:>10,}      {tp:>10,}")

    # Importance des features
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\n  🔍 Top 10 Variables les plus importantes :")
    for _, row in importances.head(10).iterrows():
        bar = '█' * int(row['Importance'] * 200)
        print(f"    {row['Feature']:30s} {row['Importance']:.4f}  {bar}")

    return y_proba_test, auc_test, ap_score, cv_scores, importances

# ─── 5. GRAPHIQUES ────────────────────────────────────────────────────────────
def generate_plots(model, X_test, y_test, y_proba_test, importances,
                   cv_scores, auc_test, ap_score, report_dir):
    os.makedirs(report_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Rapport d'Évaluation – Modèle XGBoost Appétence DATA\nTunisie Telecom PFE",
                 fontsize=16, fontweight='bold', color='#004a99', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── Graphique 1 : Courbe ROC ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    ax1.plot(fpr, tpr, color='#004a99', lw=2.5, label=f'AUC = {auc_test:.4f}')
    ax1.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Aléatoire (AUC=0.50)')
    ax1.fill_between(fpr, tpr, alpha=0.08, color='#004a99')
    ax1.set_xlabel('Taux de Faux Positifs', fontsize=11)
    ax1.set_ylabel('Taux de Vrais Positifs', fontsize=11)
    ax1.set_title('Courbe ROC', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Graphique 2 : Courbe Precision-Recall ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    prec, rec, _ = precision_recall_curve(y_test, y_proba_test)
    ax2.plot(rec, prec, color='#10b981', lw=2.5, label=f'AP = {ap_score:.4f}')
    ax2.axhline(y=y_test.mean(), color='gray', linestyle='--', lw=1,
                label=f'Baseline = {y_test.mean():.2f}')
    ax2.fill_between(rec, prec, alpha=0.08, color='#10b981')
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Précision', fontsize=11)
    ax2.set_title('Courbe Précision-Rappel', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ── Graphique 3 : Distribution des scores ─────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    scores_pos = y_proba_test[y_test == 1]
    scores_neg = y_proba_test[y_test == 0]
    ax3.hist(scores_neg, bins=40, alpha=0.6, color='#ef4444',
             label=f'Non-Data (n={len(scores_neg):,})', density=True)
    ax3.hist(scores_pos, bins=40, alpha=0.6, color='#10b981',
             label=f'Data (n={len(scores_pos):,})', density=True)
    ax3.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Seuil = 0.5')
    ax3.set_xlabel('Score de Propension', fontsize=11)
    ax3.set_ylabel('Densité', fontsize=11)
    ax3.set_title('Distribution des Scores', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Graphique 4 : Importance des features ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    top_imp = importances.head(14)
    colors_imp = ['#004a99' if i < 5 else '#0ea5e9' if i < 10 else '#bae6fd'
                  for i in range(len(top_imp))]
    bars = ax4.barh(top_imp['Feature'][::-1], top_imp['Importance'][::-1],
                    color=colors_imp[::-1], edgecolor='white')
    for bar, val in zip(bars, top_imp['Importance'][::-1]):
        ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9)
    ax4.set_xlabel('Importance (gain)', fontsize=11)
    ax4.set_title('Importance des Variables (Top 14)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, top_imp['Importance'].max() * 1.18)

    # ── Graphique 5 : Validation croisée ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
    bar_colors = ['#10b981' if s >= cv_scores.mean() else '#f97316' for s in cv_scores]
    ax5.bar(folds, cv_scores, color=bar_colors, edgecolor='white', width=0.6)
    ax5.axhline(y=cv_scores.mean(), color='#004a99', linestyle='--', lw=2,
                label=f'Moyenne = {cv_scores.mean():.4f}')
    ax5.fill_between(range(len(folds)),
                     cv_scores.mean() - cv_scores.std(),
                     cv_scores.mean() + cv_scores.std(),
                     alpha=0.15, color='#004a99', label=f'±1 std ({cv_scores.std():.4f})')
    for i, (fold, score) in enumerate(zip(folds, cv_scores)):
        ax5.text(i, score + 0.003, f'{score:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax5.set_ylim(min(cv_scores) - 0.03, 1.02)
    ax5.set_ylabel('AUC', fontsize=11)
    ax5.set_title('Validation Croisée 5-Fold', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    plot_path = os.path.join(report_dir, 'rapport_evaluation_xgboost.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  📊 Rapport graphique sauvegardé : {plot_path}")
    return plot_path

# ─── 6. SAUVEGARDE ────────────────────────────────────────────────────────────
def save_model(model, encoders, feature_names, auc_test, ap_score, cv_scores):
    artifact = {
        'model'        : model,
        'encoders'     : encoders,       # LabelEncoders pour les variables catégorielles
        'feature_names': feature_names,  # Ordre exact des colonnes attendu par le modèle
        'features_num' : FEATURES_NUM,
        'features_cat' : FEATURES_CAT,
        'metrics': {
            'auc_test' : round(auc_test,  4),
            'ap_score' : round(ap_score,  4),
            'auc_cv_mean': round(cv_scores.mean(), 4),
            'auc_cv_std' : round(cv_scores.std(),  4),
        }
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n  💾 Modèle + encodeurs sauvegardés : {MODEL_PATH}")
    print(f"  📦 Contenu du .joblib :")
    print(f"     - model          : XGBClassifier entraîné")
    print(f"     - encoders       : LabelEncoders pour HANDSET, STATUT, OFFRE_CAT")
    print(f"     - feature_names  : ordre exact des {len(feature_names)} features")
    print(f"     - metrics        : AUC={auc_test:.4f} | AP={ap_score:.4f} "
          f"| CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────
def train_optimized_model():
    # 1. Charger l'ABT réelle
    df = load_abt(ABT_PATH)

    # 2. Préparer les features
    X, y, feature_names, encoders = prepare_features(df)

    # 3. Entraîner le modèle
    model, X_train, X_test, y_train, y_test = train(X, y)

    # 4. Évaluer
    y_proba, auc_test, ap_score, cv_scores, importances = evaluate(
        model, X_train, X_test, y_train, y_test, feature_names
    )

    # 5. Générer les graphiques
    os.makedirs(REPORT_DIR, exist_ok=True)
    generate_plots(model, X_test, y_test, y_proba, importances,
                   cv_scores, auc_test, ap_score, REPORT_DIR)

    # 6. Sauvegarder le modèle + encodeurs
    save_model(model, encoders, feature_names, auc_test, ap_score, cv_scores)

    print(f"\n{'='*60}")
    print("  ✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print(f"{'='*60}")
    print(f"  → Prochaine étape : lancer batch_scoring.py")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    train_optimized_model()