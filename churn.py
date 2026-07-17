"""
=============================================================
  churn_model.py — Tunisie Telecom PFE
  Modèle de Prédiction du Risque de Churn (Attrition)
  Entrée  : ABT_Final_ML.csv + rfm_segments.csv
  Sortie  : churn_model.joblib + rapport_churn.png
=============================================================

DÉFINITION DU CHURN :
  Un client est considéré "à risque de churn" si l'une
  des conditions suivantes est vérifiée :
    - STATUT = 'S'        (client suspendu)
    - STATUT_RGS90 = 'R'  (inactif depuis 90 jours)
    - RECENCY >= 3 mois   (pas de recharge depuis 3 mois)
  Taux de churn observé : 5,36% du parc
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
                             confusion_matrix, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_PATH    = "data"
RFM_CSV      = "outputs/rfm_segments.csv"
MODEL_PATH   = "churn_model.joblib"
REPORT_PNG   = "outputs/rapport_churn.png"
SCORING_CSV  = "outputs/churn_scoring.csv"

COLOR_MAIN   = "#004a99"
COLOR_CHURN  = "#ef4444"
COLOR_OK     = "#10b981"

FEATURES_NUM = [
    'ANC_M',           # Ancienneté (mois)
    'RECENCY',         # Mois depuis dernière recharge
    'FREQUENCY',       # Nb de mois actifs
    'MONETARY',        # Recharge totale
    'NB_RECH_TOT',     # Nb de recharges
    'MNT_MOY',         # Recharge mensuelle moyenne
    'RFM_SCORE',       # Score RFM global
]
FEATURES_CAT = ['HANDSET', 'OFFRE_CAT']
TARGET       = 'CHURN'

# ─── 1. CHARGEMENT & CONSTRUCTION DU DATASET CHURN ───────────────────────────
def load_churn_data(data_path, rfm_csv):
    print(f"\n{'='*55}")
    print("  MODÈLE DE PRÉDICTION DU CHURN — Tunisie Telecom")
    print(f"{'='*55}")

    # Charger RFM (contient déjà les scores et segments)
    if os.path.exists(rfm_csv):
        rfm = pd.read_csv(rfm_csv)
        rfm.columns = rfm.columns.str.strip().str.upper()
        print(f"  ✅ RFM chargé depuis {rfm_csv} : {len(rfm):,} clients")
    else:
        raise FileNotFoundError(
            f"❌ {rfm_csv} introuvable.\n   → Lancez d'abord rfm_analysis.py")

    # Charger ECH pour STATUT et STATUT_RGS90
    df_ech = pd.read_excel(os.path.join(data_path, 'ECH__DECEMBRE_2025 1.xlsx'))
    df_ech.columns = df_ech.columns.str.strip().str.upper()
    df_ech['ID'] = df_ech['ID'].astype(str).str.strip().str.upper()

    # Charger données SORTANT pour enrichir les features
    df_s = pd.read_excel(os.path.join(data_path, 'SORTANT_DECEMBRE_2025 1.xlsx'))
    df_s.columns = df_s.columns.str.strip().str.upper()
    df_s['ID'] = df_s['ID'].astype(str).str.strip().str.upper()

    s_agg = df_s.groupby('ID').agg(
        DUREE_APPEL_TOT  =('DUREE_APPEL_TOT', 'sum'),
        REVENU_CDR       =('REVENU_CDR',       'sum'),
        NB_SMS_TOT       =('NB_SMS_TOT',       'sum'),
        NB_APPEL_TOT     =('NB_APPEL_TOT',     'sum'),
        REVENU_INTER     =('REVENU_INTER',      'sum'),
    ).reset_index()

    # Normaliser l'ID du RFM (format CSV peut différer de l'Excel)
    rfm['ID'] = rfm['ID'].astype(str).str.strip().str.upper()

    # Merge ECH et SORTANT séparément
    df = rfm.merge(
        df_ech[['ID', 'STATUT', 'STATUT_RGS90']], on='ID', how='left'
    ).merge(s_agg, on='ID', how='left')

    # FIX : fillna différencié par type de colonne
    # Catégorielles : 'A' (actif par défaut si NaN post-merge)
    for cat_col, default in [('STATUT', 'A'), ('STATUT_RGS90', 'A')]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna(default).astype(str)
        else:
            print(f"  ⚠️  {cat_col} absente — valeur par défaut '{default}' appliquée")
            df[cat_col] = default

    # Numériques : 0
    num_cols_df = df.select_dtypes(include='number').columns
    df[num_cols_df] = df[num_cols_df].fillna(0)

    n_s = (df['STATUT'] == 'S').sum()
    n_r = (df['STATUT_RGS90'] == 'R').sum()
    print(f"  🔗 Merge OK — STATUT=S: {n_s}, STATUT_RGS90=R: {n_r}")

    # ── Définition du CHURN ───────────────────────────────────────────────
    df['CHURN'] = (
        (df['STATUT']       == 'S') |
        (df['STATUT_RGS90'] == 'R') |
        (df['RECENCY']      >= 3)
    ).astype(int)

    print(f"\n  ── Distribution de la variable cible CHURN ──────────────")
    print(f"  Clients stables (0) : {(df['CHURN']==0).sum():,} "
          f"({(df['CHURN']==0).mean()*100:.1f}%)")
    print(f"  Clients à risque (1): {(df['CHURN']==1).sum():,} "
          f"({(df['CHURN']==1).mean()*100:.1f}%)")

    return df

# ─── 2. PRÉPARATION DES FEATURES ──────────────────────────────────────────────
def prepare_features(df):
    print(f"\n{'─'*55}")
    print("  PRÉPARATION DES FEATURES")
    print(f"{'─'*55}")

    df = df.copy()
    df['HANDSET']   = df['HANDSET'].replace(0, 'Inconnu').astype(str)
    df['OFFRE_CAT'] = df['OFFRE_CAT'].astype(str)

    # Features numériques supplémentaires si disponibles
    extra_num = ['DUREE_APPEL_TOT', 'REVENU_CDR', 'NB_SMS_TOT',
                 'NB_APPEL_TOT', 'REVENU_INTER']
    features_num_all = FEATURES_NUM + [f for f in extra_num if f in df.columns]

    encoders = {}
    for col in FEATURES_CAT:
        if col not in df.columns: continue
        le = LabelEncoder()
        df[col + '_ENC'] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  🔤 {col} encodé → {len(le.classes_)} modalités")

    features_enc = [c + '_ENC' for c in FEATURES_CAT if c in df.columns]
    all_features = features_num_all + features_enc

    # Vérification
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"  ⚠️  Features manquantes (→ 0) : {missing}")
        for f in missing: df[f] = 0

    X = df[all_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df[TARGET].astype(int)

    print(f"\n  ✅ {len(all_features)} features prêtes")
    print(f"  Numériques ({len(features_num_all)}) : {features_num_all}")
    print(f"  Catégorielles encodées ({len(features_enc)}) : {features_enc}")

    return X, y, all_features, encoders, df

# ─── 3. ENTRAÎNEMENT ──────────────────────────────────────────────────────────
def train_churn_model(X, y):
    print(f"\n{'─'*55}")
    print("  ENTRAÎNEMENT DU MODÈLE CHURN (XGBoost)")
    print(f"{'─'*55}")

    # Gestion du déséquilibre (5.36% de churn)
    ratio = (y == 0).sum() / (y == 1).sum()
    print(f"  ⚖️  Ratio négatifs/positifs : {ratio:.1f} → scale_pos_weight={ratio:.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  📊 Train : {len(X_train):,} | Test : {len(X_test):,}")

    model = XGBClassifier(
        n_estimators         = 300,
        max_depth            = 4,
        learning_rate        = 0.05,
        subsample            = 0.8,
        colsample_bytree     = 0.8,
        scale_pos_weight     = ratio,
        eval_metric          = 'auc',
        early_stopping_rounds= 25,
        random_state         = 42,
        verbosity            = 0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)

    print(f"  🛑 Early stopping : meilleure itération = {model.best_iteration}")
    return model, X_train, X_test, y_train, y_test

# ─── 4. ÉVALUATION ────────────────────────────────────────────────────────────
def evaluate_churn(model, X_train, X_test, y_train, y_test, feature_names):
    print(f"\n{'─'*55}")
    print("  ÉVALUATION DU MODÈLE CHURN")
    print(f"{'─'*55}")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_proba_train = model.predict_proba(X_train)[:, 1]

    auc_test  = roc_auc_score(y_test,  y_proba)
    auc_train = roc_auc_score(y_train, y_proba_train)
    ap_score  = average_precision_score(y_test, y_proba)

    print(f"\n  📈 AUC Train        : {auc_train:.4f}")
    print(f"  📈 AUC Test         : {auc_test:.4f}  "
          f"{'✅ Bon' if auc_test>0.75 else '⚠️ Acceptable'}")
    print(f"  📈 Average Precision: {ap_score:.4f}")
    print(f"  📉 Écart Train/Test : {auc_train-auc_test:.4f} "
          f"{'✅ Stable' if auc_train-auc_test<0.05 else '⚠️ Surapprentissage léger'}")

    # Validation croisée
    model_cv = XGBClassifier(
        n_estimators     = model.best_iteration,
        max_depth        = 4, learning_rate=0.05,
        subsample        = 0.8, colsample_bytree=0.8,
        scale_pos_weight = (y_train==0).sum()/(y_train==1).sum(),
        random_state     = 42, verbosity=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"\n  🔁 AUC CV 5-Fold   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print(f"\n  📋 Rapport de classification :")
    print(classification_report(y_test, y_pred,
                                target_names=['Stable (0)', 'Churn (1)'],
                                digits=4))

    # Importance des features
    imps = pd.DataFrame({'Feature': feature_names,
                         'Importance': model.feature_importances_}
                        ).sort_values('Importance', ascending=False)
    print(f"  🔍 Top 7 variables prédictives :")
    for _, row in imps.head(7).iterrows():
        bar = '█' * int(row['Importance'] * 200)
        print(f"     {row['Feature']:25s} {row['Importance']:.4f}  {bar}")

    return y_proba, auc_test, ap_score, cv_scores, imps

# ─── 5. GRAPHIQUES ────────────────────────────────────────────────────────────
def plot_churn(model, X_test, y_test, y_proba, df_full,
               importances, cv_scores, auc_test, ap_score):

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('#f8fafc')
    fig.suptitle(
        "Modèle de Prédiction du Churn — Tunisie Telecom PFE\n"
        "Identification des Clients à Risque d'Attrition",
        fontsize=18, fontweight='bold', color=COLOR_MAIN, y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.42, wspace=0.38,
                           top=0.93, bottom=0.05, left=0.06, right=0.97)

    # ── G1 : Courbe ROC ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax1.plot(fpr, tpr, color=COLOR_MAIN, lw=2.5,
             label=f'AUC = {auc_test:.4f}')
    ax1.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Aléatoire')
    ax1.fill_between(fpr, tpr, alpha=0.10, color=COLOR_MAIN)
    ax1.set_title('1. Courbe ROC', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Taux Faux Positifs')
    ax1.set_ylabel('Taux Vrais Positifs')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── G2 : Precision-Recall ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ax2.plot(rec, prec, color=COLOR_CHURN, lw=2.5,
             label=f'AP = {ap_score:.4f}')
    ax2.axhline(y_test.mean(), color='gray', linestyle='--', lw=1,
                label=f'Baseline = {y_test.mean():.3f}')
    ax2.fill_between(rec, prec, alpha=0.10, color=COLOR_CHURN)
    ax2.set_title('2. Courbe Précision-Rappel', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Rappel')
    ax2.set_ylabel('Précision')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── G3 : Distribution des scores de churn ────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    scores_pos = y_proba[y_test == 1]
    scores_neg = y_proba[y_test == 0]
    ax3.hist(scores_neg, bins=40, alpha=0.65, color=COLOR_OK,
             label=f'Stable (n={len(scores_neg):,})', density=True)
    ax3.hist(scores_pos, bins=40, alpha=0.65, color=COLOR_CHURN,
             label=f'Churn (n={len(scores_pos):,})', density=True)
    ax3.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Seuil=0.5')
    ax3.set_title('3. Distribution des Scores Churn', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Score de Risque Churn')
    ax3.set_ylabel('Densité')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── G4 : Validation croisée ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
    bc = [COLOR_OK if s >= cv_scores.mean() else COLOR_CHURN for s in cv_scores]
    ax4.bar(folds, cv_scores, color=bc, edgecolor='white', width=0.6)
    ax4.axhline(cv_scores.mean(), color=COLOR_MAIN, linestyle='--', lw=2,
                label=f'Moy={cv_scores.mean():.4f}')
    for i, s in enumerate(cv_scores):
        ax4.text(i, s+0.003, f'{s:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax4.set_ylim(min(cv_scores)-0.05, 1.0)
    ax4.set_title('4. Validation Croisée 5-Fold', fontweight='bold', fontsize=11)
    ax4.set_ylabel('AUC')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # ── G5 : Importance des features ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 0:2])
    top_imp = importances.head(12)
    colors_imp = [COLOR_CHURN if i < 3 else COLOR_MAIN if i < 7 else '#93c5fd'
                  for i in range(len(top_imp))]
    bars = ax5.barh(top_imp['Feature'][::-1], top_imp['Importance'][::-1],
                    color=colors_imp[::-1], edgecolor='white')
    for bar, val in zip(bars, top_imp['Importance'][::-1]):
        ax5.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9)
    ax5.set_title('5. Importance des Variables — Prédiction Churn',
                  fontweight='bold', fontsize=11)
    ax5.set_xlabel('Importance (gain)')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.set_xlim(0, top_imp['Importance'].max()*1.2)

    # ── G6 : Taux de churn par segment RFM ───────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if 'SEGMENT_RFM' in df_full.columns:
        churn_seg = df_full.groupby('SEGMENT_RFM')['CHURN'].mean().sort_values()*100
        bc6 = [COLOR_CHURN if v >= 10 else '#f97316' if v >= 5 else COLOR_OK
               for v in churn_seg.values]
        ax6.barh(churn_seg.index, churn_seg.values, color=bc6, edgecolor='white')
        ax6.axvline(df_full['CHURN'].mean()*100, color='black',
                    linestyle='--', lw=1.5,
                    label=f'Moy={df_full["CHURN"].mean()*100:.1f}%')
        for i, v in enumerate(churn_seg.values):
            ax6.text(v+0.2, i, f'{v:.1f}%', va='center', fontsize=9)
        ax6.set_title('6. Taux de Churn par Segment RFM',
                      fontweight='bold', fontsize=11)
        ax6.set_xlabel('% Clients à Risque')
        ax6.legend(fontsize=9)

    # ── G7 : Churn par Handset ────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[1, 3])
    if 'HANDSET' in df_full.columns:
        churn_hs = df_full[df_full['HANDSET']!='Inconnu'].groupby(
            'HANDSET')['CHURN'].mean().sort_values(ascending=False)*100
        bc7 = [COLOR_CHURN if v >= 7 else '#f97316' if v >= 5 else COLOR_OK
               for v in churn_hs.values]
        ax7.bar(churn_hs.index, churn_hs.values, color=bc7, edgecolor='white', width=0.6)
        for i, v in enumerate(churn_hs.values):
            ax7.text(i, v+0.2, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
        ax7.axhline(df_full['CHURN'].mean()*100, color='black',
                    linestyle='--', lw=1.5)
        ax7.set_title('7. Taux de Churn par Type Réseau',
                      fontweight='bold', fontsize=11)
        ax7.set_ylabel('% Clients à Risque')
        ax7.set_ylim(0, churn_hs.max()*1.3)

    os.makedirs('outputs', exist_ok=True)
    plt.savefig(REPORT_PNG, dpi=150, bbox_inches='tight', facecolor='#f8fafc')
    plt.close()
    print(f"  ✅ Rapport Churn sauvegardé : {REPORT_PNG}")

# ─── 6. SCORING + EXPORT ──────────────────────────────────────────────────────
def score_and_export(model, X, df_full, all_features, encoders):
    X_all = df_full[[f for f in all_features if f in df_full.columns]].copy()
    for col, le in encoders.items():
        if col in df_full.columns:
            known = set(le.classes_)
            X_all[col+'_ENC'] = df_full[col].apply(
                lambda x: x if x in known else le.classes_[0])
            X_all[col+'_ENC'] = le.transform(X_all[col+'_ENC'])
            if col in X_all.columns: X_all = X_all.drop(columns=[col])
    X_all = X_all.apply(pd.to_numeric, errors='coerce').fillna(0)
    for f in all_features:
        if f not in X_all.columns: X_all[f] = 0
    X_all = X_all[all_features]

    df_out = df_full[['ID','ANC_M','HANDSET','OFFRE_CAT',
                       'RECENCY','FREQUENCY','MONETARY',
                       'RFM_SCORE','SEGMENT_RFM','CHURN']].copy()
    df_out['SCORE_CHURN'] = model.predict_proba(X_all)[:, 1]

    # Niveau de risque
    df_out['RISQUE_CHURN'] = pd.cut(
        df_out['SCORE_CHURN'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['🟢 Faible', '🟡 Moyen', '🔴 Élevé'],
        include_lowest=True)

    df_out['ACTION_RETENTION'] = np.select(
        [df_out['SCORE_CHURN'] >= 0.6,
         df_out['SCORE_CHURN'] >= 0.3],
        ['Appel proactif + offre fidélisation urgente',
         'Campagne email retention + bonus recharge'],
        default='Suivi standard — pas d\'action immédiate')

    df_out.sort_values('SCORE_CHURN', ascending=False
                       ).to_csv(SCORING_CSV, index=False, encoding='utf-8-sig')

    print(f"\n  ✅ Scoring churn exporté : {SCORING_CSV}")
    print(f"\n  ── Répartition par niveau de risque ─────────────────────")
    for niveau in ['🔴 Élevé', '🟡 Moyen', '🟢 Faible']:
        n = (df_out['RISQUE_CHURN']==niveau).sum()
        print(f"     {niveau:<12} : {n:>6,} clients ({n/len(df_out)*100:.1f}%)")

    # Sauvegarder le modèle
    artifact = {
        'model'        : model,
        'encoders'     : encoders,
        'feature_names': all_features,
        'metrics'      : {'type': 'churn'},
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n  💾 Modèle Churn sauvegardé : {MODEL_PATH}")

# ─── PIPELINE ─────────────────────────────────────────────────────────────────
def run_churn():
    df = load_churn_data(DATA_PATH, RFM_CSV)
    X, y, all_features, encoders, df_full = prepare_features(df)
    model, X_train, X_test, y_train, y_test = train_churn_model(X, y)
    y_proba, auc, ap, cv_scores, imps = evaluate_churn(
        model, X_train, X_test, y_train, y_test, all_features)
    plot_churn(model, X_test, y_test, y_proba, df_full,
               imps, cv_scores, auc, ap)
    score_and_export(model, X, df_full, all_features, encoders)

    print(f"\n{'='*55}")
    print("  ✅ MODÈLE CHURN TERMINÉ")
    print(f"{'='*55}")
    print(f"  → {MODEL_PATH}")
    print(f"  → {REPORT_PNG}")
    print(f"  → {SCORING_CSV}")
    print(f"  → Prochaine étape : ajouter les pages RFM et Churn")
    print(f"    au dashboard dashboard_tt_final.py\n")

if __name__ == "__main__":
    run_churn()