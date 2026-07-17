"""
=============================================================
  scoring_cv.py — Tunisie Telecom PFE
  Scoring de masse avec les 2 modèles CV

  Modèle 1 : model_forfait_cv.joblib  → score activation DATA
  Modèle 2 : model_churn_cv.joblib    → score risque churn
  Entrée   : CIBLE_ECH_DEC_2025_6.xlsx
  Sortie   : outputs/scoring_complet.csv
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ────────────────────────────────────────
import glob as _glob

def _find_cible():
    for pat in ["data/CIBLE*.xlsx","data/cible*.xlsx",
                "data/ECH*.xlsx","data/ech*.xlsx"]:
        f = _glob.glob(pat)
        if f: return f[0]
    all_x = _glob.glob("data/*.xlsx")
    return all_x[0] if all_x else None

DATA_FILE = _find_cible() or "data/CIBLE_ECH_DEC_2025_6.xlsx"
MODEL_FORFAIT  = "model_forfait_cv.joblib"
MODEL_CHURN    = "model_churn_cv.joblib"
OUTPUT_DIR     = "outputs/"
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "scoring_complet.csv")

# ─── 1. CHARGEMENT ────────────────────────────────────────
def load_models_and_data():
    print(f"\n{'='*55}")
    print("  SCORING — Tunisie Telecom PFE")
    print(f"{'='*55}")

    # Vérifier les fichiers
    if DATA_FILE is None or not os.path.exists(DATA_FILE):
        import glob
        all_x = glob.glob("data/*.xlsx")
        raise FileNotFoundError(
            f"Aucun fichier CIBLE trouvé dans data/\n"
            f"Fichiers disponibles : {all_x}\n"
            "Placez CIBLE_ECH_DEC_2025_6.xlsx dans le dossier data/"
        )
    for f in [MODEL_FORFAIT, MODEL_CHURN]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Modèle introuvable : {f}\n"
                "Lancez d'abord : python train_models_cv.py"
            )

    # Charger les modèles
    art_forfait = joblib.load(MODEL_FORFAIT)
    art_churn   = joblib.load(MODEL_CHURN)
    print(f"  Modèle Forfait chargé — AUC CV : "
          f"{art_forfait['metrics']['auc_cv_mean']:.4f}")
    print(f"  Modèle Churn   chargé — AUC CV : "
          f"{art_churn['metrics']['auc_cv_mean']:.4f}")

    # Charger les données
    df = pd.read_excel(DATA_FILE)
    df.columns = df.columns.str.strip().str.upper()
    print(f"  Données chargées : {len(df):,} clients")

    return art_forfait, art_churn, df

# ─── 2. PRÉPARATION FEATURES ──────────────────────────────
def prepare_features(df, encoders, features):
    df = df.copy()
    df['HANDSET']      = df['HANDSET'].fillna('Inconnu').astype(str)
    df['CLASSE_CANAL'] = df['CLASSE_CANAL'].fillna('Inconnu').astype(str)
    df['CODE_REGION']  = df['CODE_REGION'].fillna(df['CODE_REGION'].median())

    cat_cols = ['HANDSET', 'STATUT', 'CLASSE_CANAL']
    for col in cat_cols:
        le    = encoders[col]
        known = set(le.classes_)
        df[col + '_ENC'] = df[col].astype(str).apply(
            lambda x: x if x in known else le.classes_[0]
        )
        df[col + '_ENC'] = le.transform(df[col + '_ENC'])

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    return X

# ─── 3. SCORING ───────────────────────────────────────────
def score(art_forfait, art_churn, df):
    print(f"\n{'─'*55}")
    print("  CALCUL DES SCORES")
    print(f"{'─'*55}")

    # Préparer features (identiques pour les 2 modèles)
    X = prepare_features(df,
                         art_forfait['encoders'],
                         art_forfait['feature_names'])

    # Scores Forfait DATA
    df['SCORE_FORFAIT'] = art_forfait['model'].predict_proba(X)[:, 1]
    df['PRED_FORFAIT']  = art_forfait['model'].predict(X)

    # Scores Churn
    df['SCORE_CHURN']   = art_churn['model'].predict_proba(X)[:, 1]
    df['PRED_CHURN']    = art_churn['model'].predict(X)

    # Segments Forfait
    df['SEGMENT_FORFAIT'] = pd.cut(
        df['SCORE_FORFAIT'],
        bins=[0, 0.40, 0.70, 1.0],
        labels=['Cible Voix', 'Cible Potentielle', 'Cible Prioritaire'],
        include_lowest=True
    )

    # Segments Churn (basés sur percentiles)
    p75 = df['SCORE_CHURN'].quantile(0.75)
    p50 = df['SCORE_CHURN'].quantile(0.50)
    df['RISQUE_CHURN'] = np.where(
        df['SCORE_CHURN'] >= p75, 'Risque Eleve',
        np.where(df['SCORE_CHURN'] >= p50, 'Risque Moyen', 'Risque Faible')
    )

    # Action recommandée combinée
    def action(row):
        if row['RISQUE_CHURN'] == 'Risque Eleve':
            return 'RETENTION URGENTE — appel proactif immédiat'
        elif row['SEGMENT_FORFAIT'] == 'Cible Prioritaire':
            return 'ACTIVATION DATA — offre forfait illimité'
        elif row['SEGMENT_FORFAIT'] == 'Cible Potentielle':
            return 'NURTURING — pack Combo Voix+Data'
        elif row['RISQUE_CHURN'] == 'Risque Moyen':
            return 'RETENTION — bonus recharge fidélisation'
        else:
            return 'STANDARD — pas d action immédiate'

    df['ACTION_RECOMMANDEE'] = df.apply(action, axis=1)

    # Rapport console
    print(f"\n  Segments Forfait DATA :")
    for seg in ['Cible Prioritaire', 'Cible Potentielle', 'Cible Voix']:
        n = (df['SEGMENT_FORFAIT'] == seg).sum()
        print(f"    {seg:<22} : {n:>6,} clients "
              f"({n/len(df)*100:.1f}%)")

    print(f"\n  Segments Churn :")
    for seg in ['Risque Eleve', 'Risque Moyen', 'Risque Faible']:
        n = (df['RISQUE_CHURN'] == seg).sum()
        print(f"    {seg:<22} : {n:>6,} clients "
              f"({n/len(df)*100:.1f}%)")

    return df

# ─── 4. EXPORT ────────────────────────────────────────────
def export(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cols = [c for c in [
        'ID', 'ANC_M', 'HANDSET', 'STATUT', 'CLASSE_CANAL',
        'TARGET', 'FLAG_CHURN',
        'SCORE_FORFAIT', 'PRED_FORFAIT', 'SEGMENT_FORFAIT',
        'SCORE_CHURN',   'PRED_CHURN',   'RISQUE_CHURN',
        'ACTION_RECOMMANDEE'
    ] if c in df.columns]

    df_out = df[cols].sort_values('SCORE_FORFAIT', ascending=False)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n  Fichier exporté : {OUTPUT_FILE}")
    print(f"  {len(df_out):,} clients scorés")

    # Top 10
    print(f"\n  Top 10 Cibles Prioritaires (Score Forfait) :")
    top = df_out.head(10)[
        ['ID', 'HANDSET', 'SCORE_FORFAIT', 'RISQUE_CHURN', 'ACTION_RECOMMANDEE']
    ].copy()
    top['SCORE_FORFAIT'] = (top['SCORE_FORFAIT']*100).round(1).astype(str)+'%'
    print(top.to_string(index=False))

# ─── PIPELINE ─────────────────────────────────────────────
def run():
    art_forfait, art_churn, df = load_models_and_data()
    df_scored = score(art_forfait, art_churn, df)
    export(df_scored)
    print(f"\n{'='*55}")
    print("  SCORING TERMINE AVEC SUCCES")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    run()