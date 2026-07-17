"""
=============================================================
  batch_scoring.py — Tunisie Telecom PFE
  Scoring de masse : tout le parc client (31 200 clients)
  Entrée  : ABT_Final_ML.csv + model_appetence_tt.joblib
  Sortie  : 3 fichiers CSV segmentés par priorité marketing
=============================================================

CORRECTIONS PAR RAPPORT À L'ORIGINAL :
  [BUG 1] Lisait ECH brut (4 features) → lit maintenant l'ABT complète (17 features)
  [BUG 2] Ne chargeait que model, pas les encodeurs → charge l'artifact complet
  [BUG 3] Encodage catégoriel absent → applique les LabelEncoders sauvegardés
  [BUG 4] Ne scorait que 100 clients → score tout le parc (31 200 clients)
  [BUG 5] 1 seul fichier output → 3 fichiers segmentés par segment de priorité
  [BUG 6] Aucune statistique de sortie → rapport complet avec distribution des scores
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
ABT_PATH      = "ABT_Final_ML.csv"
MODEL_PATH    = "model_appetence_tt.joblib"
OUTPUT_FOLDER = "outputs/campagnes_marketing/"

# Seuils de segmentation marketing (ajustables selon la stratégie commerciale)
SEUIL_PRIORITAIRE = 0.75   # Score ≥ 75% → Cible Prioritaire  (offre Data illimitée)
SEUIL_POTENTIEL   = 0.45   # Score ≥ 45% → Cible Potentielle  (pack Combo Voix+Data)
                            # Score <  45% → Cible Voix         (bonus minutes)

# ─── 1. CHARGEMENT DU MODÈLE ──────────────────────────────────────────────────
def load_model_artifact(model_path):
    """
    Charge l'artifact complet sauvegardé par train_model_fixed.py :
    - model         : XGBClassifier entraîné
    - encoders      : LabelEncoders pour HANDSET, STATUT, OFFRE_CAT
    - feature_names : ordre exact des 17 features attendues
    - metrics       : AUC, AP, CV scores
    """
    print(f"\n{'='*60}")
    print("  CHARGEMENT DU MODÈLE")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Modèle introuvable : '{model_path}'\n"
            "   → Lancez d'abord train_model_fixed.py"
        )

    artifact = joblib.load(model_path)

    # Vérification que c'est bien le nouveau format d'artifact
    required_keys = ['model', 'encoders', 'feature_names', 'metrics']
    missing = [k for k in required_keys if k not in artifact]
    if missing:
        raise ValueError(
            f"❌ Format de modèle incompatible. Clés manquantes : {missing}\n"
            "   → Ré-entraînez avec train_model_fixed.py"
        )

    model    = artifact['model']
    encoders = artifact['encoders']
    features = artifact['feature_names']
    metrics  = artifact['metrics']

    print(f"  ✅ Modèle chargé")
    print(f"  📊 Performances enregistrées à l'entraînement :")
    print(f"     AUC Test  : {metrics['auc_test']:.4f}")
    print(f"     AP Score  : {metrics['ap_score']:.4f}")
    print(f"     AUC CV    : {metrics['auc_cv_mean']:.4f} ± {metrics['auc_cv_std']:.4f}")
    print(f"  📋 Features attendues ({len(features)}) : {features}")

    return model, encoders, features

# ─── 2. CHARGEMENT DE L'ABT ───────────────────────────────────────────────────
def load_abt(abt_path):
    """Charge l'ABT produite par preprocessing_pfe_fixed.py."""
    print(f"\n{'─'*60}")
    print("  CHARGEMENT DE L'ABT")
    print(f"{'─'*60}")

    if not os.path.exists(abt_path):
        raise FileNotFoundError(
            f"❌ ABT introuvable : '{abt_path}'\n"
            "   → Lancez d'abord preprocessing_pfe_fixed.py"
        )

    df = pd.read_csv(abt_path)
    df.columns = df.columns.str.strip().str.upper()
    print(f"  ✅ ABT chargée : {df.shape[0]:,} clients × {df.shape[1]} colonnes")

    return df

# ─── 3. PRÉPARATION DES FEATURES (identique à train_model_fixed.py) ───────────
def prepare_features_for_scoring(df, encoders, feature_names):
    """
    Applique exactement les mêmes transformations qu'à l'entraînement.
    Utilise les LabelEncoders sauvegardés pour garantir la cohérence.
    """
    print(f"\n{'─'*60}")
    print("  PRÉPARATION DES FEATURES")
    print(f"{'─'*60}")

    df = df.copy()

    # Nettoyage identique à l'entraînement
    df['MNT_RECH_TOT'] = df['MNT_RECH_TOT'].clip(lower=0)
    df['MNT_RECH_MOY'] = df['MNT_RECH_MOY'].clip(lower=0)
    df['HANDSET']   = df['HANDSET'].replace(0, 'Inconnu').astype(str)
    df['STATUT']    = df['STATUT'].astype(str)
    df['OFFRE_CAT'] = df['OFFRE_CAT'].astype(str)

    # Encodage catégoriel avec les LabelEncoders sauvegardés
    for col, le in encoders.items():
        col_enc = col + '_ENC'
        # Gérer les nouvelles modalités inconnues (transform() planterait sinon)
        known_classes = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
        df[col_enc] = le.transform(df[col])
        print(f"  🔤 {col} encodé avec LabelEncoder sauvegardé ({len(le.classes_)} modalités)")

    # Vérification des features manquantes
    missing_feats = [f for f in feature_names if f not in df.columns]
    if missing_feats:
        print(f"  ⚠️  Features manquantes (remplacement par 0) : {missing_feats}")
        for f in missing_feats:
            df[f] = 0

    # Extraction dans le bon ordre (CRITIQUE pour XGBoost)
    X = df[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"  ✅ Matrice de scoring prête : {X.shape[0]:,} lignes × {X.shape[1]} features")

    return X

# ─── 4. SCORING ───────────────────────────────────────────────────────────────
def score_parc(model, X, df):
    """Calcule les scores de propension pour tout le parc."""
    print(f"\n{'─'*60}")
    print("  SCORING DU PARC COMPLET")
    print(f"{'─'*60}")

    print(f"  🧠 Calcul des scores XGBoost sur {len(X):,} clients...")
    probas = model.predict_proba(X)[:, 1]
    df = df.copy()
    df['SCORE_APPETENCE'] = probas

    # Segmentation marketing selon les seuils
    df['SEGMENT_MARKETING'] = pd.cut(
        df['SCORE_APPETENCE'],
        bins=[0, SEUIL_POTENTIEL, SEUIL_PRIORITAIRE, 1.0],
        labels=['🧊 Cible Voix', '⚡ Cible Potentielle', '🎯 Cible Prioritaire'],
        include_lowest=True
    )

    # Action recommandée par segment
    conditions = [
        df['SCORE_APPETENCE'] >= SEUIL_PRIORITAIRE,
        df['SCORE_APPETENCE'] >= SEUIL_POTENTIEL,
    ]
    actions = [
        'Offre Forfait DATA Illimité — contact SMS/App prioritaire',
        'Pack Combo Voix+Data — offre essai 1 mois',
    ]
    df['ACTION_RECOMMANDEE'] = np.select(conditions, actions,
                                          default='Bonus Minutes — pas de push Data')

    print(f"  ✅ Scoring terminé !")
    return df

# ─── 5. RAPPORT & EXPORT ──────────────────────────────────────────────────────
def export_results(df, output_folder):
    """Génère 3 fichiers CSV segmentés + 1 fichier de synthèse."""
    print(f"\n{'─'*60}")
    print("  EXPORT DES RÉSULTATS")
    print(f"{'─'*60}")

    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Colonnes à conserver dans les exports (identité + score + contexte)
    cols_export = [c for c in [
        'ID', 'REGION', 'OFFRE', 'OFFRE_CAT', 'HANDSET', 'STATUT',
        'ANC_M', 'MNT_RECH_TOT', 'MNT_RECH_MOY', 'DUREE_APPEL_TOT',
        'NB_SMS_TOT', 'MNT_FORFAIT_DATA_TOT', 'NB_FORFAIT_DATA_TOT',
        'REVENU_CDR_TOT', 'TARGET_IA',
        'SCORE_APPETENCE', 'SEGMENT_MARKETING', 'ACTION_RECOMMANDEE'
    ] if c in df.columns]

    df_export = df[cols_export].sort_values('SCORE_APPETENCE', ascending=False)

    # ── Fichier 1 : Tout le parc scoré ────────────────────────────────────
    path_all = os.path.join(output_folder, f'SCORING_PARC_COMPLET_{timestamp}.csv')
    df_export.to_csv(path_all, index=False, encoding='utf-8-sig')
    print(f"\n  📂 [1/3] Parc complet  : {len(df_export):,} clients → {path_all}")

    # ── Fichier 2 : Cibles prioritaires (score ≥ seuil_prioritaire) ────────
    cibles_prio = df_export[df_export['SCORE_APPETENCE'] >= SEUIL_PRIORITAIRE]
    path_prio = os.path.join(output_folder, f'CIBLES_PRIORITAIRES_{timestamp}.csv')
    cibles_prio.to_csv(path_prio, index=False, encoding='utf-8-sig')
    print(f"  📂 [2/3] Prioritaires  : {len(cibles_prio):,} clients "
          f"(score ≥ {SEUIL_PRIORITAIRE:.0%}) → {path_prio}")

    # ── Fichier 3 : Cibles potentielles ────────────────────────────────────
    cibles_pot = df_export[
        (df_export['SCORE_APPETENCE'] >= SEUIL_POTENTIEL) &
        (df_export['SCORE_APPETENCE'] <  SEUIL_PRIORITAIRE)
    ]
    path_pot = os.path.join(output_folder, f'CIBLES_POTENTIELLES_{timestamp}.csv')
    cibles_pot.to_csv(path_pot, index=False, encoding='utf-8-sig')
    print(f"  📂 [3/3] Potentielles  : {len(cibles_pot):,} clients "
          f"(score {SEUIL_POTENTIEL:.0%}–{SEUIL_PRIORITAIRE:.0%}) → {path_pot}")

    return df_export, cibles_prio, cibles_pot

# ─── 6. RAPPORT CONSOLE ───────────────────────────────────────────────────────
def print_report(df, cibles_prio, cibles_pot):
    """Affiche un rapport complet de la campagne dans le terminal."""
    print(f"\n{'='*60}")
    print("  RAPPORT DE CAMPAGNE MARKETING")
    print(f"{'='*60}")

    total = len(df)
    n_prio = len(cibles_prio)
    n_pot  = len(cibles_pot)
    n_voix = total - n_prio - n_pot

    print(f"\n  📊 Distribution des scores :")
    print(f"     Score moyen    : {df['SCORE_APPETENCE'].mean():.4f}")
    print(f"     Score médian   : {df['SCORE_APPETENCE'].median():.4f}")
    print(f"     Score max      : {df['SCORE_APPETENCE'].max():.4f}")
    print(f"     Score min      : {df['SCORE_APPETENCE'].min():.4f}")

    print(f"\n  🎯 Segmentation Marketing ({total:,} clients) :")
    print(f"  {'─'*50}")
    print(f"  {'Segment':<25} {'Nb Clients':>10}  {'% Parc':>8}  {'Score Moy':>10}")
    print(f"  {'─'*50}")
    print(f"  {'🎯 Cible Prioritaire':<25} {n_prio:>10,}  "
          f"{n_prio/total*100:>7.1f}%  "
          f"{cibles_prio['SCORE_APPETENCE'].mean():>10.4f}")
    print(f"  {'⚡ Cible Potentielle':<25} {n_pot:>10,}  "
          f"{n_pot/total*100:>7.1f}%  "
          f"{cibles_pot['SCORE_APPETENCE'].mean():>10.4f}")
    voix_df = df[df['SCORE_APPETENCE'] < SEUIL_POTENTIEL]
    print(f"  {'🧊 Cible Voix':<25} {n_voix:>10,}  "
          f"{n_voix/total*100:>7.1f}%  "
          f"{voix_df['SCORE_APPETENCE'].mean():>10.4f}")
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':<25} {total:>10,}  {'100.0%':>8}")

    # Top 10 clients prioritaires
    print(f"\n  🏆 Top 10 clients à contacter en priorité :")
    print(f"  {'─'*75}")
    cols_top = [c for c in ['ID','REGION','HANDSET','OFFRE_CAT','ANC_M',
                             'MNT_RECH_TOT','SCORE_APPETENCE'] if c in df.columns]
    top10 = df.nlargest(10, 'SCORE_APPETENCE')[cols_top]
    top10['SCORE_APPETENCE'] = (top10['SCORE_APPETENCE']*100).round(2).astype(str) + '%'
    print(top10.to_string(index=False))

    # Analyse par région (si disponible)
    if 'REGION' in df.columns:
        print(f"\n  📍 Top 5 Régions – Score moyen :")
        reg = df.groupby('REGION')['SCORE_APPETENCE'].mean().sort_values(ascending=False).head(5)
        for region, score in reg.items():
            bar = '█' * int(score * 30)
            print(f"    {region:<20} {score:.4f}  {bar}")

    # Analyse par Handset
    if 'HANDSET' in df.columns:
        print(f"\n  📱 Score moyen par type réseau :")
        hs = df.groupby('HANDSET')['SCORE_APPETENCE'].mean().sort_values(ascending=False)
        for handset, score in hs.items():
            bar = '█' * int(score * 30)
            print(f"    {str(handset):<10} {score:.4f}  {bar}")

# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────
def run_batch_scoring():
    print(f"\n{'='*60}")
    print(f"  SCORING DE MASSE — Tunisie Telecom PFE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 1. Charger le modèle et ses encodeurs
    model, encoders, feature_names = load_model_artifact(MODEL_PATH)

    # 2. Charger l'ABT
    df = load_abt(ABT_PATH)

    # 3. Préparer les features avec les mêmes encodeurs qu'à l'entraînement
    X = prepare_features_for_scoring(df, encoders, feature_names)

    # 4. Scorer tout le parc
    df_scored = score_parc(model, X, df)

    # 5. Exporter les résultats
    df_export, cibles_prio, cibles_pot = export_results(df_scored, OUTPUT_FOLDER)

    # 6. Afficher le rapport complet
    print_report(df_export, cibles_prio, cibles_pot)

    print(f"\n{'='*60}")
    print("  ✅ BATCH SCORING TERMINÉ AVEC SUCCÈS")
    print(f"{'='*60}")
    print(f"  → Prochaine étape : connecter le dashboard au modèle .joblib")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_batch_scoring()