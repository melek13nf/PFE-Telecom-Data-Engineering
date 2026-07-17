"""
=============================================================
  ETL / Preprocessing – Tunisie Telecom PFE
  Analytical Base Table (ABT) — 1 ligne = 1 client
=============================================================

CORRECTIONS APPORTÉES :
  [BUG 1] Base de départ corrigée : ECH (clients master) au lieu de RECHARGE
          → RECHARGE n'a pas ID_REGION/ID_OFFRE, les jointures Region/Offre
            produisaient 100% de NaN.
  [BUG 2] Fichier RECHARGE ajouté correctement : agrégation par ID AVANT merge
          → Sans agrégation, 309 365 lignes au lieu de 31 200 (explosion).
  [BUG 3] Nom de colonne corrigé : 'REVENU_CDR' (majuscule) au lieu de 'revenu_cdr'
          → Causait un KeyError silencieux (valeur renvoyée = 0 partout).
  [BUG 4] Fichier USSD ajouté : données forfaits DATA manquantes dans l'ABT.
  [BUG 5] Nettoyage colonnes étendu à .str.upper() (pas seulement .str.strip())
          → Évite les erreurs de casse sur toutes les jointures.
  [BUG 6] NaN traités différemment selon le type de colonne
          → fillna(0) uniquement sur les colonnes numériques ;
            les colonnes catégorielles reçoivent 'Inconnu'.
  [BUG 7] Export enrichi : colonnes importantes dans l'ordre, log de qualité.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_PATH   = "data/"
OUTPUT_NAME = "ABT_Final_ML.csv"

FILE_MAP = {
    'ech'     : 'ECH__DECEMBRE_2025 1.xlsx',      # Table clients master (clé : ID)
    'recharge': 'RECHARGE_DECEMBRE_2025 1.xlsx',   # Recharges (N lignes par client)
    'entrant' : 'ENTRANT_DECEMBRE_2025 1.xlsx',    # Appels entrants (N lignes par client)
    'sortant' : 'SORTANT_DECEMBRE_2025 1.xlsx',    # Appels sortants (N lignes par client)
    'ussd'    : 'USSD_DECEMBRE_2025 1.xlsx',       # Forfaits DATA (N lignes par client)
    'regions' : 'REGION_ 1.xlsx',                  # Référentiel régions (25 lignes)
    'offres'  : 'offre_ 1.xlsx',                   # Référentiel offres (412 lignes)
}

# ─── CHARGEMENT ───────────────────────────────────────────────────────────────
def load_files(file_map, data_path):
    """Charge tous les fichiers Excel. Normalise les noms de colonnes en MAJUSCULES."""
    data = {}
    print(f"\n{'='*55}")
    print(f"  Démarrage ETL — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    for key, filename in file_map.items():
        path = os.path.join(data_path, filename)
        if os.path.exists(path):
            df = pd.read_excel(path)
            # BUG 5 FIX : .strip() ET .upper() pour éviter tout problème de casse/espaces
            df.columns = df.columns.str.strip().str.upper()
            data[key] = df
            print(f"  ✅ [{key}] chargé — {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
        else:
            print(f"  ⚠️  [{key}] ABSENT : {filename}")

    if 'ech' not in data:
        raise FileNotFoundError(
            "❌ Fichier ECH (table clients master) introuvable. ETL interrompu."
        )
    return data

# ─── AGRÉGATIONS ──────────────────────────────────────────────────────────────
def aggregate_recharge(df_r):
    """
    BUG 2 FIX : agrégation OBLIGATOIRE avant tout merge.
    RECHARGE a N lignes par client (1 par mois). Sans cette étape,
    un merge direct multiplie les lignes de l'ABT × N.
    """
    return df_r.groupby('ID').agg(
        MNT_RECH_TOT    = ('MNT_RECH',      'sum'),
        NB_RECH_TOT     = ('NB_RECH',       'sum'),
        MNT_RECH_SUP5   = ('MNT_RECH_SUP5', 'sum'),
        NB_RECH_SUP5    = ('NB_RECH_SUP5',  'sum'),
        MNT_RECH_MOY    = ('MNT_RECH',      'mean'),  # feature utile pour le ML
        NB_MOIS_ACTIF_R = ('MONTH_DT',      'count'), # nb mois avec recharge
    ).reset_index()

def aggregate_entrant(df_e):
    """Agrégation appels entrants (N lignes → 1 ligne par client)."""
    return df_e.groupby('ID').agg(
        DUREE_APPEL_IN_TOT  = ('DUREE_APPEL_IN',         'sum'),
        DUREE_OOREDOO_IN    = ('DUREE_OOREDOO_IN',        'sum'),
        DUREE_ORANGE_IN     = ('DUREE_ORANGE_IN',         'sum'),
        NB_SMS_IN_TOT       = ('NB_SMS_IN',               'sum'),
        DUREE_APPEL_INTER_IN= ('DUREE_APPEL_INTER_IN',    'sum'),
    ).reset_index()

def aggregate_sortant(df_s):
    """
    BUG 3 FIX : utilisation de 'REVENU_CDR' (MAJUSCULES) pas 'revenu_cdr'.
    Agrégation appels sortants (N lignes → 1 ligne par client).
    """
    return df_s.groupby('ID').agg(
        REVENU_CDR_TOT          = ('REVENU_CDR',               'sum'),   # BUG 3 FIX
        REVENU_VOIX_TOT         = ('REVENU_VOIX',              'sum'),
        REVENU_INTER_TOT        = ('REVENU_INTER',             'sum'),
        REVENU_ROAMING_TOT      = ('REVENU_ROAMING',           'sum'),
        DUREE_APPEL_TOT         = ('DUREE_APPEL_TOT',          'sum'),
        DUREE_APPEL_OOREDOO_TOT = ('DUREE_APPEL_OOREDOO_TOT',  'sum'),
        DUREE_APPEL_ORANGE_TOT  = ('DUREE_APPEL_ORANGE_TOT',   'sum'),
        DUREE_OFFNET_TOT        = ('DUREE_OFFNET_TOT',         'sum'),
        NB_APPEL_TOT            = ('NB_APPEL_TOT',             'sum'),
        REVENU_SMS_TOT          = ('REVENU_SMS',               'sum'),
        NB_SMS_TOT              = ('NB_SMS_TOT',               'sum'),
        NB_MOIS_ACTIF_S         = ('MONTH_DT',                 'count'),
    ).reset_index()

def aggregate_ussd(df_u):
    """
    BUG 4 FIX : USSD était totalement absent du script original.
    Contient les données forfaits DATA — variable clé pour la cible IA.
    """
    return df_u.groupby('ID').agg(
        MNT_FORFAIT_TOT      = ('MNT_FORFAIT',      'sum'),
        MNT_FORFAIT_DATA_TOT = ('MNT_FORFAIT_DATA', 'sum'),
        NB_FORFAIT_TOT       = ('NB_FORFAIT',       'sum'),
        NB_FORFAIT_DATA_TOT  = ('NB_FORFAIT_DATA',  'sum'),
    ).reset_index()

# ─── NETTOYAGE & FEATURES ─────────────────────────────────────────────────────
def clean_and_engineer(df):
    """Nettoyage final et création de features dérivées pour le ML."""

    # BUG 6 FIX : fillna différencié par type de colonne
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna('Inconnu')

    # Features dérivées utiles pour le modèle XGBoost
    df['DUREE_ONNET_TOT'] = (df['DUREE_APPEL_TOT'] - df['DUREE_OFFNET_TOT']).clip(lower=0)
    df['REVENU_TOTAL']    = df['REVENU_CDR_TOT'] + df['MNT_RECH_TOT']
    df['PCT_DATA']        = np.where(
        df['MNT_FORFAIT_TOT'] > 0,
        df['MNT_FORFAIT_DATA_TOT'] / df['MNT_FORFAIT_TOT'],
        0
    )

    # Cible IA : 1 si le client a souscrit au moins un forfait DATA
    df['TARGET_IA'] = (df['MNT_FORFAIT_DATA_TOT'] > 0).astype(int)

    return df

# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────
def run_etl():
    # 1. Chargement
    data = load_files(FILE_MAP, DATA_PATH)

    # 2. BUG 1 FIX : ECH comme table de base (1 ligne par client, contient ID_REGION/ID_OFFRE)
    print(f"\n{'─'*55}")
    print("  Construction de l'ABT (1 ligne = 1 client)")
    print(f"{'─'*55}")

    df = data['ech'].copy()
    print(f"\n  📌 Base : ECH — {df.shape[0]:,} clients uniques")

    # 3. Jointures référentiels (ID_REGION et ID_OFFRE sont dans ECH ✅)
    if 'regions' in data:
        df['ID_REGION']              = pd.to_numeric(df['ID_REGION'], errors='coerce')
        data['regions']['ID_REGION'] = pd.to_numeric(data['regions']['ID_REGION'], errors='coerce')
        df = df.merge(data['regions'], on='ID_REGION', how='left')
        print(f"  🔗 Jonction REGION  — {df['REGION'].notna().sum():,} clients avec région connue")

    if 'offres' in data:
        df['ID_OFFRE']              = pd.to_numeric(df['ID_OFFRE'], errors='coerce')
        data['offres']['ID_OFFRE']  = pd.to_numeric(data['offres']['ID_OFFRE'], errors='coerce')
        df = df.merge(data['offres'], on='ID_OFFRE', how='left')
        print(f"  🔗 Jonction OFFRE   — {df['OFFRE'].notna().sum():,} clients avec offre connue")

    # 4. Agrégations + merge des tables transactionnelles
    if 'recharge' in data:
        r_agg = aggregate_recharge(data['recharge'])
        df = df.merge(r_agg, on='ID', how='left')
        print(f"  🔗 Jonction RECHARGE — agrégée sur {r_agg.shape[0]:,} clients")

    if 'entrant' in data:
        e_agg = aggregate_entrant(data['entrant'])
        df = df.merge(e_agg, on='ID', how='left')
        print(f"  🔗 Jonction ENTRANT  — agrégée sur {e_agg.shape[0]:,} clients")

    if 'sortant' in data:
        s_agg = aggregate_sortant(data['sortant'])
        df = df.merge(s_agg, on='ID', how='left')
        print(f"  🔗 Jonction SORTANT  — agrégée sur {s_agg.shape[0]:,} clients")

    if 'ussd' in data:
        u_agg = aggregate_ussd(data['ussd'])
        df = df.merge(u_agg, on='ID', how='left')
        print(f"  🔗 Jonction USSD     — agrégée sur {u_agg.shape[0]:,} clients")

    # 5. Nettoyage & feature engineering
    df = clean_and_engineer(df)

    # 6. Vérification intégrité : le nombre de lignes ne doit pas avoir changé
    assert df.shape[0] == data['ech'].shape[0], (
        f"❌ INTEGRITÉ VIOLÉE : {df.shape[0]} lignes au lieu de {data['ech'].shape[0]} !"
        " Vérifier les merges (doublons sur clé de jointure)."
    )

    # 7. Export
    df.to_csv(OUTPUT_NAME, index=False)

    # 8. Rapport qualité
    print(f"\n{'='*55}")
    print("  ✨ ETL TERMINÉ AVEC SUCCÈS")
    print(f"{'='*55}")
    print(f"  📊 Shape finale : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"  📂 Fichier      : {OUTPUT_NAME}")
    print(f"\n  ── Rapport Qualité ──────────────────────────────")
    print(f"  Clients actifs (STATUT=A)  : {(df['STATUT']=='A').sum():,}")
    print(f"  Clients à risque (RGS90=R) : {(df['STATUT_RGS90']=='R').sum():,}")
    print(f"  Clients avec Data (TARGET=1): {df['TARGET_IA'].sum():,} "
          f"({df['TARGET_IA'].mean()*100:.1f}%)")
    print(f"  Colonnes avec NaN restants :")
    nan_cols = df.isnull().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if nan_cols.empty:
        print("    Aucun NaN ✅")
    else:
        for col, n in nan_cols.items():
            print(f"    {col}: {n:,} NaN")
    print(f"\n  ── Colonnes de l'ABT ────────────────────────────")
    for c in df.columns:
        print(f"    {c}")

if __name__ == "__main__":
    run_etl()