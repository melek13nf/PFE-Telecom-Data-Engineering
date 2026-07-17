import pandas as pd
import numpy as np
import os

def run_full_etl():
    print("🚀 Initialisation de l'ETL (Extract - Transform - Load)...")

    # --- 1. EXTRACTION (Gestion des fichiers manquants) ---
    files = {
        'clients': 'clients_brut.xlsx',
        'recharges': 'recharges_brut.xlsx',
        'appels': 'appels_brut.xlsx'
    }

    # Vérification si les fichiers existent, sinon on crée des données de test
    if not os.path.exists(files['clients']):
        print("⚠️ Fichiers sources introuvables. Génération de données de simulation...")
        df_clients = pd.DataFrame({
            'Client_Key': [101, 102, 103, 104],
            'HANDSET': ['Samsung', np.nan, 'iPhone', 'Nokia'], # Un NULL ici
            'Region_Key': [1, 2, np.nan, 1]                    # Un NULL ici
        })
        df_recharges = pd.DataFrame({
            'Client_Key': [101, 101, 103],
            'MNT_RECH': [10, 5, 20]
        })
        df_appels = pd.DataFrame({
            'Client_Key': [101, 102],
            'DUREE_SEC': [120, 45],
            'NB_SMS': [5, 0]
        })
    else:
        print("📁 Lecture des fichiers Excel locaux...")
        df_clients = pd.read_excel(files['clients'])
        df_recharges = pd.read_excel(files['recharges'])
        df_appels = pd.read_excel(files['appels'])

    # --- 2. TRANSFORMATION (Nettoyage des NULL & Agrégation) ---
    print("🔧 Phase de Transformation (Nettoyage des données)...")

    # A. Nettoyage Dimension Client (Imputation)
    df_clients['HANDSET'] = df_clients['HANDSET'].fillna('Smartphone_Standard')
    df_clients['Region_Key'] = df_clients['Region_Key'].fillna(0).astype(int)

    # B. Agrégation des Recharges (Fact_Recharge)
    # On transforme les transactions multiples en un total par client
    fact_recharge = df_recharges.groupby('Client_Key')['MNT_RECH'].sum().reset_index()
    fact_recharge.rename(columns={'MNT_RECH': 'Total_Mnt_Recharge'}, inplace=True)

    # C. Transformation des Appels (Fact_Sortant)
    # Conversion secondes -> minutes
    df_appels['DUREE_MIN'] = df_appels['DUREE_SEC'] / 60
    fact_sortant = df_appels.groupby('Client_Key').agg({
        'DUREE_MIN': 'sum',
        'NB_SMS': 'sum'
    }).reset_index()

    # D. Fusion (JOIN) pour créer l'Analytical Base Table (ABT)
    # On utilise 'left' pour ne perdre aucun client
    df_final = pd.merge(df_clients, fact_recharge, on='Client_Key', how='left')
    df_final = pd.merge(df_final, fact_sortant, on='Client_Key', how='left')

    # E. Traitement des NULL après fusion (les clients sans activité)
    # C'est ici qu'on règle ton problème de NULL final
    cols_to_fix = ['Total_Mnt_Recharge', 'DUREE_MIN', 'NB_SMS']
    df_final[cols_to_fix] = df_final[cols_to_fix].fillna(0)

    # F. Feature Engineering (La Cible IA)
    # 1 si Petit Consommateur (Risque de Churn), 0 sinon
    df_final['Target_Churn'] = np.where(df_final['Total_Mnt_Recharge'] < 7, 1, 0)

    # --- 3. CHARGEMENT (Load) ---
    output_file = 'dw_telecom_cleaned.xlsx'
    df_final.to_excel(output_file, index=False)
    
    print(f"✅ ETL Terminé avec succès !")
    print(f"📊 Fichier généré : {output_file}")
    print(df_final.head())

if __name__ == "__main__":
    run_full_etl()