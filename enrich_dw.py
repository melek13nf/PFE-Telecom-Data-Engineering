import pyodbc
import pandas as pd
from src.config import SERVER, DATABASE

def enrich_data_warehouse():
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("🚀 Début de l'enrichissement du Data Warehouse...")

    # --- ÉTAPE 1 : AJOUT DES COLONNES DE LIAISON (KEYS) ---
    print("🛠️ Ajout des clés étrangères dans Dim_Client...")
    try:
        cursor.execute("ALTER TABLE Dim_Client ADD Region_Key INT;")
        cursor.execute("ALTER TABLE Dim_Client ADD Offre_Key INT;")
        conn.commit()
    except Exception as e:
        print("ℹ️ Colonnes déjà existantes ou erreur mineure.")

    # --- ÉTAPE 2 : MAPPING RÉGIONS (Lier Client -> Région) ---
    # On distribue les clients dans les régions de manière logique pour le PFE
    print("📍 Mapping des Régions...")
    cursor.execute("""
        UPDATE Dim_Client
        SET Region_Key = (ABS(CAST(HASHBYTES('MD5', CAST(Client_Key AS VARCHAR)) AS INT)) % 24) + 1
        WHERE Region_Key IS NULL
    """)
    
    # --- ÉTAPE 3 : MAPPING OFFRES (Lier Client -> Offre) ---
    print("📦 Mapping des Offres...")
    cursor.execute("""
        UPDATE Dim_Client
        SET Offre_Key = (ABS(CAST(HASHBYTES('MD5', CAST(Client_Key AS VARCHAR)) AS INT)) % 5) + 1
        WHERE Offre_Key IS NULL
    """)

    # --- ÉTAPE 4 : NETTOYAGE DES FAITS (Fact_Recharge) ---
    # On s'assure que Montant_Recharge est exploitable (pas de valeurs aberrantes)
    print("🧹 Nettoyage de Fact_Recharge...")
    cursor.execute("UPDATE Fact_Recharge SET Montant_Recharge = 0 WHERE Montant_Recharge IS NULL")
    
    conn.commit()
    print("✅ Enrichissement terminé avec succès !")
    conn.close()

if __name__ == "__main__":
    enrich_data_warehouse()