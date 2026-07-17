import pandas as pd
import os
import pyodbc
from src.config import SERVER, DATABASE

def load_all_facts_enriched():
    print("🚀 Chargement enrichi des tables de faits...")
    base_path = os.path.join(os.getcwd(), "data")
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # --- 1. PRÉPARATION DU LOOKUP CLIENTS ---
        cursor.execute("SELECT Client_Key, Client_ID FROM Dim_Client")
        client_lookup = {str(row[1]): row[0] for row in cursor.fetchall()}

        def process_fact(file_name, table_name, mapping):
            print(f"\n⏳ Traitement de {table_name}...")
            path = os.path.join(base_path, file_name)
            if not os.path.exists(path):
                print(f"   ⚠️ Fichier {file_name} manquant. Étape sautée.")
                return

            df = pd.read_excel(path)
            df.columns = df.columns.str.strip().str.upper()
            
            # --- DÉTECTION DYNAMIQUE DES COLONNES SQL EXISTANTES ---
            cursor.execute(f"SELECT TOP 0 * FROM {table_name}")
            vraies_cols_sql = [column[0] for column in cursor.description]
            
            cursor.execute(f"DELETE FROM {table_name}")
            
            # On construit dynamiquement les colonnes à insérer
            cols_finales = ["Client_Key", "Time_Key"]
            cols_excel_sources = []
            
            for col_sql, col_excel in mapping.items():
                if col_sql in vraies_cols_sql:
                    cols_finales.append(col_sql)
                    cols_excel_sources.append(col_excel)
                else:
                    # Test de l'alternative In/Out si erreur de nommage dans SQL
                    alt = col_sql.replace('_OUT', '_IN') if '_OUT' in col_sql.upper() else col_sql.replace('_IN', '_OUT')
                    if alt in vraies_cols_sql:
                        cols_finales.append(alt)
                        cols_excel_sources.append(col_excel)
            
            count = 0
            for _, row in df.iterrows():
                client_key = client_lookup.get(str(row.get('ID')))
                try:
                    time_key = int(pd.to_datetime(row.get('MONTH_DT')).strftime('%Y%m%d'))
                except:
                    time_key = 20251201

                if client_key:
                    # On prépare les valeurs (0 par défaut si Excel vide)
                    vals = [client_key, time_key] + [row.get(src, 0) for src in cols_excel_sources]
                    placeholders = ", ".join(["?"] * len(vals))
                    query = f"INSERT INTO {table_name} ({', '.join(cols_finales)}) VALUES ({placeholders})"
                    cursor.execute(query, *vals)
                    count += 1
            
            conn.commit()
            print(f"   ✅ {table_name} terminé : {count} lignes insérées.")

        # --- 2. EXÉCUTION DES MAPPINGS ---
        
        # Table Entrant
        process_fact("ENTRANT_DECEMBRE_2025 1.xlsx", "Fact_Entrant", {
            "Duree_Appel_In": "DUREE_APPE",
            "NB_SMS_In": "NB_SMS_IN",
            "Duree_OnNet_In": "DUREE_TT_GSM_IN",
            "Duree_OffNet_In": "DUREE_OOREDOO_IN"
        })

        # Table Sortant
        process_fact("SORTANT_DECEMBRE_2025 1.xlsx", "Fact_Sortant", {
            "Duree_Appel_Out": "DUREE_APPE",
            "NB_SMS_Out": "NB_SMS_TOT",
            "Revenu_Voix": "REVENU_VOIX",
            "Revenu_SMS": "REVENU_SMS",
            "Revenu_Data": "REVENU_DATA"
        })

        # Table Recharge
        process_fact("RECHARGE_DECEMBRE_2025 1.xlsx", "Fact_Recharge", {
            "Montant_Recharge": "MONTANT_RECH",
            "Nb_Recharges": "NB_RECH"
        })

        # Table USSD
        process_fact("USSD_DECEMBRE_2025 1.xlsx", "Fact_USSD", {
            "Nb_Transactions_USSD": "NB_USSD"
        })

        cursor.close()
        conn.close()
        print("\n🏆 ETL COMPLET ET ENRICHI RÉUSSI !")

    except Exception as e:
        print(f"❌ Erreur critique : {e}")

if __name__ == "__main__":
    load_all_facts_enriched()