import pandas as pd
import os
import pyodbc
from src.config import SERVER, DATABASE

def load_facts():
    print("🚀 Chargement de la Table de Faits (Fact_Entrant)...")
    base_path = os.path.join(os.getcwd(), "data")
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
    
    try:
        # 1. Chargement de l'Excel
        path_entrant = os.path.join(base_path, "ENTRANT_DECEMBRE_2025 1.xlsx")
        df_entrant = pd.read_excel(path_entrant)
        
        # Nettoyage des noms de colonnes : tout en majuscules et sans espaces
        df_entrant.columns = df_entrant.columns.str.strip().str.upper()
        
        print(f"🔍 Colonnes détectées dans l'Excel : {df_entrant.columns.tolist()}")

        # 2. Connexion
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # 3. Récupération des Client_Key (Lookup)
        cursor.execute("SELECT Client_Key, Client_ID FROM Dim_Client")
        client_lookup = {str(row[1]): row[0] for row in cursor.fetchall()}

        # 4. Vidage de la table de faits pour éviter les doublons
        cursor.execute("DELETE FROM Fact_Entrant")

        # 5. Insertion optimisée
        # On essaie de trouver les colonnes même si le nom varie un peu
        col_id = 'ID'
        col_date = 'MONTH_DT'
        # On cherche une colonne qui contient 'DUREE' et 'SMS'
        col_duree = [c for c in df_entrant.columns if 'DUREE' in c][0]
        col_sms = [c for c in df_entrant.columns if 'SMS' in c][0]

        print(f"   -> Utilisation des colonnes : {col_duree} et {col_sms}")
        print(f"   -> Insertion de {len(df_entrant)} lignes...")
        
        # Pour aller plus vite sur 345 000 lignes, on peut utiliser fast_executemany si nécessaire
        # Mais restons sur la boucle simple pour l'instant
        for _, row in df_entrant.iterrows():
            client_id_excel = str(row[col_id])
            client_key = client_lookup.get(client_id_excel)
            
            try:
                date_val = pd.to_datetime(row[col_date])
                time_key = int(date_val.strftime('%Y%m%d'))
            except:
                time_key = 20251201

            if client_key:
                cursor.execute("""
                    INSERT INTO Fact_Entrant (Client_Key, Time_Key, Duree_Appel_In, NB_SMS_In)
                    VALUES (?, ?, ?, ?)""",
                    client_key, time_key, row[col_duree], row[col_sms]
                )

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Table Fact_Entrant chargée avec succès !")

    except Exception as e:
        print(f"❌ Erreur lors du chargement des faits : {e}")

if __name__ == "__main__":
    load_facts()