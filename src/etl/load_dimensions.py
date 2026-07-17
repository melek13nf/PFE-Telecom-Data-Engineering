import pandas as pd
import os
import pyodbc
from src.config import SERVER, DATABASE

def load_all_dimensions():
    print("⏳ Chargement de TOUTES les dimensions (Region, Offre, Client, Temps)...")
    base_path = os.path.join(os.getcwd(), "data")
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # --- 1. REGION & OFFRE ---
        for table, file in [("Dim_Region", "REGION_ 1.xlsx"), ("Dim_Offre", "offre_ 1.xlsx")]:
            print(f"   -> Mise à jour de {table}...")
            df = pd.read_excel(os.path.join(base_path, file))
            try:
                cursor.execute(f"DELETE FROM {table}")
            except: pass
            for _, row in df.iterrows():
                try: cursor.execute(f"INSERT INTO {table} VALUES (?, ?)", row.iloc[0], row.iloc[1])
                except: continue

        # --- 2. CLIENT ---
        print("   -> Mise à jour de Dim_Client...")
        df_ech = pd.read_excel(os.path.join(base_path, "ECH__DECEMBRE_2025 1.xlsx"))
        df_ech.columns = df_ech.columns.str.strip().str.upper()
        for client_id in df_ech['ID'].drop_duplicates().tolist():
            try: cursor.execute("INSERT INTO Dim_Client (Client_ID) VALUES (?)", client_id)
            except: continue

        # --- 3. TEMPS (GESTION DU CONFLIT) ---
        print("   -> Mise à jour du Calendrier 2025...")
        try:
            cursor.execute("DELETE FROM Dim_Temps")
            cursor.execute("SET IDENTITY_INSERT Dim_Temps ON")
            dates = pd.date_range(start='2025-01-01', end='2025-12-31')
            for d in dates:
                cursor.execute("""
                    INSERT INTO Dim_Temps (Time_Key, Date_Mois, Mois, Annee, Nom_Mois) 
                    VALUES (?, ?, ?, ?, ?)""",
                    int(d.strftime('%Y%m%d')), d.date(), d.month, d.year, d.strftime('%B')
                )
            cursor.execute("SET IDENTITY_INSERT Dim_Temps OFF")
        except pyodbc.Error as e:
            if '547' in str(e):
                print("   ⚠️ Calendrier déjà lié aux faits, conservation des dates existantes.")
            else:
                print(f"   ❌ Erreur Temps : {e}")

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ DIMENSIONS PRÊTES !")
    except Exception as e:
        print(f"❌ Erreur critique : {e}")