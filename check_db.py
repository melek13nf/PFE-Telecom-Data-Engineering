import pyodbc
from src.config import SERVER, DATABASE

def check_structure():
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        tables = ['Dim_Client', 'Fact_Recharge', 'Dim_Region', 'Dim_Offre']
        print(f"✅ Connexion réussie à {DATABASE}\n")
        
        for table in tables:
            print(f"🔎 Table : {table}")
            print("-" * 30)
            try:
                # Récupère les noms de colonnes et les types de données
                cursor.execute(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}'")
                rows = cursor.fetchall()
                if not rows:
                    print("⚠️ Table vide ou non trouvée.")
                for row in rows:
                    print(f"  Column: {row[0]:<20} | Type: {row[1]}")
            except Exception as e:
                print(f"❌ Erreur lors de l'inspection de {table}: {e}")
            print("\n")
            
        conn.close()
    except Exception as e:
        print(f"🚫 Erreur de connexion : {e}")

if __name__ == "__main__":
    check_structure()