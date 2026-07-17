import sqlalchemy as sa
import urllib

# Vos paramètres validés
SERVER = r'DESKTOP-26NRQU5\SQLEXPRESS01' 
DATABASE = 'DW_Telecom'

connection_string = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={SERVER};'
    f'DATABASE={DATABASE};'
    f'Trusted_Connection=yes;'
    'Encrypt=no;'
    'TrustServerCertificate=yes;'
)

params = urllib.parse.quote_plus(connection_string)
engine = sa.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

def forcer_liaison_dim_offre():
    # On utilise "begin()" pour que SQLAlchemy gère le commit automatiquement
    try:
        with engine.begin() as conn:
            print(f"✅ Connecté avec succès à {DATABASE}")
            
            # Bloc SQL pour réparer et lier
            sql = """
            -- 1. Nettoyage
            IF EXISTS (SELECT * FROM sys.foreign_keys WHERE name = 'FK_Client_Offre')
                ALTER TABLE Dim_Client DROP CONSTRAINT FK_Client_Offre;

            -- 2. Préparation de la Clé Primaire sur Dim_Offre
            ALTER TABLE Dim_Offre ALTER COLUMN Offre_ID INT NOT NULL;
            
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE object_id = OBJECT_ID('Dim_Offre') AND is_primary_key = 1)
                ALTER TABLE Dim_Offre ADD CONSTRAINT PK_Dim_Offre_PFE PRIMARY KEY (Offre_ID);

            -- 3. Alignement des données (Intégrité)
            UPDATE Dim_Client 
            SET ID_OFFRE = 0 
            WHERE ID_OFFRE NOT IN (SELECT Offre_ID FROM Dim_Offre) OR ID_OFFRE IS NULL;

            -- 4. Création de la liaison
            ALTER TABLE Dim_Client 
            ADD CONSTRAINT FK_Client_Offre 
            FOREIGN KEY (ID_OFFRE) REFERENCES Dim_Offre(Offre_ID);
            """
            
            conn.execute(sa.text(sql))
            print("🎉 La liaison est maintenant établie dans la base de données !")

    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    forcer_liaison_dim_offre()