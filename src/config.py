import sqlalchemy as sa
import pyodbc 

SERVER = r'DESKTOP-26NRQU5\SQLEXPRESS01'
DATABASE = 'DW_Telecom'

# URL pour SQLAlchemy
connection_url = f"mssql+pyodbc://{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
engine = sa.create_engine(connection_url)

# Connexion DIRECTE pour éviter le bug de curseur
def get_raw_conn():
    return pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;')