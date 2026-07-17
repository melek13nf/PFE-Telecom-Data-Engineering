from src.config import engine
try:
    with engine.connect() as conn:
        print("✅ Connexion à SSMS réussie !")
except Exception as e:
    print(f"❌ Erreur de connexion : {e}")