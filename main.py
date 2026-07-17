from src.etl.load_dimensions import load_all_dimensions
from src.etl.load_all_facts import load_all_facts_enriched # On ajoute _enriched

print("--- DÉMARRAGE GLOBAL DE L'ETL ---")
load_all_dimensions()
load_all_facts_enriched() # On appelle la nouvelle fonction
print("--- TOUT EST PRÊT POUR STREAMLIT ! ---")