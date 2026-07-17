import streamlit as st
import pandas as pd
import joblib

# Configuration
st.set_page_config(page_title="Tunisie Telecom - Scoring Appétence", layout="wide")
st.title("🎯 Prédiction de l'Appétence Data")

# Chargement du modèle et des colonnes
model = joblib.load("model_appetence_tt.joblib")
features = joblib.load("model_features.joblib")

# Sidebar pour les entrées utilisateur
st.sidebar.header("Profil du Client à tester")
def user_input():
    duree_in = st.sidebar.slider("Durée Appels Entrants", 0, 500, 100)
    revenu = st.sidebar.number_input("Revenu CDR (Sortant)", 0, 1000, 50)
    offre = st.sidebar.selectbox("ID Offre", [1, 2, 3, 4, 5])
    region = st.sidebar.selectbox("ID Région", [10, 20, 30, 40])
    
    data = {'DUREE_APPEL_IN': duree_in, 'revenu_cdr': revenu, 
            'ID_OFFRE': offre, 'ID_REGION': region}
    return pd.DataFrame([data])

input_df = user_input()

# Prédiction
if st.button("Lancer le Scoring"):
    prediction = model.predict(input_df)
    probabilite = model.predict_proba(input_df)[0][1]
    
    st.subheader("Résultat de l'Analyse")
    if prediction[0] == 1:
        st.success(f"🚀 Client à FORT POTENTIEL (Probabilité : {probabilite:.2%})")
        st.write("Action recommandée : Envoyer une offre Forfait Data illimité.")
    else:
        st.warning(f"📉 Client à FAIBLE POTENTIEL (Probabilité : {probabilite:.2%})")
        st.write("Action recommandée : Maintenir sur les offres Voix classiques.")