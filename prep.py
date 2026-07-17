import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def prepare_professional_data():
    print("🔄 Fusion des sources de données en cours...")
    
    # Chargement des différentes sources
    df_base = pd.read_excel("data/ECH__DECEMBRE_2025 1.xlsx")
    df_sortant = pd.read_excel("data/SORTANT_DECEMBRE_2025 1.xlsx")
    df_entrant = pd.read_excel("data/ENTRANT_DECEMBRE_2025 1.xlsx")
    df_recharge = pd.read_excel("data/RECHARGE_DECEMBRE_2025 1.xlsx")
    df_ussd = pd.read_excel("data/USSD_DECEMBRE_2025 1.xlsx")
    
    # Agrégation des données transactionnelles (on prend la moyenne par ID)
    df_sortant_agg = df_sortant.groupby('ID')[['revenu_cdr', 'DUREE_APPEL_TOT', 'revenu_sms']].mean().reset_index()
    df_entrant_agg = df_entrant.groupby('ID')[['DUREE_APPEL_IN', 'NB_SMS_IN']].mean().reset_index()
    df_recharge_agg = df_recharge.groupby('ID')[['MNT_RECH', 'NB_RECH']].mean().reset_index()
    df_ussd_agg = df_ussd.groupby('ID')[['MNT_FORFAIT_DATA', 'NB_FORFAIT_DATA']].mean().reset_index()

    # Fusion (Left Join sur l'ID)
    df_final = df_base.merge(df_sortant_agg, on='ID', how='left')
    df_final = df_final.merge(df_entrant_agg, on='ID', how='left')
    df_final = df_final.merge(df_recharge_agg, on='ID', how='left')
    df_final = df_final.merge(df_ussd_agg, on='ID', how='left')

    # Nettoyage des valeurs manquantes (si un client n'a pas rechargé, on met 0)
    df_final = df_final.fillna(0)
    
    return df_final

def train_final_model(df):
    # Sélection d'un set de variables beaucoup plus riche
    features = [
        'ANC_M', 'ID_REGION', 'ID_OFFRE', 
        'revenu_cdr', 'DUREE_APPEL_TOT', 'revenu_sms',
        'DUREE_APPEL_IN', 'NB_SMS_IN',
        'MNT_RECH', 'MNT_FORFAIT_DATA'
    ]
    
    X = df[features]
    # Création d'une cible : Appétent si consomme de la DATA ou recharge beaucoup
    y = ((df['MNT_FORFAIT_DATA'] > df['MNT_FORFAIT_DATA'].median()) | (df['MNT_RECH'] > 30)).astype(int)
    
    # Entraînement XGBoost avec SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_res, y_res)
    
    joblib.dump(model, "model_appetence_tt.joblib")
    print("✅ Modèle Professionnel Sauvegardé !")

# Exécution
df_full = prepare_professional_data()
train_final_model(df_full)