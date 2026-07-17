import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- ÉTAPE 0 : CHARGEMENT ET PRÉPARATION MINIMALE DES DONNÉES (Simulé ici) ---
# Dans ton vrai projet, utilise le dataframe 'df' déjà nettoyé et fusionné.
# Pour ce script exemple, nous créons un dataframe factice 'df'
data = {
    'MNT_RECH': [10, 50, 20, 100, 30, 80, 15, 60, 25, 90],
    'DUREE_APPEL_TOT': [100, 500, 200, 1000, 300, 800, 150, 600, 250, 900],
    'NB_SMS_TOT': [20, 100, 40, 200, 60, 160, 30, 120, 50, 180],
    'ANC_M': [12, 60, 24, 120, 36, 96, 18, 72, 30, 108],
    'TARGET_DATA': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # 1 = Achat Data, 0 = Pas d'achat
}
df = pd.DataFrame(data)

# --- ÉTAPE 1 : DÉFINITION DES VARIABLES X (Features) ET Y (Target) ---
# On exclut l'ID et la variable Target de X
features = ['MNT_RECH', 'DUREE_APPEL_TOT', 'NB_SMS_TOT', 'ANC_M']
X = df[features]
y = df['TARGET_DATA']

# --- ÉTAPE 2 : SÉPARATION TRAIN/TEST (Crucial pour la validation) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- ÉTAPE 3 : ENTRAÎNEMENT DU MODÈLE RANDOM FOREST ---
# On utilise random_state pour la reproductibilité des résultats
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# --- ÉTAPE 4 : EXTRACTION ET VISUALISATION DE L'IMPORTANCE DES VARIABLES ---
# Récupération des scores d'importance
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Tri par ordre décroissant
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# Génération du graphique à barres
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')

plt.title('Importance des Variables dans la Prédiction d\'Achat Data (Random Forest)')
plt.xlabel('Score d\'Importance (Gini Importance)')
plt.ylabel('Variable (Feature)')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)

# Sauvegarde pour le rapport
plt.savefig('feature_importance_pfe.png')
plt.show()

print("✅ Graphique 'Feature Importance' généré avec succès !")
print(feature_imp_df) # Affiche aussi les valeurs numériques