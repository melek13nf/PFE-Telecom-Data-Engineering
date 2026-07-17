import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# --- ÉTAPE 0 : PRÉPARATION DES DONNÉES (Simulé ici) ---
# Dans ton vrai projet, utilise le dataframe 'df' déjà nettoyé, fusionné et avec 'TARGET_DATA'.
# Pour ce script exemple, nous créons un dataframe factice 'df' réaliste.
data = {
    'MNT_RECH': [10, 50, 20, 100, 30, 80, 15, 60, 25, 90] * 5,
    'DUREE_APPEL_TOT': [100, 500, 200, 1000, 300, 800, 150, 600, 250, 900] * 5,
    'NB_SMS_TOT': [20, 100, 40, 200, 60, 160, 30, 120, 50, 180] * 5,
    'ANC_M': [12, 60, 24, 120, 36, 96, 18, 72, 30, 108] * 5,
    'TARGET_DATA': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5 # 1 = Achat Data, 0 = Pas d'achat
}
df = pd.DataFrame(data)

# --- ÉTAPE 1 : DÉFINITION DES VARIABLES X (Features) ET Y (Target) ---
features = ['MNT_RECH', 'DUREE_APPEL_TOT', 'NB_SMS_TOT', 'ANC_M']
X = df[features]
y = df['TARGET_DATA']

# --- ÉTAPE 2 : SÉPARATION TRAIN/TEST (Important d'évaluer sur le Test) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- ÉTAPE 3 : ENTRAÎNEMENT DU MODÈLE RANDOM FOREST ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# --- ÉTAPE 4 : CALCUL DES PROBABILITÉS DE PRÉDICTION ---
# C'est la probabilité d'appartenir à la classe '1' (Achat Data)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# --- ÉTAPE 5 : CALCUL DE LA COURBE ROC ET DE L'AUC ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

# --- ÉTAPE 6 : GÉNÉRATION DU GRAPHIQUE ROC ---
plt.figure(figsize=(10, 7))
sns.set_style('whitegrid') # Style épuré pour le rapport

# Traçage de la courbe ROC (Ligne bleue)
plt.plot(fpr, tpr, color='#1f77b4', lw=3, label=f'Courbe ROC du Random Forest (AUC = {auc_score:.2f})')

# Traçage de la ligne de référence (Ciblage aléatoire = Diagonale en pointillés)
plt.plot([0, 1], [0, 1], color='#d62728', lw=2, linestyle='--', label='Ciblage Aléatoire (AUC = 0.50)')

# Zone remplie sous la courbe (Optionnel mais joli pour le rapport)
plt.fill_between(fpr, tpr, alpha=0.2, color='#1f77b4')

# Titres et labels professionnels
plt.title('Performance du Modèle d\'Appétence Data (Tunisie Telecom)', fontsize=16, fontweight='bold')
plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=12)
plt.ylabel('Taux de Vrais Positifs (Sensibilité / Rappel)', fontsize=12)
plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)

# Limites des axes
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])

# Sauvegarde pour le rapport
plt.savefig('courbe_roc_auc_pfe.png', dpi=300) # dpi=300 pour haute qualité d'impression
plt.show()

print(f"✅ Courbe ROC générée avec succès ! Score AUC = {auc_score:.4f}")