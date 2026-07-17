import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- ÉTAPE 0 : CHARGEMENT DES DONNÉES (Indispensable !) ---
path = 'data'
# On charge le fichier profil qui contient les infos de base
df_p = pd.read_excel(os.path.join(path, 'ECH__DECEMBRE_2025 1.xlsx'))
df_r = pd.read_excel(os.path.join(path, 'RECHARGE_DECEMBRE_2025 1.xlsx'))
df_u = pd.read_excel(os.path.join(path, 'SORTANT_DECEMBRE_2025 1.xlsx'))

# Nettoyage rapide des noms de colonnes
for d in [df_p, df_r, df_u]:
    d.columns = d.columns.str.strip().str.upper()

# Fusion pour créer le 'df' dont le script a besoin
rech_agg = df_r.groupby('ID')['MNT_RECH'].sum().reset_index()
usage_agg = df_u.groupby('ID')[['DUREE_APPEL_TOT', 'NB_SMS_TOT']].sum().reset_index()

df = df_p.merge(rech_agg, on='ID', how='left').merge(usage_agg, on='ID', how='left')
df = df.fillna(0) # Remplacer les vides par 0

# --- ÉTAPE 1 : SELECTION DES VARIABLES ---
features = ['MNT_RECH', 'DUREE_APPEL_TOT', 'NB_SMS_TOT']
X = df[features]

# --- ÉTAPE 2 : MISE À L'ÉCHELLE (Standardization) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- ÉTAPE 3 : APPLICATION DU K-MEANS (K=3) ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['CLUSTER'] = kmeans.fit_predict(X_scaled)

# --- ÉTAPE 4 : GÉNÉRATION DU GRAPHIQUE ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='DUREE_APPEL_TOT', y='MNT_RECH', hue='CLUSTER', palette='viridis', s=100)

plt.title('Dispersion des Clusters Clients (Tunisie Telecom)')
plt.xlabel('Usage Voix (DUREE_APPEL_TOT)')
plt.ylabel('Revenu (MNT_RECH)')
plt.grid(True)
plt.savefig('dispersion_clusters_pfe.png')
plt.show()

print("✅ Graphique généré avec succès !")