import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
st.set_page_config(page_title="Tunisie Telecom AI - PFE", layout="wide", page_icon="📡")

st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.07); }
    h1, h2, h3 { color: #004a99; font-family: 'Arial Black'; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 1. CHARGEMENT & PRÉPARATION
# ============================================================
@st.cache_data
def load_and_prepare_data():
    folder = "data"
    try:
        df_p    = pd.read_excel(os.path.join(folder, 'ECH__DECEMBRE_2025 1.xlsx'))
        df_r    = pd.read_excel(os.path.join(folder, 'RECHARGE_DECEMBRE_2025 1.xlsx'))
        df_s    = pd.read_excel(os.path.join(folder, 'SORTANT_DECEMBRE_2025 1.xlsx'))
        df_e    = pd.read_excel(os.path.join(folder, 'ENTRANT_DECEMBRE_2025 1.xlsx'))
        df_ussd = pd.read_excel(os.path.join(folder, 'USSD_DECEMBRE_2025 1.xlsx'))
        df_reg  = pd.read_excel(os.path.join(folder, 'REGION_ 1.xlsx'))
        df_o    = pd.read_excel(os.path.join(folder, 'offre_ 1.xlsx'))
    except Exception as e:
        st.error(f"⚠️ Erreur chargement : {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "ID"

    for d in [df_p, df_r, df_s, df_e, df_ussd, df_reg, df_o]:
        d.columns = d.columns.str.strip().str.upper()

    key = 'ID'
    for d in [df_p, df_r, df_s, df_e, df_ussd]:
        d[key] = d[key].astype(str).str.strip().str.upper()

    # Parse dates
    for d in [df_r, df_s, df_e, df_ussd]:
        d['MONTH_DT'] = pd.to_datetime(d['MONTH_DT'].str[:9], format='%d%b%Y', errors='coerce')

    # --- Agrégations ---
    r_agg = df_r.groupby(key).agg(
        MNT_RECH=('MNT_RECH', 'sum'),
        NB_RECH=('NB_RECH', 'sum'),
        MNT_RECH_SUP5=('MNT_RECH_SUP5', 'sum'),
        NB_RECH_SUP5=('NB_RECH_SUP5', 'sum')
    ).reset_index()

    s_agg = df_s.groupby(key).agg(
        REVENU_CDR=('REVENU_CDR', 'sum'),
        REVENU_VOIX=('REVENU_VOIX', 'sum'),
        REVENU_INTER=('REVENU_INTER', 'sum'),
        REVENU_ROAMING=('REVENU_ROAMING', 'sum'),
        DUREE_APPEL_TOT=('DUREE_APPEL_TOT', 'sum'),
        DUREE_APPEL_OOREDOO_TOT=('DUREE_APPEL_OOREDOO_TOT', 'sum'),
        DUREE_APPEL_ORANGE_TOT=('DUREE_APPEL_ORANGE_TOT', 'sum'),
        NB_APPEL_TOT=('NB_APPEL_TOT', 'sum'),
        REVENU_SMS=('REVENU_SMS', 'sum'),
        NB_SMS_TOT=('NB_SMS_TOT', 'sum'),
        DUREE_OFFNET_TOT=('DUREE_OFFNET_TOT', 'sum'),
    ).reset_index()

    e_agg = df_e.groupby(key).agg(
        DUREE_APPEL_IN=('DUREE_APPEL_IN', 'sum'),
        DUREE_OOREDOO_IN=('DUREE_OOREDOO_IN', 'sum'),
        DUREE_ORANGE_IN=('DUREE_ORANGE_IN', 'sum'),
        NB_SMS_IN=('NB_SMS_IN', 'sum'),
    ).reset_index()

    ussd_agg = df_ussd.groupby(key).agg(
        MNT_FORFAIT=('MNT_FORFAIT', 'sum'),
        MNT_FORFAIT_DATA=('MNT_FORFAIT_DATA', 'sum'),
        NB_FORFAIT=('NB_FORFAIT', 'sum'),
        NB_FORFAIT_DATA=('NB_FORFAIT_DATA', 'sum'),
    ).reset_index()

    # Master join
    master = (df_p
        .merge(r_agg,    on=key, how='left')
        .merge(s_agg,    on=key, how='left')
        .merge(e_agg,    on=key, how='left')
        .merge(ussd_agg, on=key, how='left')
        .merge(df_reg,   on='ID_REGION', how='left')
        .merge(df_o,     on='ID_OFFRE',  how='left')
    ).fillna(0)

    # Derived features
    master['DUREE_ONNET_TOT'] = master['DUREE_APPEL_TOT'] - master['DUREE_OFFNET_TOT']
    master['REVENU_TOTAL']    = master['REVENU_CDR'] + master['MNT_RECH']
    master['PCT_DATA']        = np.where(
        master['MNT_FORFAIT'] > 0,
        master['MNT_FORFAIT_DATA'] / master['MNT_FORFAIT'],
        0
    )
    master['TRANCHE_ANC'] = pd.cut(
        master['ANC_M'], bins=[0, 12, 36, 72, 500],
        labels=['0-1 an', '1-3 ans', '3-6 ans', '6 ans+']
    ).astype(str)  # Conversion en str pour éviter les erreurs de tri MultiIndex

    # Target IA (appétence data)
    ussd_ids = set(df_ussd[df_ussd['MNT_FORFAIT_DATA'] > 0][key].unique())
    master['TARGET_IA'] = master[key].apply(lambda x: 1 if x in ussd_ids else 0)
    if master['TARGET_IA'].nunique() < 2:
        master['TARGET_IA'] = (np.random.random(len(master)) > 0.7).astype(int)

    return master, df_r, df_s, df_e, df_ussd, key

df, df_r_raw, df_s_raw, df_e_raw, df_ussd_raw, id_col = load_and_prepare_data()

# ============================================================
# 2. CLUSTERING
# ============================================================
@st.cache_data
def perform_clustering(data):
    features = ['MNT_RECH', 'DUREE_APPEL_TOT', 'NB_SMS_TOT', 'REVENU_CDR']
    X = data[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    data = data.copy()
    data['CLUSTER'] = km.fit_predict(X_scaled)
    # Name clusters by revenue
    centers = data.groupby('CLUSTER')['MNT_RECH'].mean().sort_values(ascending=False)
    labels = {centers.index[0]: "🏆 Clients VIP", centers.index[1]: "📱 Consommateurs Data",
              centers.index[2]: "📞 Consommateurs Voix", centers.index[3]: "💰 Petits Budgets"}
    data['PROFIL'] = data['CLUSTER'].map(labels)
    return data

# ============================================================
# 3. MODÈLE IA – Chargement depuis model_appetence_tt.joblib
# ============================================================
MODEL_PATH         = "model_appetence_tt.joblib"
MODEL_FORFAIT_PATH = "model_forfait_cv.joblib"
MODEL_CHURN_PATH   = "model_churn_cv.joblib"

@st.cache_resource
def load_cv_models():
    """Charge les 2 modèles CV (Forfait + Churn) depuis les fichiers joblib."""
    import os
    results = {}
    for name, path in [("forfait", MODEL_FORFAIT_PATH),
                        ("churn",   MODEL_CHURN_PATH)]:
        if os.path.exists(path):
            art = joblib.load(path)
            results[name] = art
            print(f"Modèle {name} chargé — AUC CV : {art['metrics']['auc_cv_mean']:.4f}")
        else:
            results[name] = None
    return results

def prepare_features_cv(df, encoders, features):
    """Prépare les features pour les modèles CV."""
    d = df.copy()
    d['HANDSET']      = d['HANDSET'].replace(0,'Inconnu').astype(str)
    d['STATUT']       = d['STATUT'].astype(str)
    cat_cols = ['HANDSET','STATUT','CLASSE_CANAL'] if 'CLASSE_CANAL' in d.columns else ['HANDSET','STATUT']
    if 'CLASSE_CANAL' not in d.columns:
        d['CLASSE_CANAL'] = 'Inconnu'
    d['CLASSE_CANAL'] = d['CLASSE_CANAL'].fillna('Inconnu').astype(str)
    for col in cat_cols:
        if col in encoders:
            le    = encoders[col]
            known = set(le.classes_)
            d[col+'_ENC'] = d[col].apply(lambda x: x if x in known else le.classes_[0])
            d[col+'_ENC'] = le.transform(d[col+'_ENC'])
    for f in features:
        if f not in d.columns:
            d[f] = 0
    return d[features].apply(pd.to_numeric, errors='coerce').fillna(0)

@st.cache_resource
def load_model_artifact(path):
    """
    Charge l'artifact produit par train_model_fixed.py.
    Contient : model XGBoost + encodeurs LabelEncoder + feature_names + metrics.
    Fallback automatique si le .joblib est absent (mode démo).
    """
    if not os.path.exists(path):
        return None  # Géré en bas avec le fallback

    artifact = joblib.load(path)
    required = ['model', 'encoders', 'feature_names', 'metrics']
    if not all(k in artifact for k in required):
        return None  # Format incompatible → fallback

    return artifact

@st.cache_data
def prepare_features_from_artifact(data, _artifact):
    """
    Applique les mêmes transformations qu'à l'entraînement.
    Gère le mapping entre les noms de colonnes du dashboard (Excel bruts)
    et les noms attendus par le modèle (ABT de train_model_v2.py).
    """
    encoders      = _artifact['encoders']
    feature_names = _artifact['feature_names']

    df_feat = data.copy()

    # ── Mapping dashboard → ABT ──────────────────────────────────────────────
    # Le dashboard agrège les Excel avec des noms courts (MNT_RECH, REVENU_CDR…)
    # Le modèle attend les noms longs de l'ABT (MNT_RECH_TOT, REVENU_CDR_TOT…)
    ALIAS = {
        'MNT_RECH_TOT'      : ['MNT_RECH_TOT',      'MNT_RECH'],
        'MNT_RECH_MOY'      : ['MNT_RECH_MOY',      'MNT_RECH'],
        'NB_RECH_TOT'       : ['NB_RECH_TOT',       'NB_RECH'],
        'NB_MOIS_ACTIF_R'   : ['NB_MOIS_ACTIF_R',   'NB_RECH'],
        'REVENU_CDR_TOT'    : ['REVENU_CDR_TOT',    'REVENU_CDR'],
        'DUREE_OFFNET_TOT'  : ['DUREE_OFFNET_TOT',  'DUREE_OFFNET_TOT'],
        'NB_APPEL_TOT'      : ['NB_APPEL_TOT',      'NB_APPEL_TOT'],
        'NB_SMS_TOT'        : ['NB_SMS_TOT',        'NB_SMS_TOT'],
        'REVENU_INTER_TOT'  : ['REVENU_INTER_TOT',  'REVENU_INTER'],
        'DUREE_APPEL_IN_TOT': ['DUREE_APPEL_IN_TOT','DUREE_APPEL_IN'],
        'NB_SMS_IN_TOT'     : ['NB_SMS_IN_TOT',     'NB_SMS_IN'],
    }
    for target_col, candidates in ALIAS.items():
        if target_col not in df_feat.columns:
            for src in candidates:
                if src in df_feat.columns:
                    df_feat[target_col] = df_feat[src]
                    break
            else:
                df_feat[target_col] = 0  # valeur par défaut si introuvable

    # Nettoyage
    df_feat['MNT_RECH_TOT'] = df_feat['MNT_RECH_TOT'].clip(lower=0)
    df_feat['MNT_RECH_MOY'] = df_feat['MNT_RECH_MOY'].clip(lower=0)
    df_feat['HANDSET']   = df_feat['HANDSET'].replace(0, 'Inconnu').astype(str)
    df_feat['STATUT']    = df_feat['STATUT'].astype(str)
    df_feat['OFFRE_CAT'] = df_feat['OFFRE_CAT'].astype(str)

    # Encodage catégoriel avec les LabelEncoders sauvegardés
    for col, le in encoders.items():
        known = set(le.classes_)
        df_feat[col] = df_feat[col].apply(lambda x: x if x in known else le.classes_[0])
        df_feat[col + '_ENC'] = le.transform(df_feat[col])

    # Toute feature encore manquante → 0
    for f in feature_names:
        if f not in df_feat.columns:
            df_feat[f] = 0

    X = df_feat[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0)
    return X

# --- Chargement de l'artifact ---
artifact = load_model_artifact(MODEL_PATH)
MODEL_SOURCE = "joblib"  # mode par défaut

if not df.empty:
    df = perform_clustering(df)

    if artifact is not None:
        # ── Mode production : modèle réel depuis le .joblib ──────────────────
        model         = artifact['model']
        model_features= artifact['feature_names']
        feat_importances = model.feature_importances_
        metrics        = artifact['metrics']
        accuracy       = metrics.get('auc_test', 0.0)   # On affiche l'AUC comme référence
        auc_score      = metrics['auc_test']
        conf_matrix    = None  # Pas recalculé en prod (calculé à l'entraînement)

        # Scorer tout le parc avec le vrai modèle
        X_parc = prepare_features_from_artifact(df, artifact)
        df['TARGET_IA']  = (df.get('MNT_FORFAIT_DATA_TOT',
                             df.get('MNT_FORFAIT_DATA', 0)) > 0).astype(int)
        df['PROBA_DATA'] = model.predict_proba(X_parc)[:, 1]

    else:
        # ── Mode fallback : si le .joblib n'existe pas encore ────────────────
        MODEL_SOURCE = "fallback"
        from sklearn.ensemble import RandomForestClassifier
        feats_fb = ['ANC_M', 'MNT_RECH', 'DUREE_APPEL_TOT',
                    'NB_SMS_TOT', 'REVENU_CDR', 'NB_APPEL_TOT',
                    'DUREE_OFFNET_TOT', 'NB_SMS_IN']
        # Noms alignés sur le master dashboard (Excel bruts, pas l'ABT)
        # MNT_FORFAIT_DATA retiré : leakage direct avec TARGET_IA
        feats_fb = [f for f in feats_fb if f in df.columns]

        X_fb = df[feats_fb].apply(pd.to_numeric, errors='coerce').fillna(0)
        y_fb = (df.get('MNT_FORFAIT_DATA_TOT', df.get('MNT_FORFAIT_DATA', 0)) > 0).astype(int)

        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X_fb, y_fb, test_size=0.2,
                                                    random_state=42, stratify=y_fb)
        model_fb = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model_fb.fit(X_tr, y_tr)

        model          = model_fb
        model_features = feats_fb
        feat_importances = model_fb.feature_importances_
        y_pred_fb      = model_fb.predict(X_te)
        y_proba_fb     = model_fb.predict_proba(X_te)[:, 1]
        accuracy       = accuracy_score(y_te, y_pred_fb)
        auc_score      = roc_auc_score(y_te, y_proba_fb)
        conf_matrix    = confusion_matrix(y_te, y_pred_fb)
        df['TARGET_IA']  = y_fb.values
        df['PROBA_DATA'] = model_fb.predict_proba(X_fb)[:, 1]

# ============================================================
# NAVIGATION
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/fr/thumb/7/7a/Tunisie_T%C3%A9l%C3%A9com_logo_2021.svg/1200px-Tunisie_T%C3%A9l%C3%A9com_logo_2021.svg.png",
                 width=200)
st.sidebar.title("Navigation")
nav = st.sidebar.radio("", [
    "📊 Dashboard 360°",
    "📈 Analyse Temporelle",
    "🗺️ Analyse Géographique",
    "📞 Analyse Usage",
    "🧩 Segmentation Clients",
    "🧠 Évaluation IA",
    "🔮 Prédiction Individuelle",
    "📊 Analyse RFM",
    "⚠️  Risque de Churn",
    "📈 Modèles CV",
    "🎯 Scoring CV",
    "⚙️ MLOps"
])
st.sidebar.markdown("---")
if MODEL_SOURCE == "joblib":
    st.sidebar.success("🟢 Modèle : XGBoost (.joblib)")
else:
    st.sidebar.warning("🟡 Modèle : Fallback (lancez train_model_fixed.py)")
st.sidebar.caption("PFE Tunisie Telecom – Master Data Science 2026")

# ============================================================
# PAGE 1 : DASHBOARD 360°
# ============================================================
if nav == "📊 Dashboard 360°":
    st.title("📊 Dashboard 360° – Vue Globale")

    if df.empty:
        st.warning("Données manquantes.")
    else:
        # KPIs
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("👥 Abonnés",         f"{len(df):,}")
        c2.metric("💰 Revenu Recharge", f"{df['MNT_RECH'].sum():,.0f} DT")
        c3.metric("📞 Durée Voix (h)",  f"{df['DUREE_APPEL_TOT'].sum()/60:,.0f}")
        c4.metric("📡 Dépense Data",    f"{df['MNT_FORFAIT_DATA'].sum():,.0f} DT")
        c5.metric("🎯 Appétence Data",  f"{(df['TARGET_IA']==1).mean():.1%}")
        c6.metric("📲 Taux 4G/5G",      f"{(df['HANDSET'].isin(['4G','5G'])).mean():.1%}")

        st.divider()

        # Ligne 1
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.subheader("📡 Répartition par Type Réseau (Handset)")
            hs = df['HANDSET'].value_counts().reset_index()
            hs.columns = ['Handset', 'Count']
            colors = {'2G': '#ef4444', '3G': '#f97316', '4G': '#3b82f6', '5G': '#10b981'}
            fig = px.pie(hs, names='Handset', values='Count', hole=0.55,
                         color='Handset', color_discrete_map=colors,
                         title="")
            fig.update_traces(textinfo='label+percent', textfont_size=13)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎟️ Répartition par Catégorie d'Offre")
            oc = df['OFFRE_CAT'].value_counts().reset_index()
            oc.columns = ['Offre', 'Count']
            fig2 = px.bar(oc, x='Count', y='Offre', orientation='h',
                          color='Count', color_continuous_scale='Blues',
                          text='Count')
            fig2.update_traces(textposition='outside')
            fig2.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            st.subheader("📋 Statut Parc")
            st.markdown("**Statut Activité**")
            stat = df['STATUT'].value_counts()
            for s, v in stat.items():
                pct = v / len(df) * 100
                color = "🟢" if s == 'A' else "🔴"
                label = "Actif" if s == 'A' else "Suspendu"
                st.metric(f"{color} {label}", f"{v:,}", f"{pct:.1f}%")

            st.markdown("**Statut RGS90**")
            stat2 = df['STATUT_RGS90'].value_counts()
            for s, v in stat2.items():
                pct = v / len(df) * 100
                color = "🟢" if s == 'A' else "🟡"
                label = "Actif" if s == 'A' else "Risque"
                st.metric(f"{color} {label}", f"{v:,}", f"{pct:.1f}%")

        st.divider()

        # Ligne 2 : Canal & Ancienneté
        col4, col5 = st.columns(2)
        with col4:
            st.subheader("🏪 Canal de Vente")
            canal = df[df['CANAL_DE_VENTE'] != '0']['CANAL_DE_VENTE'].value_counts().reset_index()
            canal.columns = ['Canal', 'Count']
            fig3 = px.pie(canal, names='Canal', values='Count', hole=0.4,
                          color_discrete_sequence=['#004a99', '#0ea5e9', '#bae6fd'])
            st.plotly_chart(fig3, use_container_width=True)

        with col5:
            st.subheader("🕒 Distribution de l'Ancienneté")
            fig4 = px.histogram(df, x='ANC_M', nbins=40,
                                color_discrete_sequence=['#004a99'],
                                labels={'ANC_M': 'Ancienneté (mois)'},
                                title="")
            fig4.update_layout(bargap=0.05)
            st.plotly_chart(fig4, use_container_width=True)

        # Ligne 3 : Revenu par offre cat
        st.divider()
        col6, col7 = st.columns(2)
        with col6:
            st.subheader("💳 Revenu Moyen par Catégorie d'Offre")
            rev_offre = df.groupby('OFFRE_CAT')['MNT_RECH'].mean().sort_values(ascending=False).reset_index()
            fig5 = px.bar(rev_offre, x='OFFRE_CAT', y='MNT_RECH',
                          color='MNT_RECH', color_continuous_scale='Teal',
                          labels={'MNT_RECH': 'Recharge Moy. (DT)', 'OFFRE_CAT': 'Catégorie'},
                          text_auto='.1f')
            st.plotly_chart(fig5, use_container_width=True)

        with col7:
            st.subheader("📊 Profil des Clients par Handset")
            fig6 = px.box(df, x='HANDSET', y='MNT_RECH', color='HANDSET',
                          labels={'MNT_RECH': 'Recharge Totale (DT)', 'HANDSET': 'Type Réseau'})
            st.plotly_chart(fig6, use_container_width=True)

# ============================================================
# PAGE 2 : ANALYSE TEMPORELLE
# ============================================================
elif nav == "📈 Analyse Temporelle":
    st.title("📈 Analyse Temporelle – Évolution Mensuelle")

    if df_r_raw.empty:
        st.warning("Données manquantes.")
    else:
        # Recharge mensuelle
        st.subheader("💰 Évolution des Recharges (Jan → Mai 2025)")
        r_month = df_r_raw.groupby('MONTH_DT').agg(
            MNT_RECH=('MNT_RECH', 'sum'),
            NB_RECH=('NB_RECH', 'sum'),
            NB_CLIENTS=('ID', 'nunique')
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=r_month['MONTH_DT'], y=r_month['MNT_RECH'],
                             name='Montant Recharge (DT)', marker_color='#004a99'), secondary_y=False)
        fig.add_trace(go.Scatter(x=r_month['MONTH_DT'], y=r_month['NB_CLIENTS'],
                                 name='Nb Clients Actifs', mode='lines+markers',
                                 line=dict(color='#f97316', width=3)), secondary_y=True)
        fig.update_layout(legend=dict(orientation='h'), hovermode='x unified')
        fig.update_yaxes(title_text="Montant DT", secondary_y=False)
        fig.update_yaxes(title_text="Nb Clients", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Usage voix mensuel
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📞 Durée Voix Sortante Mensuelle")
            s_month = df_s_raw.groupby('MONTH_DT')['DUREE_APPEL_TOT'].sum().reset_index()
            fig2 = px.area(s_month, x='MONTH_DT', y='DUREE_APPEL_TOT',
                           color_discrete_sequence=['#3b82f6'],
                           labels={'DUREE_APPEL_TOT': 'Durée (min)', 'MONTH_DT': 'Mois'})
            fig2.update_traces(fill='tozeroy')
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.subheader("📡 Dépense Data Forfait Mensuelle")
            u_month = df_ussd_raw.groupby('MONTH_DT')['MNT_FORFAIT_DATA'].sum().reset_index()
            u_month = u_month.dropna(subset=['MONTH_DT'])
            fig3 = px.area(u_month, x='MONTH_DT', y='MNT_FORFAIT_DATA',
                           color_discrete_sequence=['#10b981'],
                           labels={'MNT_FORFAIT_DATA': 'Montant DT', 'MONTH_DT': 'Mois'})
            fig3.update_traces(fill='tozeroy')
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # Revenu Voix vs Inter vs Roaming mensuel
        st.subheader("💹 Décomposition du Revenu Sortant par Type")
        s_rev = df_s_raw.groupby('MONTH_DT').agg(
            Voix=('REVENU_VOIX', 'sum'),
            International=('REVENU_INTER', 'sum'),
            Roaming=('REVENU_ROAMING', 'sum'),
            SMS=('REVENU_SMS', 'sum')
        ).reset_index().melt(id_vars='MONTH_DT', var_name='Type', value_name='Revenu')
        fig4 = px.bar(s_rev, x='MONTH_DT', y='Revenu', color='Type',
                      barmode='stack', color_discrete_sequence=px.colors.qualitative.Set2,
                      labels={'MONTH_DT': 'Mois', 'Revenu': 'Revenu (DT)'})
        st.plotly_chart(fig4, use_container_width=True)

        st.divider()

        # Recharge SUP5 DT vs Totale
        st.subheader("🔍 Part des Recharges ≥ 5 DT vs Totales")
        r_sup5 = df_r_raw.groupby('MONTH_DT').agg(
            Total=('MNT_RECH', 'sum'),
            SUP5=('MNT_RECH_SUP5', 'sum')
        ).reset_index()
        r_sup5['PCT_SUP5'] = r_sup5['SUP5'] / r_sup5['Total'] * 100
        fig5 = px.line(r_sup5, x='MONTH_DT', y='PCT_SUP5',
                       markers=True, line_shape='spline',
                       labels={'PCT_SUP5': '% Recharges ≥5 DT', 'MONTH_DT': 'Mois'},
                       color_discrete_sequence=['#8b5cf6'])
        fig5.update_traces(line=dict(width=3))
        st.plotly_chart(fig5, use_container_width=True)

# ============================================================
# PAGE 3 : GÉOGRAPHIQUE
# ============================================================
elif nav == "🗺️ Analyse Géographique":
    st.title("🗺️ Analyse Géographique – Par Région")

    if df.empty:
        st.warning("Données manquantes.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 Nombre d'Abonnés par Région")
            reg_cnt = df.groupby('REGION').size().sort_values(ascending=True).reset_index()
            reg_cnt.columns = ['REGION', 'Abonnés']
            fig1 = px.bar(reg_cnt, x='Abonnés', y='REGION', orientation='h',
                          color='Abonnés', color_continuous_scale='Blues', text='Abonnés')
            fig1.update_traces(textposition='outside')
            fig1.update_layout(height=550)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("💰 Revenu Moyen par Région")
            reg_rev = df.groupby('REGION')['MNT_RECH'].mean().sort_values(ascending=True).reset_index()
            fig2 = px.bar(reg_rev, x='MNT_RECH', y='REGION', orientation='h',
                          color='MNT_RECH', color_continuous_scale='Teal',
                          labels={'MNT_RECH': 'Recharge Moy. (DT)'},
                          text_auto='.1f')
            fig2.update_traces(textposition='outside')
            fig2.update_layout(height=550)
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("📡 Adoption Data par Région")
            reg_data = df.groupby('REGION').agg(
                PCT_DATA=('TARGET_IA', 'mean'),
                MNT_DATA=('MNT_FORFAIT_DATA', 'mean')
            ).reset_index()
            reg_data['PCT_DATA'] = reg_data['PCT_DATA'] * 100
            fig3 = px.scatter(reg_data, x='PCT_DATA', y='MNT_DATA',
                              text='REGION', size='MNT_DATA',
                              color='PCT_DATA', color_continuous_scale='Viridis',
                              labels={'PCT_DATA': '% Clients Data (%)', 'MNT_DATA': 'Dépense Data Moy. (DT)'})
            fig3.update_traces(textposition='top center')
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.subheader("📊 Mix Handset par Région (Top 6)")
            top6_reg = df.groupby('REGION').size().nlargest(6).index
            df_top6 = df[df['REGION'].isin(top6_reg)]
            hs_reg = df_top6.groupby(['REGION', 'HANDSET']).size().reset_index(name='Count')
            fig4 = px.bar(hs_reg, x='REGION', y='Count', color='HANDSET',
                          barmode='stack', text_auto=True,
                          color_discrete_map={'2G': '#ef4444', '3G': '#f97316', '4G': '#3b82f6', '5G': '#10b981'})
            st.plotly_chart(fig4, use_container_width=True)

        st.divider()
        st.subheader("⚠️ Clients à Risque (STATUT_RGS90 = R) par Région")
        risk = df[df['STATUT_RGS90'] == 'R'].groupby('REGION').size().reset_index(name='Clients_à_Risque')
        tot  = df.groupby('REGION').size().reset_index(name='Total')
        risk = risk.merge(tot, on='REGION')
        risk['Taux_Risque_%'] = (risk['Clients_à_Risque'] / risk['Total'] * 100).round(2)
        fig5 = px.bar(risk.sort_values('Taux_Risque_%', ascending=False),
                      x='REGION', y='Taux_Risque_%',
                      color='Taux_Risque_%', color_continuous_scale='Reds',
                      text_auto='.2f',
                      labels={'Taux_Risque_%': 'Taux Risque (%)'})
        st.plotly_chart(fig5, use_container_width=True)

# ============================================================
# PAGE 4 : ANALYSE USAGE
# ============================================================
elif nav == "📞 Analyse Usage":
    st.title("📞 Analyse de l'Usage – Voix, SMS & Data")

    if df.empty:
        st.warning("Données manquantes.")
    else:
        tabs = st.tabs(["📞 Voix", "💬 SMS", "📡 Data", "🔀 Opérateurs"])

        # --- Voix ---
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribution Durée Voix Sortante")
                df_v = df[df['DUREE_APPEL_TOT'] > 0]
                fig = px.histogram(df_v, x='DUREE_APPEL_TOT', nbins=50,
                                   color_discrete_sequence=['#004a99'],
                                   labels={'DUREE_APPEL_TOT': 'Durée (min)'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Durée Voix : Entrant vs Sortant")
                d_voix = pd.DataFrame({
                    'Type': ['Sortant', 'Entrant'],
                    'Durée Totale': [df['DUREE_APPEL_TOT'].sum(), df['DUREE_APPEL_IN'].sum()]
                })
                fig2 = px.pie(d_voix, names='Type', values='Durée Totale', hole=0.45,
                              color_discrete_sequence=['#004a99', '#10b981'])
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Durée Voix Moyenne par Ancienneté & Handset")
            voice_anc = df.groupby(['TRANCHE_ANC', 'HANDSET'])['DUREE_APPEL_TOT'].mean().reset_index()
            fig3 = px.bar(voice_anc, x='TRANCHE_ANC', y='DUREE_APPEL_TOT', color='HANDSET',
                          barmode='group', color_discrete_map={'2G': '#ef4444', '3G': '#f97316',
                                                               '4G': '#3b82f6', '5G': '#10b981'},
                          labels={'DUREE_APPEL_TOT': 'Durée Moy. (min)', 'TRANCHE_ANC': 'Ancienneté'})
            st.plotly_chart(fig3, use_container_width=True)

        # --- SMS ---
        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("SMS Sortants par Catégorie d'Offre")
                sms_offre = df.groupby('OFFRE_CAT')['NB_SMS_TOT'].mean().sort_values(ascending=False).reset_index()
                fig = px.bar(sms_offre, x='OFFRE_CAT', y='NB_SMS_TOT',
                             color='NB_SMS_TOT', color_continuous_scale='Purp',
                             labels={'NB_SMS_TOT': 'SMS Moy.', 'OFFRE_CAT': 'Offre'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("SMS Entrant vs Sortant")
                sms_df = pd.DataFrame({
                    'Type': ['SMS Sortant', 'SMS Entrant'],
                    'Total': [df['NB_SMS_TOT'].sum(), df['NB_SMS_IN'].sum()]
                })
                fig2 = px.pie(sms_df, names='Type', values='Total', hole=0.45,
                              color_discrete_sequence=['#8b5cf6', '#f59e0b'])
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Corrélation SMS ↔ Voix")
            fig3 = px.density_heatmap(df[df['DUREE_APPEL_TOT'] < df['DUREE_APPEL_TOT'].quantile(0.95)],
                                      x='DUREE_APPEL_TOT', y='NB_SMS_TOT',
                                      nbinsx=30, nbinsy=30, color_continuous_scale='Magma',
                                      labels={'DUREE_APPEL_TOT': 'Durée Voix (min)', 'NB_SMS_TOT': 'Nb SMS'})
            st.plotly_chart(fig3, use_container_width=True)

        # --- Data ---
        with tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💸 Distribution Dépense Data")
                df_d = df[df['MNT_FORFAIT_DATA'] > 0]
                fig = px.histogram(df_d, x='MNT_FORFAIT_DATA', nbins=50,
                                   color_discrete_sequence=['#10b981'],
                                   labels={'MNT_FORFAIT_DATA': 'Dépense Data (DT)'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 Nb Forfaits Data par Catégorie d'Offre")
                data_offre = df.groupby('OFFRE_CAT')['NB_FORFAIT_DATA'].mean().sort_values(ascending=False).reset_index()
                fig2 = px.bar(data_offre, x='NB_FORFAIT_DATA', y='OFFRE_CAT', orientation='h',
                              color='NB_FORFAIT_DATA', color_continuous_scale='Greens',
                              labels={'NB_FORFAIT_DATA': 'Nb Forfaits Moy.', 'OFFRE_CAT': 'Offre'})
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("📱 Dépense Data Moyenne par Handset & Ancienneté")
            data_hs = df.groupby(['HANDSET', 'TRANCHE_ANC'])['MNT_FORFAIT_DATA'].mean().reset_index()
            fig3 = px.bar(data_hs, x='HANDSET', y='MNT_FORFAIT_DATA', color='TRANCHE_ANC',
                          barmode='group', color_discrete_sequence=px.colors.qualitative.Set2,
                          labels={'MNT_FORFAIT_DATA': 'Dépense Data Moy. (DT)', 'HANDSET': 'Type Réseau'})
            st.plotly_chart(fig3, use_container_width=True)

        # --- Opérateurs ---
        with tabs[3]:
            st.subheader("🔀 Répartition des Appels Sortants par Opérateur")
            op_data = pd.DataFrame({
                'Opérateur': ['Tunisie Telecom (Onnet)', 'Ooredoo', 'Orange', 'Autres Offnet'],
                'Durée Totale': [
                    df['DUREE_ONNET_TOT'].sum(),
                    df['DUREE_APPEL_OOREDOO_TOT'].sum(),
                    df['DUREE_APPEL_ORANGE_TOT'].sum(),
                    (df['DUREE_OFFNET_TOT'] - df['DUREE_APPEL_OOREDOO_TOT'] - df['DUREE_APPEL_ORANGE_TOT']).clip(lower=0).sum()
                ]
            })
            fig = px.pie(op_data, names='Opérateur', values='Durée Totale', hole=0.5,
                         color_discrete_sequence=['#004a99', '#e00717', '#ff6b00', '#6b7280'])
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Durée Onnet vs Offnet – par Handset")
                op_hs = df.groupby('HANDSET').agg(
                    Onnet=('DUREE_ONNET_TOT', 'mean'),
                    Offnet=('DUREE_OFFNET_TOT', 'mean')
                ).reset_index().melt(id_vars='HANDSET', var_name='Type', value_name='Durée')
                fig2 = px.bar(op_hs, x='HANDSET', y='Durée', color='Type',
                              barmode='group', color_discrete_sequence=['#004a99', '#f97316'])
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                st.subheader("Appels Entrants par Opérateur Source")
                op_in = pd.DataFrame({
                    'Opérateur': ['Tunisie Telecom', 'Ooredoo', 'Orange'],
                    'Durée': [
                        df['DUREE_APPEL_IN'].sum() - df['DUREE_OOREDOO_IN'].sum() - df['DUREE_ORANGE_IN'].sum(),
                        df['DUREE_OOREDOO_IN'].sum(),
                        df['DUREE_ORANGE_IN'].sum()
                    ]
                })
                fig3 = px.pie(op_in, names='Opérateur', values='Durée', hole=0.45,
                              color_discrete_sequence=['#004a99', '#e00717', '#ff6b00'])
                st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# PAGE 5 : SEGMENTATION
# ============================================================
elif nav == "🧩 Segmentation Clients":
    st.title("🧩 Segmentation Comportementale – K-Means (4 clusters)")

    if df.empty:
        st.warning("Données manquantes.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Scatter : Recharge vs Durée Voix")
            fig = px.scatter(df, x='DUREE_APPEL_TOT', y='MNT_RECH',
                             color='PROFIL', size='MNT_FORFAIT_DATA',
                             hover_name=id_col, opacity=0.6,
                             labels={'DUREE_APPEL_TOT': 'Durée Voix (min)', 'MNT_RECH': 'Recharge (DT)'},
                             color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Répartition des Profils")
            fig2 = px.pie(df, names='PROFIL', hole=0.45,
                          color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("📊 Caractéristiques Moyennes par Profil")
        profile_stats = df.groupby('PROFIL').agg(
            Abonnés=('ID', 'count'),
            Recharge_Moy=('MNT_RECH', 'mean'),
            Durée_Voix_Moy=('DUREE_APPEL_TOT', 'mean'),
            SMS_Moy=('NB_SMS_TOT', 'mean'),
            Data_Moy=('MNT_FORFAIT_DATA', 'mean'),
            Ancienneté_Moy=('ANC_M', 'mean'),
            Pct_Data=('TARGET_IA', 'mean')
        ).round(1).reset_index()
        profile_stats['Pct_Data'] = (profile_stats['Pct_Data'] * 100).round(1).astype(str) + '%'
        st.dataframe(profile_stats, use_container_width=True)

        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Mix Handset par Profil")
            hs_profil = df.groupby(['PROFIL', 'HANDSET']).size().reset_index(name='Count')
            fig3 = px.bar(hs_profil, x='PROFIL', y='Count', color='HANDSET',
                          barmode='stack',
                          color_discrete_map={'2G': '#ef4444', '3G': '#f97316', '4G': '#3b82f6', '5G': '#10b981'})
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.subheader("Ancienneté par Profil")
            fig4 = px.box(df, x='PROFIL', y='ANC_M', color='PROFIL',
                          color_discrete_sequence=px.colors.qualitative.Bold,
                          labels={'ANC_M': 'Ancienneté (mois)'})
            st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# PAGE 6 : ÉVALUATION IA
# ============================================================
elif nav == "🧠 Évaluation IA":
    st.title("🧠 Performance du Modèle XGBoost")

    if df.empty:
        st.warning("Données manquantes.")
    else:
        # Bandeau source du modèle
        if MODEL_SOURCE == "joblib":
            st.success(f"✅ Modèle chargé depuis **{MODEL_PATH}** — métriques issues de l'entraînement réel sur l'ABT complète.")
        else:
            st.warning("⚠️ Modèle de substitution (RandomForest). Lancez `train_model_fixed.py` pour utiliser le vrai XGBoost.")

        c1, c2, c3 = st.columns(3)
        c1.metric("📈 AUC Score (Test)", f"{auc_score:.4f}")
        c2.metric("🎯 Clients Data (TARGET=1)", f"{(df['TARGET_IA']==1).sum():,}")
        c3.metric("📊 Features utilisées", f"{len(model_features)}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if MODEL_SOURCE == "joblib":
                st.subheader("📊 Métriques enregistrées à l'entraînement")
                metrics_display = artifact['metrics']
                m_df = pd.DataFrame([
                    {"Métrique": "AUC Test",         "Valeur": f"{metrics_display['auc_test']:.4f}"},
                    {"Métrique": "Average Precision", "Valeur": f"{metrics_display['ap_score']:.4f}"},
                    {"Métrique": "AUC CV Moyenne",   "Valeur": f"{metrics_display['auc_cv_mean']:.4f}"},
                    {"Métrique": "AUC CV Std",       "Valeur": f"± {metrics_display['auc_cv_std']:.4f}"},
                ])
                st.dataframe(m_df, use_container_width=True, hide_index=True)
                st.info("💡 La matrice de confusion détaillée est dans `outputs/model_report/rapport_evaluation_xgboost.png`")
            else:
                st.subheader("🧮 Matrice de Confusion")
                fig_cm = px.imshow(conf_matrix, text_auto=True,
                                   x=['Non-Data Prédit', 'Data Prédit'],
                                   y=['Non-Data Réel', 'Data Réel'],
                                   color_continuous_scale='Blues', aspect='auto')
                st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            st.subheader("📌 Importance des Variables")
            imp = pd.DataFrame({
                'Variable': model_features,
                'Impact': feat_importances
            }).sort_values('Impact')
            fig2 = px.bar(imp, x='Impact', y='Variable', orientation='h',
                          color='Impact', color_continuous_scale='Reds', text_auto='.3f')
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("📊 Distribution du Score de Propension Data")
        fig3 = px.histogram(df, x='PROBA_DATA', color='TARGET_IA',
                            nbins=40, barmode='overlay', opacity=0.7,
                            color_discrete_map={0: '#ef4444', 1: '#10b981'},
                            labels={'PROBA_DATA': 'Score Data', 'TARGET_IA': 'Réel (0=Non-Data, 1=Data)'},
                            category_orders={'TARGET_IA': [0, 1]})
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("🎯 Clients Ciblables par Seuil de Score")
        seuil = st.slider("Seuil de propension (%)", 30, 90, 60)
        n_cibles = (df['PROBA_DATA'] >= seuil / 100).sum()
        st.metric(f"Clients avec score ≥ {seuil}%", f"{n_cibles:,}",
                  f"{n_cibles/len(df)*100:.1f}% du parc")

# ============================================================
# PAGE 7 : PRÉDICTION INDIVIDUELLE
# ============================================================
elif nav == "🔮 Prédiction Individuelle":
    st.title("🔮 Scoring & Recommandation Marketing Individuelle")

    if df.empty:
        st.warning("Données manquantes.")
    else:
        # PROBA_DATA est déjà calculé au chargement via le vrai modèle .joblib
        col_in, col_top = st.columns([1, 2])
        with col_in:
            sid = st.text_input(f"Saisir l'identifiant client ({id_col}) :").strip().upper()
            if MODEL_SOURCE == "joblib":
                st.caption("🟢 Scores calculés par le modèle XGBoost réel")
            else:
                st.caption("🟡 Scores estimés (modèle de substitution)")

        with col_top:
            st.subheader("🏆 Top 10 Cibles Prioritaires")
            top10 = df.nlargest(10, 'PROBA_DATA')[
                [c for c in [id_col, 'REGION', 'HANDSET', 'OFFRE_CAT', 'ANC_M', 'PROBA_DATA']
                 if c in df.columns]
            ].copy()
            top10['PROBA_DATA'] = (top10['PROBA_DATA'] * 100).round(1).astype(str) + '%'
            st.dataframe(top10, use_container_width=True, hide_index=True)

        if sid:
            client = df[df[id_col] == sid]
            if not client.empty:
                prob = float(client['PROBA_DATA'].values[0])
                st.divider()

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("🎯 Score Propension", f"{prob*100:.1f}%")
                r2.metric("🏷️ Profil Cluster",   client['PROFIL'].values[0])
                r3.metric("📱 Réseau",            str(client['HANDSET'].values[0]))
                r4.metric("🕒 Ancienneté",        f"{int(client['ANC_M'].values[0])} mois")

                # Jauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Score de Propension DATA (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar':  {'color': "#004a99"},
                        'steps': [
                            {'range': [0,  45], 'color': "#fee2e2"},
                            {'range': [45, 75], 'color': "#fef9c3"},
                            {'range': [75,100], 'color': "#dcfce7"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 75
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Recommandation
                st.subheader("🤖 Action Marketing Recommandée")
                if prob >= 0.75:
                    st.success("🎯 **CIBLE PRIORITAIRE** — Score ≥ 75%\n\n"
                               "Déployer une offre Forfait DATA illimité. "
                               "Contacter via push SMS / notification app en priorité.")
                elif prob >= 0.45:
                    st.warning("⚡ **CIBLE POTENTIELLE** — Score entre 45% et 75%\n\n"
                               "Proposer un pack Combo Voix + Data. "
                               "Offre d'essai 1 mois avec remise de bienvenue.")
                else:
                    st.error("🧊 **CIBLE VOIX** — Score < 45%\n\n"
                             "Client peu enclin à la Data. "
                             "Maintenir sur bonus minutes, éviter le push Data.")

                # Fiche client — noms de colonnes adaptés à l'ABT
                st.subheader("📋 Fiche Client Détaillée")
                def get_val(col_candidates, default=0):
                    """Essaie plusieurs noms de colonnes possibles (ABT vs dashboard)."""
                    for c in col_candidates:
                        if c in client.columns:
                            return client[c].values[0]
                    return default

                fiche = {
                    "Région":                get_val(['REGION']),
                    "Offre":                 get_val(['OFFRE']),
                    "Catégorie Offre":       get_val(['OFFRE_CAT']),
                    "Statut":                get_val(['STATUT']),
                    "Type Réseau":           get_val(['HANDSET']),
                    "Ancienneté (mois)":     int(get_val(['ANC_M'])),
                    "Recharge Totale (DT)":  round(float(get_val(['MNT_RECH_TOT', 'MNT_RECH'])), 2),
                    "Recharge Moyenne (DT)": round(float(get_val(['MNT_RECH_MOY'])), 2),
                    "Durée Voix (min)":      round(float(get_val(['DUREE_APPEL_TOT'])), 1),
                    "Nb Appels":             int(get_val(['NB_APPEL_TOT'])),
                    "Nb SMS Sortants":       round(float(get_val(['NB_SMS_TOT'])), 0),
                    "Dépense Data (DT)":     round(float(get_val(['MNT_FORFAIT_DATA_TOT', 'MNT_FORFAIT_DATA'])), 2),
                    "Nb Forfaits Data":      int(get_val(['NB_FORFAIT_DATA_TOT', 'NB_FORFAIT_DATA'])),
                    "Revenu CDR (DT)":       round(float(get_val(['REVENU_CDR_TOT', 'REVENU_CDR'])), 2),
                }
                fiche_df = pd.DataFrame(list(fiche.items()), columns=['Indicateur', 'Valeur'])
                st.table(fiche_df)

            else:
                st.error(f"Client '{sid}' introuvable dans la base.")


# ============================================================
# PAGE 8 : ANALYSE RFM
# ============================================================
elif nav == "📊 Analyse RFM":
    st.title("📊 Analyse RFM — Segmentation Comportementale")
    st.markdown("La méthode **RFM** (Recency · Frequency · Monetary) segmente les clients selon leur comportement de recharge réel.")

    import os as _os
    RFM_CSV = "outputs/rfm_segments.csv"
    if not _os.path.exists(RFM_CSV):
        st.warning("⚠️ Fichier RFM introuvable. Lancez d'abord :")
        st.code("python rfm_analysis.py", language="bash")
    else:
        df_rfm = pd.read_csv(RFM_CSV)
        df_rfm.columns = df_rfm.columns.str.strip().str.upper()
        PALETTE_RFM = {
            "Champions": "#10b981", "Clients Fidèles": "#3b82f6",
            "Clients Potentiels": "#0ea5e9", "Nouveaux Clients": "#8b5cf6",
            "Clients Ordinaires": "#f59e0b", "Clients à Risque": "#f97316",
            "Clients Perdus": "#ef4444",
        }
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Clients analysés", f"{len(df_rfm):,}")
        c2.metric("🏆 Champions", f"{(df_rfm['SEGMENT_RFM']=='Champions').sum():,}")
        c3.metric("⚠️ À Risque", f"{(df_rfm['SEGMENT_RFM']=='Clients à Risque').sum():,}")
        c4.metric("🔴 Perdus", f"{(df_rfm['SEGMENT_RFM']=='Clients Perdus').sum():,}")
        st.divider()
        # Graphiques
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Répartition des Segments RFM")
            seg_cnt = df_rfm['SEGMENT_RFM'].value_counts().reset_index()
            seg_cnt.columns = ['Segment', 'Nb Clients']
            fig_r1 = px.pie(seg_cnt, names='Segment', values='Nb Clients',
                            hole=0.45, color='Segment', color_discrete_map=PALETTE_RFM)
            st.plotly_chart(fig_r1, use_container_width=True)
        with col2:
            st.subheader("Revenu Moyen par Segment (DT)")
            rev_seg = df_rfm.groupby('SEGMENT_RFM')['MONETARY'].mean().sort_values(ascending=True).reset_index()
            fig_r2 = px.bar(rev_seg, x='MONETARY', y='SEGMENT_RFM', orientation='h',
                            color='SEGMENT_RFM', color_discrete_map=PALETTE_RFM,
                            labels={'MONETARY': 'Recharge Moy. (DT)', 'SEGMENT_RFM': 'Segment'},
                            text_auto='.0f')
            fig_r2.update_layout(showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Score RFM Moyen par Segment")
            score_seg = df_rfm.groupby('SEGMENT_RFM')['RFM_SCORE'].mean().sort_values(ascending=True).reset_index()
            fig_r3 = px.bar(score_seg, x='RFM_SCORE', y='SEGMENT_RFM', orientation='h',
                            color='SEGMENT_RFM', color_discrete_map=PALETTE_RFM,
                            labels={'RFM_SCORE': 'Score RFM (3-15)'}, text_auto='.1f')
            fig_r3.update_layout(showlegend=False)
            st.plotly_chart(fig_r3, use_container_width=True)
        with col4:
            st.subheader("Taux Appétence DATA par Segment")
            if 'TARGET_DATA' in df_rfm.columns:
                data_seg = df_rfm.groupby('SEGMENT_RFM')['TARGET_DATA'].mean().sort_values(ascending=True).reset_index()
                data_seg['Taux DATA (%)'] = (data_seg['TARGET_DATA'] * 100).round(1)
                fig_r4 = px.bar(data_seg, x='Taux DATA (%)', y='SEGMENT_RFM',
                                orientation='h', color='SEGMENT_RFM',
                                color_discrete_map=PALETTE_RFM, text_auto='.1f')
                fig_r4.update_layout(showlegend=False)
                st.plotly_chart(fig_r4, use_container_width=True)
        st.divider()
        st.subheader("📋 Caractéristiques Moyennes par Segment")
        summary_rfm = df_rfm.groupby('SEGMENT_RFM').agg(
            Clients=('ID', 'count'),
            Recency_Moy=('RECENCY', 'mean'),
            Frequency_Moy=('FREQUENCY', 'mean'),
            Monetary_Moy=('MONETARY', 'mean'),
            Score_RFM=('RFM_SCORE', 'mean')
        ).round(1).reset_index()
        st.dataframe(summary_rfm, use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("💡 Recommandations Marketing par Segment")
        recos_rfm = {
            "Champions":         ("🏆", "#10b981", "Récompenser la fidélité — offres VIP, programme de parrainage"),
            "Clients Fidèles":   ("🎁", "#3b82f6", "Upselling — forfaits DATA premium ou offres groupées"),
            "Clients Potentiels":("⚡", "#0ea5e9", "Nurturing — campagnes ciblées pour convertir en Champions"),
            "Nouveaux Clients":  ("👋", "#8b5cf6", "Onboarding — offre bienvenue DATA, accompagnement personnalisé"),
            "Clients Ordinaires":("📊", "#f59e0b", "Engagement — promotions flash, bonus recharge pour augmenter fréquence"),
            "Clients à Risque":  ("⚠️", "#f97316", "Rétention URGENTE — offre fidélisation, appel proactif immédiat"),
            "Clients Perdus":    ("🔴", "#ef4444", "Réactivation — campagne win-back avec remise exceptionnelle"),
        }
        for seg, (icon, color, reco) in recos_rfm.items():
            n = (df_rfm['SEGMENT_RFM'] == seg).sum()
            st.markdown(
                f'<div style="background:{color}18;border-left:4px solid {color};'
                f'padding:10px 16px;margin:6px 0;border-radius:4px;">'
                f'<b>{icon} {seg}</b> <span style="color:#666">({n:,} clients)</span><br>'
                f'<span style="font-size:0.9em">{reco}</span></div>',
                unsafe_allow_html=True)

# ============================================================
# PAGE 9 : RISQUE DE CHURN
# ============================================================
elif nav == "⚠️  Risque de Churn":
    st.title("⚠️ Analyse du Risque de Churn (Attrition)")
    st.markdown("Identification des clients à risque de désabonnement via le modèle **XGBoost de churn**.")

    import os as _os2
    CHURN_CSV = "outputs/churn_scoring.csv"
    if not _os2.path.exists(CHURN_CSV):
        st.warning("⚠️ Fichier churn introuvable. Lancez d'abord :")
        st.code("python rfm_analysis.py\npython churn_model.py", language="bash")
    else:
        df_ch = pd.read_csv(CHURN_CSV)
        df_ch.columns = df_ch.columns.str.strip().str.upper()
        if 'SCORE_CHURN' in df_ch.columns:
            # Seuils dynamiques basés sur les percentiles réels des scores
            # Évite le problème de "Risque Élevé = 0" quand les scores sont concentrés
            p75 = df_ch['SCORE_CHURN'].quantile(0.75)  # top 25% = Risque Élevé
            p50 = df_ch['SCORE_CHURN'].quantile(0.50)  # top 50% = Risque Moyen
            # Garantir que les seuils sont distincts
            seuil_eleve = max(p75, df_ch['SCORE_CHURN'].min() + 0.001)
            seuil_moyen = max(p50, df_ch['SCORE_CHURN'].min())

            n_el  = (df_ch['SCORE_CHURN'] >= seuil_eleve).sum()
            n_mo  = ((df_ch['SCORE_CHURN'] >= seuil_moyen) & (df_ch['SCORE_CHURN'] < seuil_eleve)).sum()
            taux  = df_ch['CHURN'].mean() * 100 if 'CHURN' in df_ch.columns else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("👥 Clients analysés", f"{len(df_ch):,}")
            c2.metric("🔴 Risque Élevé (top 25%)",  f"{n_el:,}", f"{n_el/len(df_ch)*100:.1f}% | seuil ≥ {seuil_eleve:.2f}")
            c3.metric("🟡 Risque Moyen (top 50%)",  f"{n_mo:,}", f"{n_mo/len(df_ch)*100:.1f}% | seuil ≥ {seuil_moyen:.2f}")
            c4.metric("📊 Taux Churn Réel",  f"{taux:.2f}%")
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribution des Scores de Risque Churn")
                fig_ch1 = px.histogram(df_ch, x='SCORE_CHURN', nbins=40,
                                       color_discrete_sequence=['#ef4444'],
                                       labels={'SCORE_CHURN': 'Score Risque Churn'})
                fig_ch1.add_vline(x=float(seuil_moyen), line_dash="dash", line_color="#f97316",
                                  annotation_text=f"Seuil Moyen (P50={seuil_moyen:.2f})")
                fig_ch1.add_vline(x=float(seuil_eleve), line_dash="dash", line_color="#ef4444",
                                  annotation_text=f"Seuil Élevé (P75={seuil_eleve:.2f})")
                st.plotly_chart(fig_ch1, use_container_width=True)
            with col2:
                if 'RISQUE_CHURN' in df_ch.columns:
                    st.subheader("Répartition par Niveau de Risque")
                    rc = df_ch['RISQUE_CHURN'].value_counts().reset_index()
                    rc.columns = ['Risque', 'Nb']
                    fig_ch2 = px.pie(rc, names='Risque', values='Nb', hole=0.45,
                                     color='Risque',
                                     color_discrete_map={
                                         '🔴 Élevé': '#ef4444',
                                         '🟡 Moyen': '#f59e0b',
                                         '🟢 Faible': '#10b981'})
                    st.plotly_chart(fig_ch2, use_container_width=True)
            st.divider()
            col3, col4 = st.columns(2)
            with col3:
                if 'SEGMENT_RFM' in df_ch.columns and 'CHURN' in df_ch.columns:
                    st.subheader("Taux Churn par Segment RFM")
                    cs = df_ch.groupby('SEGMENT_RFM')['CHURN'].mean().sort_values(ascending=True).reset_index()
                    cs['Taux (%)'] = (cs['CHURN'] * 100).round(2)
                    fig_ch3 = px.bar(cs, x='Taux (%)', y='SEGMENT_RFM', orientation='h',
                                     color='Taux (%)', color_continuous_scale='Reds',
                                     text_auto='.2f')
                    st.plotly_chart(fig_ch3, use_container_width=True)
            with col4:
                if 'HANDSET' in df_ch.columns and 'CHURN' in df_ch.columns:
                    st.subheader("Taux Churn par Type Réseau")
                    ch_hs = df_ch[df_ch['HANDSET'] != 'Inconnu'].groupby(
                        'HANDSET')['CHURN'].mean().sort_values(ascending=False).reset_index()
                    ch_hs['Taux (%)'] = (ch_hs['CHURN'] * 100).round(2)
                    fig_ch4 = px.bar(ch_hs, x='HANDSET', y='Taux (%)',
                                     color='Taux (%)', color_continuous_scale='Reds',
                                     text_auto='.2f',
                                     labels={'HANDSET': 'Type Réseau'})
                    st.plotly_chart(fig_ch4, use_container_width=True)
            st.divider()
            seuil_ch = st.slider("🎯 Seuil de risque churn (%)", 1, 99,
                                    int(seuil_eleve * 100))
            at_risk = df_ch[df_ch['SCORE_CHURN'] >= seuil_ch / 100]
            st.metric(f"Clients avec score ≥ {seuil_ch}%",
                      f"{len(at_risk):,}",
                      f"{len(at_risk)/len(df_ch)*100:.1f}% du parc")
            st.subheader(f"🚨 Top 15 Clients à Risque (Score ≥ {seuil_ch}%)")
            disp_cols = [c for c in ['ID', 'SEGMENT_RFM', 'HANDSET', 'OFFRE_CAT',
                                      'RECENCY', 'FREQUENCY', 'MONETARY',
                                      'SCORE_CHURN', 'ACTION_RETENTION']
                         if c in df_ch.columns]
            top15_ch = at_risk.head(15)[disp_cols].copy()
            if 'SCORE_CHURN' in top15_ch.columns:
                top15_ch['SCORE_CHURN'] = (top15_ch['SCORE_CHURN'] * 100).round(1).astype(str) + '%'
            st.dataframe(top15_ch, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 10 : MODÈLES CV — Résultats Cross-Validation
# ============================================================
elif nav == "📈 Modèles CV":
    st.title("📈 Résultats de la Validation Croisée — 2 Modèles")
    st.markdown("Entraînement par **Cross-Validation Stratifiée 5-Fold** sur le fichier `CIBLE_ECH_DEC_2025_6.xlsx`")

    cv_models = load_cv_models()

    if cv_models["forfait"] is None and cv_models["churn"] is None:
        st.warning("⚠️ Modèles CV introuvables. Lancez d'abord :")
        st.code("python train_models_cv.py", language="bash")
    else:
        # ── Explication de la méthode ─────────────────────────────────
        st.info("""
        **Principe de la Cross-Validation Stratifiée 5-Fold :**

        Les données sont divisées en **5 partitions (folds)** de taille égale.
        À chaque itération, 4 folds servent à l'entraînement et 1 fold sert au test.
        Ce processus se répète 5 fois — chaque fold joue le rôle de jeu de test une fois.
        Les métriques finales sont la **moyenne des 5 évaluations**, ce qui garantit
        une estimation robuste et non biaisée des performances réelles.
        """)

        st.divider()

        # ── Métriques des 2 modèles ───────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Modèle 1 — Activation Forfait DATA")
            st.caption("Variable cible : TARGET (0 = pas de forfait, 1 = forfait activé)")
            if cv_models["forfait"]:
                m = cv_models["forfait"]["metrics"]
                c1,c2,c3 = st.columns(3)
                c1.metric("AUC Moyenne", f"{m['auc_cv_mean']:.4f}")
                c2.metric("AUC Std",     f"± {m['auc_cv_std']:.4f}")
                c3.metric("AP Score",    f"{m['ap_cv_mean']:.4f}")

                # Tableau des AUC par fold
                fold_data = pd.DataFrame({
                    'Fold'     : [f"Fold {i+1}" for i in range(m['n_folds'])],
                    'AUC ROC'  : m['fold_aucs'],
                    'Ecart/Moy': [round(a - m['auc_cv_mean'], 4) for a in m['fold_aucs']],
                })
                st.dataframe(fold_data, use_container_width=True, hide_index=True)

                # Graphique barres AUC par fold
                fig1 = px.bar(fold_data, x='Fold', y='AUC ROC',
                              color='AUC ROC', color_continuous_scale='Blues',
                              text='AUC ROC', title="AUC par Fold — Modèle Forfait")
                fig1.add_hline(y=m['auc_cv_mean'], line_dash="dash",
                               annotation_text=f"Moy={m['auc_cv_mean']:.4f}")
                fig1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig1.update_layout(yaxis_range=[max(0, min(m['fold_aucs'])-0.05), 1.0])
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("Modèle Forfait non trouvé.")

        with col2:
            st.subheader("⚠️ Modèle 2 — Prédiction Churn")
            st.caption("Variable cible : FLAG_CHURN (0 = stable, 1 = risque attrition)")
            if cv_models["churn"]:
                m = cv_models["churn"]["metrics"]
                c1,c2,c3 = st.columns(3)
                c1.metric("AUC Moyenne", f"{m['auc_cv_mean']:.4f}")
                c2.metric("AUC Std",     f"± {m['auc_cv_std']:.4f}")
                c3.metric("AP Score",    f"{m['ap_cv_mean']:.4f}")

                fold_data2 = pd.DataFrame({
                    'Fold'     : [f"Fold {i+1}" for i in range(m['n_folds'])],
                    'AUC ROC'  : m['fold_aucs'],
                    'Ecart/Moy': [round(a - m['auc_cv_mean'], 4) for a in m['fold_aucs']],
                })
                st.dataframe(fold_data2, use_container_width=True, hide_index=True)

                fig2 = px.bar(fold_data2, x='Fold', y='AUC ROC',
                              color='AUC ROC', color_continuous_scale='Reds',
                              text='AUC ROC', title="AUC par Fold — Modèle Churn")
                fig2.add_hline(y=m['auc_cv_mean'], line_dash="dash",
                               annotation_text=f"Moy={m['auc_cv_mean']:.4f}")
                fig2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig2.update_layout(yaxis_range=[max(0, min(m['fold_aucs'])-0.05), 1.0])
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Modèle Churn non trouvé.")

        # ── Comparaison des deux modèles ──────────────────────────────
        st.divider()
        st.subheader("⚖️ Comparaison des 2 Modèles")
        if cv_models["forfait"] and cv_models["churn"]:
            comp = pd.DataFrame({
                'Modèle'     : ['Activation Forfait DATA', 'Prédiction Churn'],
                'Cible'      : ['TARGET (0/1)', 'FLAG_CHURN (0/1)'],
                'AUC Moyenne': [cv_models["forfait"]["metrics"]["auc_cv_mean"],
                                cv_models["churn"]["metrics"]["auc_cv_mean"]],
                'AUC Std'    : [cv_models["forfait"]["metrics"]["auc_cv_std"],
                                cv_models["churn"]["metrics"]["auc_cv_std"]],
                'AP Score'   : [cv_models["forfait"]["metrics"]["ap_cv_mean"],
                                cv_models["churn"]["metrics"]["ap_cv_mean"]],
                'N Folds'    : [cv_models["forfait"]["metrics"]["n_folds"],
                                cv_models["churn"]["metrics"]["n_folds"]],
            })
            st.dataframe(comp, use_container_width=True, hide_index=True)

# ============================================================
# PAGE 11 : SCORING CV — Prédiction Individuelle (nouveaux modèles)
# ============================================================
elif nav == "🎯 Scoring CV":
    st.title("🎯 Scoring CV — Prédiction par les Modèles Cross-Validation")
    st.markdown("Scoring individuel utilisant les modèles entraînés sur `CIBLE_ECH_DEC_2025_6.xlsx`")

    cv_models = load_cv_models()

    if cv_models["forfait"] is None:
        st.warning("⚠️ Modèles CV introuvables. Lancez : python train_models_cv.py")
    else:
        # Charger le fichier CIBLE
        import os
        # Détection automatique du fichier CIBLE
        import glob as _g
        _candidates = (
            _g.glob("data/CIBLE*.xlsx") +
            _g.glob("data/cible*.xlsx") +
            _g.glob("data/ECH*.xlsx") +
            _g.glob("data/ech*.xlsx") +
            _g.glob("data/*.xlsx")
        )
        CIBLE_PATH = _candidates[0] if _candidates else None

        if CIBLE_PATH is None:
            all_files = _g.glob("data/*")
            st.error(
                f"Aucun fichier Excel trouvé dans le dossier data/. "
                f"Fichiers présents : {all_files if all_files else 'dossier vide'}"
            )
            st.info("Placez le fichier CIBLE_ECH_DEC_2025_6.xlsx dans le dossier data/")
        else:
            st.caption(f"Fichier détecté : `{CIBLE_PATH}`")
            df_cible = pd.read_excel(CIBLE_PATH)
            df_cible.columns = df_cible.columns.str.strip().str.upper()

            # Scorer tout le fichier CIBLE
            art_f = cv_models["forfait"]
            art_c = cv_models["churn"]
            X_cible = prepare_features_cv(df_cible,
                                          art_f['encoders'],
                                          art_f['feature_names'])
            df_cible['SCORE_FORFAIT'] = art_f['model'].predict_proba(X_cible)[:,1]
            df_cible['SCORE_CHURN']   = art_c['model'].predict_proba(X_cible)[:,1]

            p75 = df_cible['SCORE_CHURN'].quantile(0.75)
            p50 = df_cible['SCORE_CHURN'].quantile(0.50)

            df_cible['SEGMENT_FORFAIT'] = pd.cut(
                df_cible['SCORE_FORFAIT'],
                bins=[0, 0.40, 0.70, 1.0],
                labels=['Cible Voix','Cible Potentielle','Cible Prioritaire'],
                include_lowest=True)
            df_cible['RISQUE_CHURN'] = np.where(
                df_cible['SCORE_CHURN'] >= p75, 'Risque Eleve',
                np.where(df_cible['SCORE_CHURN'] >= p50, 'Risque Moyen', 'Risque Faible'))

            # KPIs
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Clients scorés", f"{len(df_cible):,}")
            n_prio = (df_cible['SEGMENT_FORFAIT']=='Cible Prioritaire').sum()
            c2.metric("Cible Prioritaire", f"{n_prio:,}",
                      f"{n_prio/len(df_cible)*100:.1f}% du parc")
            n_el = (df_cible['RISQUE_CHURN']=='Risque Eleve').sum()
            c3.metric("Risque Churn Elevé", f"{n_el:,}",
                      f"{n_el/len(df_cible)*100:.1f}% du parc")
            acc_t = (df_cible['TARGET']==1).mean()*100 if 'TARGET' in df_cible.columns else 0
            c4.metric("Taux Activation Réel", f"{acc_t:.1f}%")

            st.divider()

            # Distribution des scores
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Score Activation Forfait DATA")
                fig_f = px.histogram(df_cible, x='SCORE_FORFAIT', nbins=40,
                                     color_discrete_sequence=['#004a99'],
                                     labels={'SCORE_FORFAIT':'Score Forfait'})
                fig_f.add_vline(x=0.40, line_dash="dash", line_color="#f97316",
                                annotation_text="Seuil 40%")
                fig_f.add_vline(x=0.70, line_dash="dash", line_color="#10b981",
                                annotation_text="Seuil 70%")
                st.plotly_chart(fig_f, use_container_width=True)

            with col2:
                st.subheader("Score Risque Churn")
                fig_c = px.histogram(df_cible, x='SCORE_CHURN', nbins=40,
                                     color_discrete_sequence=['#ef4444'],
                                     labels={'SCORE_CHURN':'Score Churn'})
                fig_c.add_vline(x=float(p50), line_dash="dash",
                                annotation_text=f"P50={p50:.2f}")
                fig_c.add_vline(x=float(p75), line_dash="dash", line_color="#ef4444",
                                annotation_text=f"P75={p75:.2f}")
                st.plotly_chart(fig_c, use_container_width=True)

            st.divider()

            # Prédiction individuelle
            st.subheader("🔍 Prédiction Individuelle")
            sid = st.text_input("Saisir l'identifiant client (ID) :").strip().upper()
            if sid and 'ID' in df_cible.columns:
                client = df_cible[df_cible['ID'].astype(str).str.upper() == sid]
                if not client.empty:
                    score_f = float(client['SCORE_FORFAIT'].values[0])
                    score_c = float(client['SCORE_CHURN'].values[0])

                    r1,r2,r3,r4 = st.columns(4)
                    r1.metric("Score Forfait DATA", f"{score_f*100:.1f}%")
                    r2.metric("Segment", str(client['SEGMENT_FORFAIT'].values[0]))
                    r3.metric("Score Churn", f"{score_c*100:.1f}%")
                    r4.metric("Risque Churn", str(client['RISQUE_CHURN'].values[0]))

                    # Jauges
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_g1 = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score_f*100,
                            title={'text':"Score Activation Forfait (%)"},
                            gauge={'axis':{'range':[0,100]},
                                   'bar':{'color':'#004a99'},
                                   'steps':[
                                       {'range':[0,40],'color':'#fee2e2'},
                                       {'range':[40,70],'color':'#fef9c3'},
                                       {'range':[70,100],'color':'#dcfce7'}
                                   ]}))
                        fig_g1.update_layout(height=260)
                        st.plotly_chart(fig_g1, use_container_width=True)

                    with col_g2:
                        fig_g2 = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score_c*100,
                            title={'text':"Score Risque Churn (%)"},
                            gauge={'axis':{'range':[0,100]},
                                   'bar':{'color':'#ef4444'},
                                   'steps':[
                                       {'range':[0,50],'color':'#dcfce7'},
                                       {'range':[50,75],'color':'#fef9c3'},
                                       {'range':[75,100],'color':'#fee2e2'}
                                   ]}))
                        fig_g2.update_layout(height=260)
                        st.plotly_chart(fig_g2, use_container_width=True)

                    # Recommandation
                    if client['RISQUE_CHURN'].values[0] == 'Risque Eleve':
                        st.error("🚨 RÉTENTION URGENTE — Ce client est à très fort risque de churn. Appel proactif immédiat.")
                    elif score_f >= 0.70:
                        st.success("🎯 CIBLE PRIORITAIRE — Forte propension à activer un forfait DATA. Proposer offre illimitée.")
                    elif score_f >= 0.40:
                        st.warning("⚡ CIBLE POTENTIELLE — Propension modérée. Proposer un pack Combo Voix+Data.")
                    else:
                        st.info("📞 CIBLE VOIX — Peu de propension Data. Maintenir sur bonus minutes.")
                else:
                    st.error(f"Client '{sid}' introuvable.")

            # Top 15 prioritaires
            st.divider()
            st.subheader("🏆 Top 15 Cibles Prioritaires (Score Forfait le plus élevé)")
            cols_show = [c for c in ['ID','HANDSET','STATUT','ANC_M',
                                      'SCORE_FORFAIT','SEGMENT_FORFAIT',
                                      'SCORE_CHURN','RISQUE_CHURN'] if c in df_cible.columns]
            top15 = df_cible.nlargest(15,'SCORE_FORFAIT')[cols_show].copy()
            top15['SCORE_FORFAIT'] = (top15['SCORE_FORFAIT']*100).round(1).astype(str)+'%'
            top15['SCORE_CHURN']   = (top15['SCORE_CHURN']*100).round(1).astype(str)+'%'
            st.dataframe(top15, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 12 : MLOPS — Monitoring des modèles
# ============================================================
elif nav == "⚙️ MLOps":
    st.title("⚙️ MLOps — Monitoring et Versioning des Modèles")
    st.markdown("Suivi des expériences MLflow, détection de drift et historique des performances.")

    import os, json, glob

    # ── Statut des modèles ────────────────────────────────
    st.subheader("📦 Statut des Modèles en Production")
    models_info = [
        ("model_appetence_tt.joblib",  "Scoring Appétence DATA (ABT)"),
        ("model_forfait_cv.joblib",    "Activation Forfait DATA (CV)"),
        ("model_churn_cv.joblib",      "Prédiction Churn (CV)"),
    ]
    c1, c2, c3 = st.columns(3)
    for col, (path, label) in zip([c1,c2,c3], models_info):
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path) / 1024
            mtime = os.path.getmtime(path)
            import datetime
            date_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            col.success(f"✅ **{label}**")
            col.caption(f"Taille : {size:.0f} KB | Mis à jour : {date_str}")
        else:
            col.error(f"❌ **{label}**")
            col.caption("Introuvable — lancez le script d'entraînement")

    st.divider()

    # ── Historique des métriques ──────────────────────────
    METRICS_FILE = "outputs/mlops_metrics_history.json"
    st.subheader("📈 Historique des Métriques (Drift Monitoring)")

    if not os.path.exists(METRICS_FILE):
        st.warning("Aucun historique disponible. Lancez d'abord :")
        st.code("python train_models_cv_mlops.py", language="bash")
    else:
        with open(METRICS_FILE) as f:
            history = json.load(f)

        for model_name, data in history.items():
            last_auc = data.get('last_auc', 0)
            best_auc = data.get('best_auc', 0)
            delta    = last_auc - best_auc
            label    = model_name.replace("_", " ")

            if delta < -0.03:
                st.error(f"⚠️ **{label}** — DRIFT DÉTECTÉ : AUC {last_auc:.4f} "
                         f"(meilleur : {best_auc:.4f}, delta : {delta:+.4f})")
            elif delta < 0:
                st.warning(f"🟡 **{label}** — Légère dégradation : AUC {last_auc:.4f} "
                           f"(meilleur : {best_auc:.4f}, delta : {delta:+.4f})")
            else:
                st.success(f"✅ **{label}** — Stable : AUC {last_auc:.4f} "
                           f"(meilleur : {best_auc:.4f}, delta : {delta:+.4f})")

            col1, col2, col3 = st.columns(3)
            col1.metric("Dernier AUC",  f"{last_auc:.4f}")
            col2.metric("Meilleur AUC", f"{best_auc:.4f}")
            col3.metric("Variation",    f"{delta:+.4f}",
                        delta_color="normal" if delta >= 0 else "inverse")
            st.caption(f"Dernier run : {data.get('last_run_date','N/A')}")
            st.divider()

    # ── Rapport MLOps ─────────────────────────────────────
    MLOPS_REPORT = "outputs/mlops_report.txt"
    st.subheader("📋 Rapport MLOps Complet")
    if os.path.exists(MLOPS_REPORT):
        with open(MLOPS_REPORT, encoding='utf-8') as f:
            report_content = f.read()
        st.code(report_content, language="text")
    else:
        st.info("Rapport non disponible. Lancez train_models_cv_mlops.py")

    # ── Instructions MLflow UI ─────────────────────────────
    st.divider()
    st.subheader("🌐 Accéder à l'Interface MLflow")
    st.markdown("""
    Pour visualiser tous les runs, comparer les métriques et accéder
    aux artefacts, lancez l'interface web MLflow :
    """)
    st.code("mlflow ui", language="bash")
    st.info("Puis ouvrir dans le navigateur : **http://localhost:5000**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Ce que MLflow enregistre :**
        - AUC, AP Score, F1 par fold et en moyenne
        - Hyperparamètres XGBoost (max_depth, lr, etc.)
        - Le modèle entraîné avec sa signature
        - Les graphiques AUC par fold
        """)
    with col2:
        st.markdown("""
        **Ce que le Drift Monitor surveille :**
        - Comparaison AUC run actuel vs meilleur run
        - Alerte si dégradation > 3% (seuil configurable)
        - Historique JSON de tous les runs
        - Recommandation de réentraînement si drift détecté
        """)