"""
=============================================================
  analyse_descriptive_v2.py — Tunisie Telecom PFE
  Analyse Exploratoire des Données (EDA) Complète
  16 visualisations organisées en 5 sections thématiques
=============================================================

AMÉLIORATIONS PAR RAPPORT À L'ORIGINAL :
  [+] Section 1 : Qualité & Aperçu des données (manquants, types, outliers)
  [+] Section 2 : Profil du parc client (statut, handset, ancienneté, canal)
  [+] Section 3 : Analyse revenus & usage (recharge, voix, SMS, opérateurs)
  [+] Section 4 : Analyse géographique (top régions, revenu, taux data)
  [+] Section 5 : Analyse bivariée TARGET_DATA (corrélations, comparaisons)
  [+] Rapport texte complet avec statistiques clés
  [+] Export PNG + CSV nettoyé
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_PATH  = "data"
OUTPUT_PNG = "rapport_eda_final.png"
OUTPUT_CSV = "data_pfe_nettoye.csv"

PALETTE_TT   = ['#004a99', '#0ea5e9', '#10b981', '#f97316', '#8b5cf6', '#ef4444']
COLOR_POS    = '#10b981'   # Clients DATA
COLOR_NEG    = '#ef4444'   # Clients Non-DATA
COLOR_MAIN   = '#004a99'   # Bleu TT

# ─── 1. CHARGEMENT & CONSTRUCTION DU MASTER ───────────────────────────────────
def load_data(path):
    print(f"\n{'='*60}")
    print("  CHARGEMENT DES DONNÉES")
    print(f"{'='*60}")
    try:
        df_p    = pd.read_excel(os.path.join(path, 'ECH__DECEMBRE_2025 1.xlsx'))
        df_r    = pd.read_excel(os.path.join(path, 'RECHARGE_DECEMBRE_2025 1.xlsx'))
        df_s    = pd.read_excel(os.path.join(path, 'SORTANT_DECEMBRE_2025 1.xlsx'))
        df_e    = pd.read_excel(os.path.join(path, 'ENTRANT_DECEMBRE_2025 1.xlsx'))
        df_ussd = pd.read_excel(os.path.join(path, 'USSD_DECEMBRE_2025 1.xlsx'))
        df_reg  = pd.read_excel(os.path.join(path, 'REGION_ 1.xlsx'))
        df_o    = pd.read_excel(os.path.join(path, 'offre_ 1.xlsx'))
    except Exception as e:
        raise FileNotFoundError(f"❌ Fichier manquant dans '{path}' : {e}")

    for d in [df_p, df_r, df_s, df_e, df_ussd, df_reg, df_o]:
        d.columns = d.columns.str.strip().str.upper()

    key = 'ID'
    for d in [df_p, df_r, df_s, df_e, df_ussd]:
        d[key] = d[key].astype(str).str.strip().str.upper()

    # Agrégations (1 ligne par client)
    r_agg = df_r.groupby(key).agg(
        MNT_RECH     =('MNT_RECH',      'sum'),
        NB_RECH      =('NB_RECH',       'sum'),
        MNT_RECH_MOY =('MNT_RECH',      'mean'),
        NB_MOIS_ACTIF=('MONTH_DT',      'count'),
    ).reset_index()

    s_agg = df_s.groupby(key).agg(
        REVENU_CDR          =('REVENU_CDR',              'sum'),
        DUREE_APPEL_TOT     =('DUREE_APPEL_TOT',         'sum'),
        NB_SMS_TOT          =('NB_SMS_TOT',              'sum'),
        NB_APPEL_TOT        =('NB_APPEL_TOT',            'sum'),
        REVENU_INTER        =('REVENU_INTER',            'sum'),
        DUREE_APPEL_OOREDOO =('DUREE_APPEL_OOREDOO_TOT', 'sum'),
        DUREE_APPEL_ORANGE  =('DUREE_APPEL_ORANGE_TOT',  'sum'),
        DUREE_OFFNET        =('DUREE_OFFNET_TOT',        'sum'),
    ).reset_index()

    e_agg = df_e.groupby(key).agg(
        DUREE_APPEL_IN=('DUREE_APPEL_IN', 'sum'),
        NB_SMS_IN     =('NB_SMS_IN',      'sum'),
    ).reset_index()

    u_agg = df_ussd.groupby(key).agg(
        MNT_FORFAIT_DATA=('MNT_FORFAIT_DATA', 'sum'),
        NB_FORFAIT_DATA =('NB_FORFAIT_DATA',  'sum'),
    ).reset_index()

    df = (df_p
          .merge(df_reg, on='ID_REGION', how='left')
          .merge(df_o,   on='ID_OFFRE',  how='left')
          .merge(r_agg,  on=key, how='left')
          .merge(s_agg,  on=key, how='left')
          .merge(e_agg,  on=key, how='left')
          .merge(u_agg,  on=key, how='left')
    ).fillna(0)

    # Variables dérivées
    df['TARGET_DATA']  = (df['MNT_FORFAIT_DATA'] > 0).astype(int)
    df['LABEL_TARGET'] = df['TARGET_DATA'].map({1: 'Client DATA', 0: 'Non-DATA'})
    df['DUREE_ONNET']  = (df['DUREE_APPEL_TOT'] - df['DUREE_OFFNET']).clip(lower=0)
    df['HANDSET']      = df['HANDSET'].replace(0, 'Inconnu').astype(str)
    df['CANAL_DE_VENTE'] = df['CANAL_DE_VENTE'].replace(0, 'Inconnu').astype(str)
    df['TRANCHE_ANC']  = pd.cut(
        df['ANC_M'], bins=[0, 12, 36, 72, 500],
        labels=['0–1 an', '1–3 ans', '3–6 ans', '6 ans+']
    ).astype(str)

    print(f"  ✅ Master construit : {df.shape[0]:,} clients × {df.shape[1]} colonnes")
    return df

# ─── 2. RAPPORT TEXTE ─────────────────────────────────────────────────────────
def print_report(df):
    print(f"\n{'='*60}")
    print("  RAPPORT D'AUDIT DES DONNÉES")
    print(f"{'='*60}")

    print(f"\n  📊 Taille du parc          : {len(df):,} clients")
    print(f"  🎯 Clients DATA (TARGET=1) : {df['TARGET_DATA'].sum():,} "
          f"({df['TARGET_DATA'].mean()*100:.1f}%)")
    print(f"  ❌ Clients Non-DATA        : {(df['TARGET_DATA']==0).sum():,} "
          f"({(df['TARGET_DATA']==0).mean()*100:.1f}%)")
    print(f"  💰 Recharge totale         : {df['MNT_RECH'].sum():,.0f} DT")
    print(f"  💰 Recharge moyenne/client : {df['MNT_RECH'].mean():.2f} DT")
    print(f"  🕒 Ancienneté moyenne      : {df['ANC_M'].mean():.0f} mois")

    print(f"\n  ── Statut & Réseau ─────────────────────────────────")
    for s, v in df['STATUT'].value_counts().items():
        label = 'Actif' if s == 'A' else 'Suspendu'
        print(f"     {label:12s}: {v:,} ({v/len(df)*100:.1f}%)")
    for h, v in df['HANDSET'].value_counts().items():
        print(f"     {str(h):12s}: {v:,} ({v/len(df)*100:.1f}%)")

    print(f"\n  ── Outliers (méthode IQR) ──────────────────────────")
    for col in ['MNT_RECH', 'DUREE_APPEL_TOT', 'NB_SMS_TOT', 'NB_APPEL_TOT']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        print(f"     {col:20s}: {n:,} outliers ({n/len(df)*100:.1f}%)")

    print(f"\n  ── Corrélations avec TARGET_DATA ───────────────────")
    num_cols = ['ANC_M','MNT_RECH','MNT_RECH_MOY','REVENU_CDR',
                'DUREE_APPEL_TOT','NB_SMS_TOT','NB_APPEL_TOT',
                'DUREE_APPEL_IN','NB_SMS_IN','TARGET_DATA']
    corrs = df[num_cols].corr()['TARGET_DATA'].drop('TARGET_DATA').sort_values(ascending=False)
    for feat, corr in corrs.items():
        bar = '█' * int(abs(corr) * 30)
        sign = '+' if corr >= 0 else '-'
        print(f"     {feat:22s}: {sign}{abs(corr):.3f}  {bar}")

    print(f"\n  ── Top 5 Régions (Revenu) ──────────────────────────")
    top_reg = df.groupby('REGION')['MNT_RECH'].sum().sort_values(ascending=False).head(5)
    for r, v in top_reg.items():
        print(f"     {r:20s}: {v:,.0f} DT")

# ─── 3. VISUALISATIONS ────────────────────────────────────────────────────────
def generate_plots(df):
    print(f"\n{'─'*60}")
    print("  GÉNÉRATION DES GRAPHIQUES (16 visualisations)")
    print(f"{'─'*60}")

    fig = plt.figure(figsize=(24, 32))
    fig.patch.set_facecolor('#f8fafc')
    fig.suptitle(
        "Analyse Exploratoire des Données — Tunisie Telecom PFE\n"
        "Décembre 2025 | 31 200 clients",
        fontsize=20, fontweight='bold', color=COLOR_MAIN, y=0.995
    )

    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.52, wspace=0.38,
                           top=0.97, bottom=0.03, left=0.06, right=0.97)

    # ══════════════════════════════════════════════════════════════
    # SECTION 1 — QUALITÉ DES DONNÉES
    # ══════════════════════════════════════════════════════════════
    fig.text(0.02, 0.965, "① QUALITÉ DES DONNÉES", fontsize=13,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_MAIN, alpha=0.9))

    # 1A. Valeurs manquantes
    ax = fig.add_subplot(gs[0, 0])
    nan_data = df.isnull().sum()
    nan_data = nan_data[nan_data > 0].sort_values(ascending=True)
    if nan_data.empty:
        ax.text(0.5, 0.5, '✅ Aucune valeur\nmanquante', ha='center', va='center',
                fontsize=14, color='green', transform=ax.transAxes)
    else:
        nan_data.plot(kind='barh', ax=ax, color='#ef4444')
    ax.set_title('1. Valeurs Manquantes', fontweight='bold', fontsize=11)
    ax.set_xlabel('Nb NaN')

    # 1B. Boxplot outliers Recharge
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot(df['MNT_RECH'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor='#bae6fd', color=COLOR_MAIN),
                     medianprops=dict(color='#ef4444', linewidth=2),
                     flierprops=dict(marker='o', color='#ef4444', alpha=0.3, markersize=3))
    Q1, Q3 = df['MNT_RECH'].quantile(0.25), df['MNT_RECH'].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df['MNT_RECH'] < Q1-1.5*IQR) | (df['MNT_RECH'] > Q3+1.5*IQR)).sum()
    ax2.set_title(f'2. Outliers Recharge\n({n_out:,} clients atypiques = {n_out/len(df)*100:.1f}%)',
                  fontweight='bold', fontsize=11)
    ax2.set_ylabel('Montant Recharge (DT)')
    ax2.set_xticks([])

    # 1C. Matrice de corrélation
    ax3 = fig.add_subplot(gs[0, 2:4])
    num_cols = ['ANC_M','MNT_RECH','MNT_RECH_MOY','REVENU_CDR',
                'DUREE_APPEL_TOT','NB_SMS_TOT','NB_APPEL_TOT',
                'DUREE_APPEL_IN','NB_SMS_IN','TARGET_DATA']
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdYlGn', fmt='.2f',
                ax=ax3, linewidths=0.5, annot_kws={'size': 8},
                vmin=-0.5, vmax=0.5, center=0)
    ax3.set_title('3. Matrice de Corrélation (Pearson)', fontweight='bold', fontsize=11)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.tick_params(axis='y', rotation=0, labelsize=8)

    # ══════════════════════════════════════════════════════════════
    # SECTION 2 — PROFIL DU PARC CLIENT
    # ══════════════════════════════════════════════════════════════
    fig.text(0.02, 0.77, "② PROFIL DU PARC CLIENT", fontsize=13,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#0ea5e9', alpha=0.9))

    # 2A. Répartition Handset
    ax4 = fig.add_subplot(gs[1, 0])
    hs = df['HANDSET'].value_counts()
    colors_hs = {'2G':'#ef4444','3G':'#f97316','4G':'#3b82f6','5G':'#10b981','Inconnu':'#9ca3af'}
    c_list = [colors_hs.get(h, '#6b7280') for h in hs.index]
    wedges, texts, autotexts = ax4.pie(hs.values, labels=hs.index, autopct='%1.1f%%',
                                        colors=c_list, startangle=90,
                                        textprops={'fontsize': 9})
    ax4.set_title('4. Répartition Type Réseau', fontweight='bold', fontsize=11)

    # 2B. Distribution Ancienneté
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(df['ANC_M'], bins=35, color=COLOR_MAIN, edgecolor='white', alpha=0.85)
    ax5.axvline(df['ANC_M'].mean(), color='#ef4444', linestyle='--', lw=2,
                label=f"Moy = {df['ANC_M'].mean():.0f} mois")
    ax5.axvline(df['ANC_M'].median(), color='#f97316', linestyle=':', lw=2,
                label=f"Méd = {df['ANC_M'].median():.0f} mois")
    ax5.set_title('5. Distribution Ancienneté (mois)', fontweight='bold', fontsize=11)
    ax5.set_xlabel('Ancienneté (mois)')
    ax5.set_ylabel('Nb clients')
    ax5.legend(fontsize=8)

    # 2C. Statut client
    ax6 = fig.add_subplot(gs[1, 2])
    stat = df['STATUT'].value_counts()
    labels_s = ['Actif (A)' if s=='A' else 'Suspendu (S)' for s in stat.index]
    ax6.bar(labels_s, stat.values, color=[COLOR_POS, COLOR_NEG], edgecolor='white', width=0.5)
    for i, (l, v) in enumerate(zip(labels_s, stat.values)):
        ax6.text(i, v + 100, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center',
                 fontsize=10, fontweight='bold')
    ax6.set_title('6. Statut du Parc Client', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Nb clients')
    ax6.set_ylim(0, stat.max() * 1.2)

    # 2D. Canal de vente
    ax7 = fig.add_subplot(gs[1, 3])
    canal = df[df['CANAL_DE_VENTE'] != 'Inconnu']['CANAL_DE_VENTE'].value_counts()
    ax7.barh(canal.index, canal.values, color=PALETTE_TT[:len(canal)], edgecolor='white')
    for i, v in enumerate(canal.values):
        ax7.text(v + 50, i, f'{v:,}', va='center', fontsize=10, fontweight='bold')
    ax7.set_title('7. Canal de Vente', fontweight='bold', fontsize=11)
    ax7.set_xlabel('Nb clients')

    # ══════════════════════════════════════════════════════════════
    # SECTION 3 — REVENUS & USAGE
    # ══════════════════════════════════════════════════════════════
    fig.text(0.02, 0.575, "③ REVENUS & USAGE", fontsize=13,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#10b981', alpha=0.9))

    # 3A. Distribution recharge par TARGET (KDE)
    ax8 = fig.add_subplot(gs[2, 0:2])
    for target, color, label in [(0, COLOR_NEG, 'Non-DATA'), (1, COLOR_POS, 'Client DATA')]:
        subset = df[df['TARGET_DATA']==target]['MNT_RECH']
        subset = subset[subset <= subset.quantile(0.97)]  # sans outliers extrêmes
        ax8.hist(subset, bins=40, density=True, alpha=0.55, color=color, label=label)
        subset.plot.kde(ax=ax8, color=color, lw=2)
    ax8.set_title('8. Distribution Recharge — DATA vs Non-DATA', fontweight='bold', fontsize=11)
    ax8.set_xlabel('Montant Recharge (DT)')
    ax8.set_ylabel('Densité')
    ax8.legend(fontsize=10)

    # 3B. Revenu moyen par catégorie d'offre
    ax9 = fig.add_subplot(gs[2, 2])
    rev_offre = df.groupby('OFFRE_CAT')['MNT_RECH'].mean().sort_values(ascending=True)
    ax9.barh(rev_offre.index, rev_offre.values,
             color=[COLOR_MAIN if v >= rev_offre.mean() else '#93c5fd' for v in rev_offre.values])
    ax9.axvline(rev_offre.mean(), color='#ef4444', linestyle='--', lw=1.5, label='Moyenne')
    ax9.set_title("9. Recharge Moy. par Offre", fontweight='bold', fontsize=11)
    ax9.set_xlabel('Montant (DT)')
    ax9.legend(fontsize=9)

    # 3C. Mix trafic opérateurs (donut)
    ax10 = fig.add_subplot(gs[2, 3])
    op_vals = [
        df['DUREE_ONNET'].sum(),
        df['DUREE_APPEL_OOREDOO'].sum(),
        df['DUREE_APPEL_ORANGE'].sum(),
        (df['DUREE_OFFNET'] - df['DUREE_APPEL_OOREDOO'] - df['DUREE_APPEL_ORANGE']).clip(lower=0).sum()
    ]
    op_labels = ['TT (Onnet)', 'Ooredoo', 'Orange', 'Autres']
    op_colors = [COLOR_MAIN, '#e00717', '#ff6b00', '#6b7280']
    ax10.pie(op_vals, labels=op_labels, autopct='%1.1f%%', colors=op_colors,
             startangle=90, pctdistance=0.82,
             wedgeprops=dict(width=0.55), textprops={'fontsize': 9})
    ax10.set_title('10. Mix Trafic par Opérateur', fontweight='bold', fontsize=11)

    # ══════════════════════════════════════════════════════════════
    # SECTION 4 — ANALYSE GÉOGRAPHIQUE
    # ══════════════════════════════════════════════════════════════
    fig.text(0.02, 0.385, "④ ANALYSE GÉOGRAPHIQUE", fontsize=13,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#8b5cf6', alpha=0.9))

    # 4A. Top 10 régions par revenu
    ax11 = fig.add_subplot(gs[3, 0:2])
    reg_rev = df.groupby('REGION')['MNT_RECH'].sum().sort_values(ascending=True).tail(10)
    colors_reg = [COLOR_MAIN if r != reg_rev.index[-1] else '#10b981' for r in reg_rev.index]
    ax11.barh(reg_rev.index, reg_rev.values, color=colors_reg, edgecolor='white')
    for i, (r, v) in enumerate(reg_rev.items()):
        ax11.text(v + 2000, i, f'{v/1000:.0f}k DT', va='center', fontsize=9)
    ax11.set_title('11. Top 10 Régions — Revenu Total (DT)', fontweight='bold', fontsize=11)
    ax11.set_xlabel('Revenu Total (DT)')

    # 4B. Taux adoption DATA par région (top 10)
    ax12 = fig.add_subplot(gs[3, 2:4])
    reg_data = df.groupby('REGION')['TARGET_DATA'].mean().sort_values(ascending=True).tail(10) * 100
    bar_colors = [COLOR_POS if v >= 80 else '#fbbf24' if v >= 60 else COLOR_NEG
                  for v in reg_data.values]
    ax12.barh(reg_data.index, reg_data.values, color=bar_colors, edgecolor='white')
    ax12.axvline(df['TARGET_DATA'].mean()*100, color='#004a99', linestyle='--',
                 lw=2, label=f"Moy nationale = {df['TARGET_DATA'].mean()*100:.1f}%")
    for i, (r, v) in enumerate(reg_data.items()):
        ax12.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax12.set_title('12. Taux Adoption DATA par Région (%)', fontweight='bold', fontsize=11)
    ax12.set_xlabel('% Clients DATA')
    ax12.legend(fontsize=9)

    # ══════════════════════════════════════════════════════════════
    # SECTION 5 — ANALYSE BIVARIÉE TARGET_DATA
    # ══════════════════════════════════════════════════════════════
    fig.text(0.02, 0.195, "⑤ ANALYSE BIVARIÉE — TARGET DATA", fontsize=13,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f97316', alpha=0.9))

    # 5A. Taux DATA par handset
    ax13 = fig.add_subplot(gs[4, 0])
    hs_data = df.groupby('HANDSET')['TARGET_DATA'].mean().sort_values(ascending=False) * 100
    hs_data = hs_data[hs_data.index != 'Inconnu']
    bar_c = [COLOR_POS if v >= 85 else '#fbbf24' if v >= 60 else COLOR_NEG for v in hs_data.values]
    ax13.bar(hs_data.index, hs_data.values, color=bar_c, edgecolor='white', width=0.6)
    for i, v in enumerate(hs_data.values):
        ax13.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax13.axhline(df['TARGET_DATA'].mean()*100, color='#004a99', linestyle='--', lw=1.5,
                 label=f"Moy = {df['TARGET_DATA'].mean()*100:.1f}%")
    ax13.set_title('13. Taux DATA par Type Réseau', fontweight='bold', fontsize=11)
    ax13.set_ylabel('% Clients DATA')
    ax13.set_ylim(0, 105)
    ax13.legend(fontsize=9)

    # 5B. Taux DATA par tranche ancienneté
    ax14 = fig.add_subplot(gs[4, 1])
    anc_order = ['0–1 an', '1–3 ans', '3–6 ans', '6 ans+']
    anc_data = df.groupby('TRANCHE_ANC')['TARGET_DATA'].mean().reindex(anc_order) * 100
    ax14.bar(anc_data.index, anc_data.values,
             color=[COLOR_POS if v >= 80 else '#fbbf24' if v >= 60 else COLOR_NEG
                    for v in anc_data.values],
             edgecolor='white', width=0.6)
    for i, v in enumerate(anc_data.values):
        ax14.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax14.axhline(df['TARGET_DATA'].mean()*100, color='#004a99', linestyle='--', lw=1.5)
    ax14.set_title("14. Taux DATA par Ancienneté", fontweight='bold', fontsize=11)
    ax14.set_ylabel('% Clients DATA')
    ax14.set_ylim(0, 105)

    # 5C. Boxplot recharge DATA vs Non-DATA par ancienneté
    ax15 = fig.add_subplot(gs[4, 2])
    df_box = df[df['MNT_RECH'] <= df['MNT_RECH'].quantile(0.95)].copy()
    df_box['LABEL_TARGET'] = df_box['TARGET_DATA'].map({1:'DATA', 0:'Non-DATA'})
    sns.boxplot(data=df_box, x='TRANCHE_ANC', y='MNT_RECH',
                hue='LABEL_TARGET', order=anc_order,
                palette={'DATA': COLOR_POS, 'Non-DATA': COLOR_NEG},
                ax=ax15, linewidth=1.2, fliersize=2)
    ax15.set_title('15. Recharge par Ancienneté & Target', fontweight='bold', fontsize=11)
    ax15.set_xlabel('Tranche Ancienneté')
    ax15.set_ylabel('Recharge (DT)')
    ax15.tick_params(axis='x', rotation=20)
    ax15.legend(title='', fontsize=8)

    # 5D. Taux DATA par catégorie d'offre
    ax16 = fig.add_subplot(gs[4, 3])
    offre_data = df.groupby('OFFRE_CAT')['TARGET_DATA'].mean().sort_values(ascending=True) * 100
    ax16.barh(offre_data.index, offre_data.values,
              color=[COLOR_POS if v >= 85 else '#fbbf24' if v >= 70 else COLOR_NEG
                     for v in offre_data.values],
              edgecolor='white')
    ax16.axvline(df['TARGET_DATA'].mean()*100, color='#004a99', linestyle='--',
                 lw=1.5, label=f"Moy = {df['TARGET_DATA'].mean()*100:.1f}%")
    for i, v in enumerate(offre_data.values):
        ax16.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)
    ax16.set_title("16. Taux DATA par Catégorie d'Offre", fontweight='bold', fontsize=11)
    ax16.set_xlabel('% Clients DATA')
    ax16.legend(fontsize=9)

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor='#f8fafc')
    plt.close()
    print(f"\n  ✅ Rapport graphique sauvegardé : {OUTPUT_PNG}")

# ─── 4. EXPORT CSV ────────────────────────────────────────────────────────────
def export_csv(df):
    cols_export = [c for c in [
        'ID', 'REGION', 'OFFRE', 'OFFRE_CAT', 'STATUT', 'STATUT_RGS90',
        'HANDSET', 'CANAL_DE_VENTE', 'ANC_M', 'TRANCHE_ANC',
        'MNT_RECH', 'NB_RECH', 'MNT_RECH_MOY', 'NB_MOIS_ACTIF',
        'REVENU_CDR', 'DUREE_APPEL_TOT', 'NB_SMS_TOT', 'NB_APPEL_TOT',
        'REVENU_INTER', 'DUREE_ONNET', 'DUREE_OFFNET',
        'DUREE_APPEL_IN', 'NB_SMS_IN',
        'MNT_FORFAIT_DATA', 'NB_FORFAIT_DATA',
        'TARGET_DATA'
    ] if c in df.columns]
    df[cols_export].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  ✅ Fichier nettoyé exporté   : {OUTPUT_CSV}")
    print(f"     {df.shape[0]:,} lignes × {len(cols_export)} colonnes")

# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────
def eda_master_telecom():
    # 1. Charger et construire le master
    df = load_data(DATA_PATH)

    # 2. Rapport texte
    print_report(df)

    # 3. Générer les 16 visualisations
    generate_plots(df)

    # 4. Exporter le CSV nettoyé
    export_csv(df)

    print(f"\n{'='*60}")
    print("  ✅ EDA TERMINÉE AVEC SUCCÈS")
    print(f"{'='*60}")
    print(f"  → Fichiers générés :")
    print(f"     📊 {OUTPUT_PNG}")
    print(f"     💾 {OUTPUT_CSV}")
    print(f"  → Prochaine étape : preprocessing_pfe_fixed.py\n")

if __name__ == "__main__":
    eda_master_telecom()