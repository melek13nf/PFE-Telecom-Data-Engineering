"""
=============================================================
  rfm_analysis.py — Tunisie Telecom PFE
  Analyse RFM (Recency, Frequency, Monetary)
  Segmentation comportementale basée sur les recharges
  Entrée  : dossier data/ (fichiers Excel)
  Sortie  : rfm_segments.csv + rapport_rfm.png
=============================================================
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
DATA_PATH   = "data"
OUTPUT_CSV  = "outputs/rfm_segments.csv"
OUTPUT_PNG  = "outputs/rapport_rfm.png"

COLOR_MAIN  = "#004a99"
PALETTE_RFM = {
    "Champions"       : "#10b981",
    "Clients Fidèles" : "#3b82f6",
    "Clients Potentiels": "#0ea5e9",
    "Nouveaux Clients": "#8b5cf6",
    "Clients Ordinaires": "#f59e0b",
    "Clients à Risque": "#f97316",
    "Clients Perdus"  : "#ef4444",
}

# ─── 1. CHARGEMENT ────────────────────────────────────────────────────────────
def load_data(path):
    print(f"\n{'='*55}")
    print("  ANALYSE RFM — Tunisie Telecom")
    print(f"{'='*55}")

    df_ech = pd.read_excel(os.path.join(path, 'ECH__DECEMBRE_2025 1.xlsx'))
    df_r   = pd.read_excel(os.path.join(path, 'RECHARGE_DECEMBRE_2025 1.xlsx'))
    df_reg = pd.read_excel(os.path.join(path, 'REGION_ 1.xlsx'))
    df_o   = pd.read_excel(os.path.join(path, 'offre_ 1.xlsx'))
    df_ussd= pd.read_excel(os.path.join(path, 'USSD_DECEMBRE_2025 1.xlsx'))

    for d in [df_ech, df_r, df_reg, df_o, df_ussd]:
        d.columns = d.columns.str.strip().str.upper()
        if 'ID' in d.columns:
            d['ID'] = d['ID'].astype(str).str.strip().str.upper()

    df_r['MONTH_DT'] = pd.to_datetime(
        df_r['MONTH_DT'].str[:9], format='%d%b%Y', errors='coerce')
    df_r = df_r.dropna(subset=['MONTH_DT'])

    print(f"  ✅ ECH chargé     : {len(df_ech):,} clients")
    print(f"  ✅ RECHARGE chargé : {len(df_r):,} transactions")
    return df_ech, df_r, df_reg, df_o, df_ussd

# ─── 2. CALCUL RFM ────────────────────────────────────────────────────────────
def compute_rfm(df_ech, df_r, df_ussd):
    print(f"\n{'─'*55}")
    print("  CALCUL DES SCORES RFM")
    print(f"{'─'*55}")

    max_date = df_r['MONTH_DT'].max()
    print(f"  Date de référence : {max_date.strftime('%B %Y')}")

    # ── Recency : nb de mois depuis la dernière recharge ──────────────────
    recency = df_r.groupby('ID')['MONTH_DT'].max().reset_index()
    recency['RECENCY'] = ((max_date - recency['MONTH_DT']).dt.days / 30
                          ).clip(0, 12).round(0).astype(int)

    # ── Frequency : nb de mois distincts avec au moins une recharge ───────
    frequency = df_r.groupby('ID')['MONTH_DT'].nunique().reset_index()
    frequency.columns = ['ID', 'FREQUENCY']

    # ── Monetary : montant total rechargé sur la période ──────────────────
    monetary = df_r.groupby('ID').agg(
        MONETARY   =('MNT_RECH',     'sum'),
        NB_RECH_TOT=('NB_RECH',      'sum'),
        MNT_MOY    =('MNT_RECH',     'mean'),
    ).reset_index()
    monetary['MONETARY'] = monetary['MONETARY'].clip(lower=0)

    # ── Assemblage RFM ────────────────────────────────────────────────────
    rfm = (df_ech[['ID','ANC_M','HANDSET','STATUT','OFFRE_CAT','ID_REGION','ID_OFFRE']]
           .merge(recency[['ID','RECENCY']],   on='ID', how='left')
           .merge(frequency,                   on='ID', how='left')
           .merge(monetary,                    on='ID', how='left')
    ).fillna({'RECENCY':12, 'FREQUENCY':0, 'MONETARY':0,
              'NB_RECH_TOT':0, 'MNT_MOY':0})

    rfm['HANDSET'] = rfm['HANDSET'].replace(0, 'Inconnu').astype(str)

    # ── Target DATA (depuis USSD) ──────────────────────────────────────────
    ussd_agg = df_ussd.groupby('ID')['MNT_FORFAIT_DATA'].sum().reset_index()
    rfm = rfm.merge(ussd_agg, on='ID', how='left').fillna({'MNT_FORFAIT_DATA':0})
    rfm['TARGET_DATA'] = (rfm['MNT_FORFAIT_DATA'] > 0).astype(int)

    # ── Scores R, F, M (1 à 5) ────────────────────────────────────────────
    # R : plus la Recency est faible (récent), meilleur le score
    rfm['R_SCORE'] = pd.qcut(
        rfm['RECENCY'].rank(method='first'), 5,
        labels=[5, 4, 3, 2, 1]).astype(int)

    # F : plus la Frequency est haute, meilleur le score
    rfm['F_SCORE'] = pd.qcut(
        rfm['FREQUENCY'].rank(method='first'), 5,
        labels=[1, 2, 3, 4, 5]).astype(int)

    # M : plus le Monetary est élevé, meilleur le score
    rfm['M_SCORE'] = pd.qcut(
        rfm['MONETARY'].rank(method='first'), 5,
        labels=[1, 2, 3, 4, 5]).astype(int)

    rfm['RFM_SCORE']     = rfm['R_SCORE'] + rfm['F_SCORE'] + rfm['M_SCORE']
    rfm['RFM_SCORE_STR'] = (rfm['R_SCORE'].astype(str)
                           + rfm['F_SCORE'].astype(str)
                           + rfm['M_SCORE'].astype(str))

    # ── Segmentation RFM ──────────────────────────────────────────────────
    def segment(row):
        r, f, m = row['R_SCORE'], row['F_SCORE'], row['M_SCORE']
        if   r >= 4 and f >= 4 and m >= 4: return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3: return 'Clients Fidèles'
        elif r >= 4 and f <= 2:             return 'Nouveaux Clients'
        elif r <= 2 and f >= 3 and m >= 3: return 'Clients à Risque'
        elif r <= 2 and f <= 2:             return 'Clients Perdus'
        elif r >= 3 and m >= 4:             return 'Clients Potentiels'
        else:                               return 'Clients Ordinaires'

    rfm['SEGMENT_RFM'] = rfm.apply(segment, axis=1)

    # ── Rapport console ───────────────────────────────────────────────────
    print(f"\n  {'Segment':<22} {'Nb Clients':>10}  {'% Parc':>7}  "
          f"{'Recency Moy':>12}  {'Freq Moy':>9}  {'Monetary Moy':>13}")
    print(f"  {'─'*80}")
    for seg in PALETTE_RFM.keys():
        sub = rfm[rfm['SEGMENT_RFM']==seg]
        if len(sub) == 0: continue
        print(f"  {seg:<22} {len(sub):>10,}  {len(sub)/len(rfm)*100:>6.1f}%  "
              f"  {sub['RECENCY'].mean():>10.1f}  "
              f"{sub['FREQUENCY'].mean():>9.1f}  "
              f"{sub['MONETARY'].mean():>12.1f} DT")
    print(f"  {'─'*80}")
    print(f"  {'TOTAL':<22} {len(rfm):>10,}  {'100.0%':>7}")

    return rfm

# ─── 3. VISUALISATIONS RFM ────────────────────────────────────────────────────
def plot_rfm(rfm):
    print(f"\n{'─'*55}")
    print("  GÉNÉRATION DES GRAPHIQUES RFM")
    print(f"{'─'*55}")

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#f8fafc')
    fig.suptitle(
        "Analyse RFM — Segmentation Comportementale | Tunisie Telecom\n"
        "Recency · Frequency · Monetary",
        fontsize=18, fontweight='bold', color=COLOR_MAIN, y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.38,
                           top=0.93, bottom=0.05, left=0.07, right=0.97)

    seg_counts = rfm['SEGMENT_RFM'].value_counts()
    colors_list = [PALETTE_RFM.get(s, '#888') for s in seg_counts.index]

    # ── G1 : Répartition segments (donut) ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    wedges, texts, autotexts = ax1.pie(
        seg_counts.values, labels=seg_counts.index,
        autopct='%1.1f%%', colors=colors_list,
        startangle=90, pctdistance=0.82,
        wedgeprops=dict(width=0.55),
        textprops={'fontsize': 8})
    for at in autotexts: at.set_fontsize(8)
    ax1.set_title('1. Répartition des Segments RFM', fontweight='bold', fontsize=11)

    # ── G2 : Nb clients par segment (barh) ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    seg_df = seg_counts.sort_values(ascending=True)
    bars = ax2.barh(seg_df.index, seg_df.values,
                    color=[PALETTE_RFM.get(s,'#888') for s in seg_df.index],
                    edgecolor='white', height=0.7)
    for bar, v in zip(bars, seg_df.values):
        ax2.text(bar.get_width()+50, bar.get_y()+bar.get_height()/2,
                 f'{v:,}', va='center', fontsize=9, fontweight='bold')
    ax2.set_title('2. Volume par Segment', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Nb Clients')
    ax2.set_xlim(0, seg_df.max()*1.2)

    # ── G3 : Score RFM moyen par segment ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    seg_score = rfm.groupby('SEGMENT_RFM')['RFM_SCORE'].mean().sort_values()
    bars3 = ax3.barh(seg_score.index, seg_score.values,
                     color=[PALETTE_RFM.get(s,'#888') for s in seg_score.index],
                     edgecolor='white', height=0.7)
    for bar, v in zip(bars3, seg_score.values):
        ax3.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                 f'{v:.1f}', va='center', fontsize=9, fontweight='bold')
    ax3.set_title('3. Score RFM Moyen par Segment', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Score RFM (3–15)')
    ax3.axvline(rfm['RFM_SCORE'].mean(), color='black',
                linestyle='--', lw=1.5, label=f'Moy={rfm["RFM_SCORE"].mean():.1f}')
    ax3.legend(fontsize=9)

    # ── G4 : Distribution Recency ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    rec_counts = rfm['RECENCY'].value_counts().sort_index()
    ax4.bar(rec_counts.index, rec_counts.values,
            color=[COLOR_MAIN if i==0 else '#ef4444' if i>=3 else '#f97316'
                   for i in rec_counts.index],
            edgecolor='white')
    ax4.set_title('4. Distribution Recency (mois depuis\nla dernière recharge)',
                  fontweight='bold', fontsize=11)
    ax4.set_xlabel('Mois d\'inactivité')
    ax4.set_ylabel('Nb Clients')
    ax4.annotate(f"Actifs récents:\n{(rfm['RECENCY']==0).sum():,} clients",
                 xy=(0, (rfm['RECENCY']==0).sum()),
                 xytext=(3, (rfm['RECENCY']==0).sum()*0.8),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9, color=COLOR_MAIN, fontweight='bold')

    # ── G5 : Distribution Frequency ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    freq_counts = rfm['FREQUENCY'].value_counts().sort_index()
    ax5.bar(freq_counts.index, freq_counts.values,
            color=[COLOR_MAIN if f>=10 else '#f97316' if f>=6 else '#ef4444'
                   for f in freq_counts.index],
            edgecolor='white')
    ax5.set_title('5. Distribution Frequency\n(mois actifs sur 12)',
                  fontweight='bold', fontsize=11)
    ax5.set_xlabel('Nb de mois avec recharge')
    ax5.set_ylabel('Nb Clients')

    # ── G6 : Distribution Monetary (KDE) ─────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    monetary_clean = rfm[rfm['MONETARY'] <= rfm['MONETARY'].quantile(0.95)]['MONETARY']
    ax6.hist(monetary_clean, bins=40, color=COLOR_MAIN,
             edgecolor='white', alpha=0.8)
    ax6.axvline(rfm['MONETARY'].mean(), color='#ef4444', lw=2, linestyle='--',
                label=f'Moy={rfm["MONETARY"].mean():.0f} DT')
    ax6.axvline(rfm['MONETARY'].median(), color='#f97316', lw=2, linestyle=':',
                label=f'Méd={rfm["MONETARY"].median():.0f} DT')
    ax6.set_title('6. Distribution Monetary\n(Recharge totale – 95ème percentile)',
                  fontweight='bold', fontsize=11)
    ax6.set_xlabel('Montant Recharge (DT)')
    ax6.set_ylabel('Nb Clients')
    ax6.legend(fontsize=9)

    # ── G7 : Revenu moyen par segment ────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    seg_rev = rfm.groupby('SEGMENT_RFM')['MONETARY'].mean().sort_values(ascending=True)
    ax7.barh(seg_rev.index, seg_rev.values,
             color=[PALETTE_RFM.get(s,'#888') for s in seg_rev.index],
             edgecolor='white', height=0.7)
    for i, v in enumerate(seg_rev.values):
        ax7.text(v+2, i, f'{v:.0f} DT', va='center', fontsize=9, fontweight='bold')
    ax7.set_title('7. Revenu Moyen par Segment (DT)',
                  fontweight='bold', fontsize=11)
    ax7.set_xlabel('Recharge Moyenne (DT)')

    # ── G8 : Taux DATA par segment RFM ───────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    seg_data = rfm.groupby('SEGMENT_RFM')['TARGET_DATA'].mean().sort_values(ascending=True)*100
    bar_colors = [PALETTE_RFM.get(s,'#888') for s in seg_data.index]
    ax8.barh(seg_data.index, seg_data.values, color=bar_colors,
             edgecolor='white', height=0.7)
    ax8.axvline(rfm['TARGET_DATA'].mean()*100, color='black',
                linestyle='--', lw=1.5,
                label=f'Moy={rfm["TARGET_DATA"].mean()*100:.1f}%')
    for i, v in enumerate(seg_data.values):
        ax8.text(v+0.5, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax8.set_title("8. Taux d'Appétence DATA par Segment RFM",
                  fontweight='bold', fontsize=11)
    ax8.set_xlabel('% Clients DATA')
    ax8.legend(fontsize=9)

    # ── G9 : Scatter R vs M coloré par segment ────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    for seg, color in PALETTE_RFM.items():
        sub = rfm[rfm['SEGMENT_RFM']==seg]
        if len(sub)==0: continue
        ax9.scatter(sub['R_SCORE'], sub['M_SCORE'],
                    c=color, label=seg, alpha=0.6, s=20, edgecolors='none')
    ax9.set_title('9. Carte RFM — Score R vs Score M',
                  fontweight='bold', fontsize=11)
    ax9.set_xlabel('Score Recency (1=inactif → 5=très récent)')
    ax9.set_ylabel('Score Monetary (1=faible → 5=élevé)')
    ax9.legend(fontsize=7, ncol=1, loc='lower right')
    ax9.set_xticks([1,2,3,4,5])
    ax9.set_yticks([1,2,3,4,5])
    ax9.grid(True, alpha=0.3)

    os.makedirs('outputs', exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor='#f8fafc')
    plt.close()
    print(f"  ✅ Rapport RFM sauvegardé : {OUTPUT_PNG}")

# ─── 4. EXPORT ────────────────────────────────────────────────────────────────
def export_rfm(rfm):
    os.makedirs('outputs', exist_ok=True)
    cols = ['ID','ANC_M','HANDSET','STATUT','OFFRE_CAT',
            'RECENCY','FREQUENCY','MONETARY','NB_RECH_TOT','MNT_MOY',
            'R_SCORE','F_SCORE','M_SCORE','RFM_SCORE','RFM_SCORE_STR',
            'SEGMENT_RFM','TARGET_DATA']
    rfm[[c for c in cols if c in rfm.columns]].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  ✅ Fichier RFM exporté     : {OUTPUT_CSV}")
    print(f"     {len(rfm):,} clients × {len(cols)} colonnes")

    # Recommandations marketing par segment
    print(f"\n{'─'*55}")
    print("  RECOMMANDATIONS MARKETING PAR SEGMENT")
    print(f"{'─'*55}")
    recommandations = {
        "Champions"        : "🏆 Récompenser la fidélité — offres exclusives VIP, programme de parrainage",
        "Clients Fidèles"  : "🎁 Upselling — proposer forfaits DATA premium ou offres groupées",
        "Clients Potentiels": "⚡ Nurturing — campagnes ciblées pour convertir en Champions",
        "Nouveaux Clients" : "👋 Onboarding — offre de bienvenue DATA, accompagnement personnalisé",
        "Clients Ordinaires": "📊 Engagement — promotions flash, bonus recharge pour augmenter fréquence",
        "Clients à Risque" : "⚠️  Rétention — offre de fidélisation urgente, appel proactif",
        "Clients Perdus"   : "🔴 Réactivation — campagne win-back avec remise exceptionnelle",
    }
    for seg, reco in recommandations.items():
        n = (rfm['SEGMENT_RFM']==seg).sum()
        print(f"  {seg:<22} ({n:>5,} clients) → {reco}")

# ─── PIPELINE ─────────────────────────────────────────────────────────────────
def run_rfm():
    df_ech, df_r, df_reg, df_o, df_ussd = load_data(DATA_PATH)
    rfm = compute_rfm(df_ech, df_r, df_ussd)
    plot_rfm(rfm)
    export_rfm(rfm)
    print(f"\n{'='*55}")
    print("  ✅ ANALYSE RFM TERMINÉE")
    print(f"{'='*55}")
    print(f"  → {OUTPUT_CSV}")
    print(f"  → {OUTPUT_PNG}")
    print(f"  → Prochaine étape : churn_model.py\n")

if __name__ == "__main__":
    run_rfm()