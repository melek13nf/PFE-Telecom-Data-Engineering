"""
=============================================================
  train_models_cv_mlops.py — Tunisie Telecom PFE
  Entraînement + MLOps (MLflow)

  MLOps intégré :
    ① Tracking des expériences  → MLflow logs AUC, params, metrics
    ② Versioning des modèles    → chaque run a un ID unique
    ③ Monitoring / Drift        → comparaison avec le run précédent
    ④ Rapport MLOps             → résumé complet dans mlops_report.txt

  Fichier source : CIBLE_ECH_DEC_2025_6.xlsx
  Modèle 1 : Activation Forfait DATA  (TARGET)
  Modèle 2 : Prédiction Churn         (FLAG_CHURN)
  Méthode   : StratifiedKFold 5-Fold
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import glob
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, average_precision_score,
    accuracy_score, f1_score
)
import matplotlib.pyplot as plt

# ─── CONFIGURATION ────────────────────────────────────────
N_FOLDS         = 5
MODEL_FORFAIT   = "model_forfait_cv.joblib"
MODEL_CHURN     = "model_churn_cv.joblib"
REPORT_DIR      = "outputs/model_report/"
MLFLOW_DIR      = "mlruns"                   # Dossier local MLflow
EXPERIMENT_NAME = "TunisieTelecom_PFE"       # Nom de l'expérience MLflow
MLOPS_REPORT    = "outputs/mlops_report.txt"  # Rapport texte MLOps
METRICS_FILE    = "outputs/mlops_metrics_history.json"  # Historique des métriques

# ─── SEUIL D'ALERTE DRIFT ─────────────────────────────────
# Si l'AUC baisse de plus de DRIFT_THRESHOLD par rapport au run précédent
# → alerte Model Drift détectée
DRIFT_THRESHOLD = 0.03

# ─── 1. DÉTECTION DU FICHIER SOURCE ───────────────────────
def find_data_file():
    for pat in ["data/CIBLE*.xlsx","data/cible*.xlsx",
                "data/ECH*.xlsx","data/ech*.xlsx","data/*.xlsx"]:
        files = glob.glob(pat)
        if files:
            print(f"  Fichier détecté : {files[0]}")
            return files[0]
    raise FileNotFoundError(
        "Aucun fichier xlsx trouvé dans data/\n"
        "Placez CIBLE_ECH_DEC_2025_6.xlsx dans le dossier data/"
    )

# ─── 2. CHARGEMENT ────────────────────────────────────────
def load_data(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.upper()
    print(f"  {len(df):,} clients  ×  {df.shape[1]} colonnes")
    print(f"  TARGET=1    : {df['TARGET'].sum():,} "
          f"({df['TARGET'].mean()*100:.1f}%)")
    print(f"  CHURN=1     : {df['FLAG_CHURN'].sum():,} "
          f"({df['FLAG_CHURN'].mean()*100:.1f}%)")
    return df

# ─── 3. PRÉPARATION DES FEATURES ──────────────────────────
def prepare_features(df):
    df = df.copy()
    df['HANDSET']      = df['HANDSET'].fillna('Inconnu').astype(str)
    df['CLASSE_CANAL'] = df['CLASSE_CANAL'].fillna('Inconnu').astype(str)
    df['CODE_REGION']  = pd.to_numeric(df['CODE_REGION'], errors='coerce').fillna(0)

    encoders = {}
    cat_cols  = ['HANDSET', 'STATUT', 'CLASSE_CANAL']
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_ENC'] = le.fit_transform(df[col].astype(str))
        encoders[col]    = le

    NUM_FEATS = ['ANC_M', 'ANC_J', 'ID_OFFRE', 'CODE_REGION']
    CAT_FEATS = [c + '_ENC' for c in cat_cols]
    ALL_FEATS = NUM_FEATS + CAT_FEATS

    X = df[ALL_FEATS].apply(pd.to_numeric, errors='coerce').fillna(0)
    return X, ALL_FEATS, encoders

# ─── 4. CROSS-VALIDATION ──────────────────────────────────
def cross_validate(X, y, model_name):
    ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    params = dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=float(ratio),
        eval_metric='auc', random_state=42, verbosity=0
    )
    model  = XGBClassifier(**params)
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_aucs, fold_aps, fold_accs, fold_f1s = [], [], [], []

    print(f"\n  {'Fold':>4} | {'AUC':>8} | {'AP':>8} | {'Acc':>8} | {'F1':>8}")
    print(f"  {'-'*48}")

    for i, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        model.fit(X_tr, y_tr, verbose=False)
        yp  = model.predict(X_te)
        ypr = model.predict_proba(X_te)[:,1]
        auc = roc_auc_score(y_te, ypr)
        ap  = average_precision_score(y_te, ypr)
        acc = accuracy_score(y_te, yp)
        f1  = f1_score(y_te, yp, zero_division=0)
        fold_aucs.append(auc); fold_aps.append(ap)
        fold_accs.append(acc); fold_f1s.append(f1)
        print(f"  {i:>4} | {auc:>8.4f} | {ap:>8.4f} | "
              f"{acc:>8.4f} | {f1:>8.4f}")

    print(f"  {'-'*48}")
    print(f"  {'Moy':>4} | {np.mean(fold_aucs):>8.4f} | {np.mean(fold_aps):>8.4f} | "
          f"{np.mean(fold_accs):>8.4f} | {np.mean(fold_f1s):>8.4f}")
    print(f"  {'Std':>4} | {np.std(fold_aucs):>8.4f} | {np.std(fold_aps):>8.4f} | "
          f"{np.std(fold_accs):>8.4f} | {np.std(fold_f1s):>8.4f}")

    # Entraînement final sur 100% des données
    model.fit(X, y, verbose=False)

    metrics = {
        'auc_cv_mean': round(float(np.mean(fold_aucs)), 4),
        'auc_cv_std' : round(float(np.std(fold_aucs)),  4),
        'ap_cv_mean' : round(float(np.mean(fold_aps)),   4),
        'acc_cv_mean': round(float(np.mean(fold_accs)),  4),
        'f1_cv_mean' : round(float(np.mean(fold_f1s)),   4),
        'n_folds'    : N_FOLDS,
        'fold_aucs'  : [round(float(a), 4) for a in fold_aucs],
        'fold_f1s'   : [round(float(f), 4) for f in fold_f1s],
    }
    return model, metrics, params

# ─── 5. MLOPS — TRACKING MLFLOW ───────────────────────────
def log_to_mlflow(model, metrics, params, features,
                  X, y, model_name, run_tag):
    """
    Enregistre dans MLflow :
      - Hyperparamètres XGBoost
      - Métriques de chaque fold (AUC, AP, Acc, F1)
      - Le modèle XGBoost lui-même avec signature
      - Tags : nom du modèle, date, version
    """
    with mlflow.start_run(run_name=f"{model_name}_{run_tag}") as run:
        run_id = run.info.run_id

        # ── Tags ──────────────────────────────────────────
        mlflow.set_tag("model_name",  model_name)
        mlflow.set_tag("date",        datetime.now().strftime("%Y-%m-%d %H:%M"))
        mlflow.set_tag("n_folds",     str(N_FOLDS))
        mlflow.set_tag("projet",      "PFE Tunisie Telecom")
        mlflow.set_tag("methode",     "StratifiedKFold")

        # ── Hyperparamètres ────────────────────────────────
        mlflow.log_params({
            "n_estimators"    : params["n_estimators"],
            "max_depth"       : params["max_depth"],
            "learning_rate"   : params["learning_rate"],
            "subsample"       : params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "scale_pos_weight": round(params["scale_pos_weight"], 3),
            "n_folds"         : N_FOLDS,
        })

        # ── Métriques moyennes ─────────────────────────────
        mlflow.log_metrics({
            "auc_cv_mean"  : metrics["auc_cv_mean"],
            "auc_cv_std"   : metrics["auc_cv_std"],
            "ap_cv_mean"   : metrics["ap_cv_mean"],
            "acc_cv_mean"  : metrics["acc_cv_mean"],
            "f1_cv_mean"   : metrics["f1_cv_mean"],
        })

        # ── Métriques par fold ─────────────────────────────
        for i, (auc, f1) in enumerate(
                zip(metrics["fold_aucs"], metrics["fold_f1s"]), 1):
            mlflow.log_metrics({
                f"fold_{i}_auc": auc,
                f"fold_{i}_f1" : f1,
            })

        # ── Graphique AUC par fold ─────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        folds = [f"Fold {i}" for i in range(1, N_FOLDS+1)]
        ax.bar(folds, metrics["fold_aucs"],
               color="#004a99", edgecolor="white")
        ax.axhline(metrics["auc_cv_mean"], color="red",
                   linestyle="--", label=f"Moy={metrics['auc_cv_mean']:.4f}")
        for i, v in enumerate(metrics["fold_aucs"]):
            ax.text(i, v + 0.003, f"{v:.4f}",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_title(f"AUC par Fold — {model_name}", fontweight="bold")
        ax.set_ylabel("AUC ROC"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(max(0, min(metrics["fold_aucs"])-0.05), 1.0)
        plt.tight_layout()
        plot_path = f"outputs/model_report/cv_{model_name.replace(' ','_')}.png"
        os.makedirs("outputs/model_report", exist_ok=True)
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(plot_path, artifact_path="plots")

        # ── Modèle MLflow ──────────────────────────────────
        signature = infer_signature(X, model.predict(X))
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            # registered_model_name retiré : nécessite un backend SQLite/PostgreSQL
            # incompatible avec le store local file://. Le modèle reste accessible
            # via joblib (model_forfait_cv.joblib / model_churn_cv.joblib).
            input_example=X.head(3),
        )

        print(f"  Run ID   : {run_id}")
        print(f"  Run Name : {model_name}_{run_tag}")

    return run_id

# ─── 6. MLOPS — MONITORING / DRIFT ───────────────────────
def check_drift(model_name, new_auc):
    """
    Compare l'AUC du run actuel avec le meilleur run précédent.
    Si la dégradation dépasse DRIFT_THRESHOLD → alerte Model Drift.
    """
    os.makedirs("outputs", exist_ok=True)
    history = {}
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            history = json.load(f)

    prev_auc = history.get(model_name, {}).get("best_auc", None)
    drift_detected = False
    drift_msg = ""

    if prev_auc is not None:
        delta = new_auc - prev_auc
        if delta < -DRIFT_THRESHOLD:
            drift_detected = True
            drift_msg = (
                f"  ⚠️  MODEL DRIFT DÉTECTÉ — {model_name}\n"
                f"     AUC précédent : {prev_auc:.4f}\n"
                f"     AUC actuel    : {new_auc:.4f}\n"
                f"     Dégradation   : {delta:.4f} (seuil={DRIFT_THRESHOLD})\n"
                f"     → Réentraînement recommandé sur données récentes"
            )
            print(f"\n{drift_msg}")
        elif delta >= 0:
            print(f"  Modèle amélioré : +{delta:.4f} vs run précédent")
        else:
            print(f"  Variation légère : {delta:.4f} (dans le seuil)")
    else:
        print(f"  Premier run — pas de comparaison possible")

    # Mettre à jour l'historique
    if model_name not in history:
        history[model_name] = {}
    history[model_name]["last_auc"]      = float(new_auc)
    history[model_name]["last_run_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    if prev_auc is None or new_auc >= prev_auc:
        history[model_name]["best_auc"] = float(new_auc)

    with open(METRICS_FILE, "w") as f:
        json.dump(history, f, indent=2)

    return drift_detected, drift_msg

# ─── 7. MLOPS — RAPPORT TEXTE ─────────────────────────────
def generate_mlops_report(results):
    """
    Génère un rapport MLOps complet dans outputs/mlops_report.txt
    Contient : résumé des runs, métriques, statut drift, historique
    """
    os.makedirs("outputs", exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 60,
        "  RAPPORT MLOPS — Tunisie Telecom PFE",
        f"  Généré le : {now}",
        "=" * 60,
        "",
        f"  Expérience MLflow  : {EXPERIMENT_NAME}",
        f"  Dossier MLflow     : {os.path.abspath(MLFLOW_DIR)}",
        f"  Méthode CV         : StratifiedKFold ({N_FOLDS} folds)",
        f"  Seuil drift        : {DRIFT_THRESHOLD}",
        "",
    ]

    for r in results:
        m = r['metrics']
        lines += [
            "─" * 60,
            f"  MODÈLE : {r['model_name']}",
            f"  Run ID : {r['run_id']}",
            "─" * 60,
            f"  AUC CV Moyenne  : {m['auc_cv_mean']:.4f} ± {m['auc_cv_std']:.4f}",
            f"  AP  CV Moyenne  : {m['ap_cv_mean']:.4f}",
            f"  Acc CV Moyenne  : {m['acc_cv_mean']:.4f}",
            f"  F1  CV Moyenne  : {m['f1_cv_mean']:.4f}",
            "",
            "  AUC par fold :",
        ]
        for i, auc in enumerate(m['fold_aucs'], 1):
            bar = "█" * int(auc * 30)
            lines.append(f"    Fold {i} : {auc:.4f}  {bar}")

        drift_status = "⚠️  DRIFT DÉTECTÉ" if r['drift'] else "✅ STABLE"
        lines += [
            "",
            f"  Statut Drift    : {drift_status}",
            f"  Modèle sauvegardé : {r['model_file']}",
            "",
        ]

    # Historique des métriques
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            history = json.load(f)
        lines += [
            "=" * 60,
            "  HISTORIQUE DES MÉTRIQUES",
            "=" * 60,
        ]
        for model_name, data in history.items():
            lines += [
                f"  {model_name}",
                f"    Meilleur AUC  : {data.get('best_auc', 'N/A')}",
                f"    Dernier AUC   : {data.get('last_auc', 'N/A')}",
                f"    Dernier run   : {data.get('last_run_date', 'N/A')}",
                "",
            ]

    lines += [
        "=" * 60,
        "  COMMANDES MLFLOW UI",
        "─" * 60,
        "  Pour visualiser les expériences dans le navigateur :",
        "  $ mlflow ui",
        "  Ouvrir : http://localhost:5000",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    with open(MLOPS_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n  Rapport MLOps sauvegardé : {MLOPS_REPORT}")
    return report_text

# ─── 8. SAUVEGARDE JOBLIB ─────────────────────────────────
def save_model(model, encoders, features, metrics, path, label, run_id):
    artifact = {
        'model'        : model,
        'encoders'     : encoders,
        'feature_names': features,
        'metrics'      : metrics,
        'label'        : label,
        'mlflow_run_id': run_id,
        'trained_at'   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(artifact, path)
    print(f"  Joblib sauvegardé : {path}")

# ─── PIPELINE PRINCIPAL ───────────────────────────────────
def run():
    print(f"\n{'='*58}")
    print("  ENTRAÎNEMENT + MLOPS — Tunisie Telecom PFE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*58}")

    # Setup MLflow — compatible Windows et Linux
    # pathlib.Path.as_uri() génère automatiquement file:///C:/... sur Windows
    # et file:///home/... sur Linux, ce que MLflow accepte dans les 2 cas
    from pathlib import Path
    mlflow_uri = Path(os.path.abspath(MLFLOW_DIR)).as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"\n  MLflow Experiment : {EXPERIMENT_NAME}")
    print(f"  Run tag           : {run_tag}")
    print(f"  Tracking URI      : {mlflow.get_tracking_uri()}")

    # Charger les données
    print(f"\n{'─'*58}")
    print("  CHARGEMENT DES DONNÉES")
    print(f"{'─'*58}")
    path  = find_data_file()
    df    = load_data(path)
    X, features, encoders = prepare_features(df)
    y_forfait = df['TARGET']
    y_churn   = df['FLAG_CHURN']

    results = []

    # ── MODÈLE 1 : Activation Forfait DATA ────────────────
    print(f"\n{'='*58}")
    print("  MODÈLE 1 — ACTIVATION FORFAIT DATA")
    print(f"{'='*58}")
    model_f, metrics_f, params_f = cross_validate(X, y_forfait,
                                                    "Activation Forfait DATA")
    print(f"\n  Logging dans MLflow...")
    run_id_f = log_to_mlflow(model_f, metrics_f, params_f, features,
                              X, y_forfait,
                              "Activation_Forfait_DATA", run_tag)
    drift_f, msg_f = check_drift("Activation_Forfait_DATA",
                                  metrics_f['auc_cv_mean'])
    save_model(model_f, encoders, features, metrics_f,
               MODEL_FORFAIT, "Activation Forfait DATA", run_id_f)
    results.append({
        'model_name': "Activation_Forfait_DATA",
        'run_id'    : run_id_f,
        'metrics'   : metrics_f,
        'drift'     : drift_f,
        'model_file': MODEL_FORFAIT,
    })

    # ── MODÈLE 2 : Prédiction Churn ───────────────────────
    print(f"\n{'='*58}")
    print("  MODÈLE 2 — PRÉDICTION CHURN")
    print(f"{'='*58}")
    model_c, metrics_c, params_c = cross_validate(X, y_churn,
                                                    "Prédiction Churn")
    print(f"\n  Logging dans MLflow...")
    run_id_c = log_to_mlflow(model_c, metrics_c, params_c, features,
                              X, y_churn,
                              "Prediction_Churn", run_tag)
    drift_c, msg_c = check_drift("Prediction_Churn",
                                  metrics_c['auc_cv_mean'])
    save_model(model_c, encoders, features, metrics_c,
               MODEL_CHURN, "Prédiction Churn", run_id_c)
    results.append({
        'model_name': "Prediction_Churn",
        'run_id'    : run_id_c,
        'metrics'   : metrics_c,
        'drift'     : drift_c,
        'model_file': MODEL_CHURN,
    })

    # ── Rapport MLOps ──────────────────────────────────────
    print(f"\n{'─'*58}")
    print("  GÉNÉRATION DU RAPPORT MLOPS")
    print(f"{'─'*58}")
    report = generate_mlops_report(results)

    # ── Résumé final ───────────────────────────────────────
    print(f"\n{'='*58}")
    print("  RÉSUMÉ FINAL")
    print(f"{'='*58}")
    for r in results:
        m = r['metrics']
        s = "DRIFT" if r['drift'] else "STABLE"
        print(f"  {r['model_name']:<30}")
        print(f"    AUC : {m['auc_cv_mean']:.4f} ± {m['auc_cv_std']:.4f}"
              f"  |  F1 : {m['f1_cv_mean']:.4f}"
              f"  |  Statut : {s}")
        print(f"    Run ID : {r['run_id']}")

    print(f"\n  Fichiers générés :")
    print(f"    {MODEL_FORFAIT}")
    print(f"    {MODEL_CHURN}")
    print(f"    {MLOPS_REPORT}")
    print(f"    {METRICS_FILE}")
    print(f"\n  Visualiser dans MLflow UI :")
    print(f"    $ mlflow ui")
    print(f"    Ouvrir : http://localhost:5000")
    print(f"{'='*58}")
    print("  TERMINÉ AVEC SUCCÈS")
    print(f"{'='*58}\n")

if __name__ == "__main__":
    run()