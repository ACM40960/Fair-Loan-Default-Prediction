# ========================= ML_modeling.py (Optuna-friendly) =========================
import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc
)

# Optional imblearn (for sampler steps + Pipeline type preservation)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    from imblearn.under_sampling import RandomUnderSampler  # type: ignore
    from imblearn.over_sampling import RandomOverSampler   # type: ignore
    from imblearn.over_sampling import SMOTE               # type: ignore
    IMB_OK = True
except Exception:
    ImbPipeline = None  # type: ignore
    RandomUnderSampler = RandomOverSampler = SMOTE = None  # type: ignore
    IMB_OK = False

# Plotting (unchanged)
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

RANDOM_STATE = 42

# -------------------- helpers: columns & transforms --------------------
# If you've already log-transformed some columns upstream, list them here to skip re-logging
already_logged = {'AMT_CREDIT', 'AMT_GOODS_PRICE'}

def build_column_groups(X_like: pd.DataFrame):
    num_cols = X_like.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_like.columns if c not in num_cols]
    return num_cols, cat_cols

def pick_log_cols(num_cols):
    # Example rule: log all "AMT*" numeric columns except the ones pre-logged
    return [c for c in num_cols if ('AMT' in c.upper()) and (c not in already_logged)]

def safe_log1p(X):
    X = np.asarray(X, dtype=float)
    X = np.where(X < 0, 0, X)  # guard against negatives
    return np.log1p(X)

def make_ohe():
    # Handle scikit-learn API differences (sparse_output added in 1.2)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessors(X_like: pd.DataFrame):
    """
    Returns:
      pre_ordinal, pre_onehot, num_cols, cat_cols, log_cols, scale_cols
    """
    num_cols, cat_cols = build_column_groups(X_like)
    log_cols = pick_log_cols(num_cols)
    scale_cols = [c for c in num_cols if c not in log_cols]

    log_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(safe_log1p, feature_names_out='one-to-one')),
        ('sc', StandardScaler())
    ])
    scale_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler())
    ])
    cat_ord_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    cat_ohe_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', make_ohe())
    ])

    pre_ordinal = ColumnTransformer(
        transformers=[
            ('log',   log_pipe,   log_cols),
            ('scale', scale_pipe, scale_cols),
            ('cat',   cat_ord_pipe, cat_cols)
        ],
        remainder='drop'
    )
    pre_onehot = ColumnTransformer(
        transformers=[
            ('log',   log_pipe,   log_cols),
            ('scale', scale_pipe, scale_cols),
            ('cat',   cat_ohe_pipe, cat_cols)
        ],
        remainder='drop'
    )
    return pre_ordinal, pre_onehot, num_cols, cat_cols, log_cols, scale_cols

# -------------------- balance reporter & final fit --------------------
class BalanceReporter(BaseEstimator, TransformerMixin):
    """No-op transformer to record class counts right AFTER any sampler during final fit."""
    def __init__(self):
        self.counts_ = None
        self.minority_share_ = None

    def fit(self, X, y=None):
        if y is not None:
            y = np.asarray(y)
            counts = np.bincount(y.astype(int), minlength=2)
            total = counts.sum() if counts.sum() else 1
            self.counts_ = counts
            self.minority_share_ = counts[1] / total
        return self

    def transform(self, X):
        return X

def _insert_after_sampling(pipeline: Pipeline):
    """
    Clone steps and insert BalanceReporter AFTER any imblearn samplers named
    'under' / 'over' / 'smote'. If none found, insert just before 'pre'.
    Keeps the same Pipeline class (sklearn vs imblearn) as the original.
    """
    steps = list(pipeline.steps)
    new_steps = []
    inserted = False

    for n, s in steps:
        new_steps.append((n, s))
        if n in ("under", "over", "smote") and not inserted:
            new_steps.append(("balance_report", BalanceReporter()))
            inserted = True

    if not inserted:
        # Insert before preprocessor if no explicit sampler
        tmp = []
        added = False
        for n, s in new_steps:
            if (n == "pre") and not added:
                tmp.append(("balance_report", BalanceReporter()))
                added = True
            tmp.append((n, s))
        new_steps = tmp

    PipeCls = ImbPipeline if (ImbPipeline is not None and isinstance(pipeline, ImbPipeline)) else Pipeline
    return PipeCls(new_steps)

def final_fit_with_balance(pipeline: Pipeline, X_train, y_train, X_test, y_test, threshold: float):
    """
    Fit a COPY of the pipeline with a BalanceReporter placed after sampling
    (or just before preprocessor if no sampler). Returns:
      proba_test, balance_dict, metrics_dict, fitted_pipeline
    """
    p2 = _insert_after_sampling(pipeline)
    p2.fit(X_train, y_train)
    proba_test = p2.predict_proba(X_test)[:, 1]

    br = dict(p2.named_steps).get("balance_report", None)
    balance = {}
    if br is not None and getattr(br, "counts_", None) is not None:
        c = br.counts_
        total = int(c.sum()) if c.sum() else 1
        balance = {
            "post_sampling_n0": int(c[0]),
            "post_sampling_n1": int(c[1]),
            "post_sampling_minority_share": float(c[1] / total)
        }

    preds = (proba_test >= float(threshold)).astype(int)
    metrics = {
        "test_P": precision_score(y_test, preds, zero_division=0),
        "test_R": recall_score(y_test, preds, zero_division=0),
        "test_F1": f1_score(y_test, preds, zero_division=0),
        "test_AP": average_precision_score(y_test, proba_test),
        "test_AUC": roc_auc_score(y_test, proba_test),
    }
    return proba_test, balance, metrics, p2

# -------------------- build tuned pipelines (simple, robust) --------------------
def build_tuned_pipe(
    model_name: str,
    params: dict,
    preprocessor,
    y_train: pd.Series,
    sampling_type: str = "hybrid",   # "under_only" | "hybrid" | "mixture" | "none"
    target_share: float = 0.24
):
    """
    Construct a final training pipeline using tuned params from Optuna.

    Pass `preprocessor` as either pre_ordinal or pre_onehot.
    """
    name = model_name.strip().upper()
    # Sampler steps
    sampler_steps = []
    if IMB_OK and sampling_type != "none":
        n1 = int((y_train == 1).sum()); n0 = int((y_train != 1).sum())
        n0_prime = int(np.floor(n1 * (1 - target_share) / target_share))
        n0_prime = max(1, min(n0, n0_prime))
        n1_final = int(np.ceil(target_share / (1 - target_share) * n0_prime))
        n1_final = max(n1, n1_final)

        if sampling_type == "under_only":
            rus = RandomUnderSampler(sampling_strategy={0: n0_prime, 1: n1}, random_state=RANDOM_STATE)
            sampler_steps = [('under', rus)]
        elif sampling_type == "hybrid":
            rus = RandomUnderSampler(sampling_strategy={0: n0_prime, 1: n1}, random_state=RANDOM_STATE)
            ros = RandomOverSampler(sampling_strategy={1: n1_final}, random_state=RANDOM_STATE)
            sampler_steps = [('under', rus), ('over', ros)]
        elif sampling_type == "mixture":
            rus = RandomUnderSampler(sampling_strategy={0: n0_prime, 1: n1}, random_state=RANDOM_STATE)
            smt = SMOTE(sampling_strategy=target_share, random_state=RANDOM_STATE)
            sampler_steps = [('under', rus), ('smote', smt)]

    # Estimator
    if name == "SVM":
        from sklearn.svm import SVC
        est = SVC(probability=False, class_weight="balanced", random_state=RANDOM_STATE, **params)
        last = ("svm", est)
    elif name == "XGB":
        from xgboost import XGBClassifier
        # Ensure useful defaults
        pos = y_train.value_counts()
        spw = float(pos.get(0, 1) / max(1, pos.get(1, 1)))
        p = dict(params)
        p.setdefault("tree_method", "hist")
        p.setdefault("objective", "binary:logistic")
        p.setdefault("eval_metric", "aucpr")
        p.setdefault("random_state", RANDOM_STATE)
        p.setdefault("n_jobs", -1)
        p.setdefault("scale_pos_weight", spw)
        est = XGBClassifier(**p)
        last = ("xgb", est)
    elif name == "LGBM":
        from lightgbm import LGBMClassifier
        p = dict(params)
        p.setdefault("random_state", RANDOM_STATE)
        p.setdefault("n_jobs", -1)
        p.setdefault("class_weight", "balanced")
        est = LGBMClassifier(**p)
        last = ("lgbm", est)
    elif name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        p = dict(params)
        p.setdefault("random_state", RANDOM_STATE)
        p.setdefault("n_jobs", -1)
        est = RandomForestClassifier(**p)
        last = ("rf", est)
    elif name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        est = KNeighborsClassifier(**params)
        last = ("knn", est)
    else:
        raise ValueError(f"Unknown model_name '{model_name}'")

    base_steps = sampler_steps + [("pre", preprocessor), last]
    PipeCls = ImbPipeline if (IMB_OK and sampler_steps) else Pipeline
    return PipeCls(base_steps)

# -------------------- plotting (unchanged) --------------------
def build_curves(y_true, proba_dict):
    curves = {}
    for name, p in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        curves[name] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    return curves

def plot_results_curves_and_metrics(y_test, proba_dict, metrics_table):
    curves = build_curves(y_test, proba_dict)

    # ROC
    fig1 = go.Figure()
    fig1.add_shape(type='line', line=dict(color='green', dash='dash'), x0=0, x1=1, y0=0, y1=1)
    for name, c in curves.items():
        fig1.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode='lines', name=f"{name} AUC={c['auc']:.3f}"))
    fig1.update_layout(title='ROC curve', xaxis_title='FPR', yaxis_title='TPR', width=700, height=500)

    # Probability histogram (use the first model as representative)
    any_name = next(iter(proba_dict))
    y_proba = proba_dict[any_name]
    fig2 = px.histogram(x=y_proba, color=y_test.astype(int), nbins=50,
                        labels=dict(color='TARGET', x='Predicted probability'))
    fig2.update_traces(opacity=0.75)
    fig2.update_layout(barmode='overlay', title=f'Probability histogram ({any_name})')

    # Metrics trend
    mt = metrics_table.copy()
    if isinstance(mt, pd.Series):
        mt = mt.to_frame(name="model")
    if "model" in mt.index:
        mt = mt.T
    if "model" in mt.columns:
        mt = mt.set_index("model")
    x_models = list(mt.columns)

    fig3 = make_subplots(rows=1, cols=1)
    for metric in ["test_P", "test_R", "test_F1", "test_AUC", "test_AP"]:
        if metric in mt.index:
            fig3.add_trace(go.Scatter(y=mt.loc[metric, x_models], x=x_models, mode="lines+markers", name=metric))
    fig3.update_layout(title="Model metrics", xaxis_title="Model", yaxis_title="Score")

    fig1.show(); fig2.show(); fig3.show()

# -------------------- save / load & "pick best" helpers --------------------
def save_model(fitted_pipeline, path: str):
    """Persist a fitted pipeline to disk (Joblib)."""
    import joblib
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(fitted_pipeline, path)
    return path

def load_model(path: str):
    """Load a fitted pipeline from disk."""
    import joblib
    return joblib.load(path)

def pick_best(metrics_by_model: dict, key: str = "test_AP"):
    """
    metrics_by_model: {"XGB": {"test_AP": ..., "test_F1": ...}, ...}
    key: one of "test_AP", "test_F1", "test_AUC", etc.
    """
    if not metrics_by_model:
        raise ValueError("metrics_by_model is empty.")
    best_name = max(metrics_by_model, key=lambda k: metrics_by_model[k].get(key, -np.inf))
    return best_name, metrics_by_model[best_name]

# -------------------- (optional) minimal MLflow logger --------------------
def run_experiment(model_name, fitted_pipeline, params, test_proba, test_metrics, threshold=None):
    """No-op if MLflow is not installed."""
    try:
        import mlflow
        from mlflow import sklearn as mlflow_sklearn
    except Exception:
        return  # silently skip if MLflow not installed

    with mlflow.start_run(run_name=model_name):
        try:
            if params: mlflow.log_params(params)
        except Exception:
            pass
        if threshold is not None:
            try:
                mlflow.log_metric("threshold", float(threshold))
            except Exception:
                pass
        if isinstance(test_metrics, dict):
            try:
                mlflow.log_metrics({k: float(v) for k, v in test_metrics.items()})
            except Exception:
                pass
        try:
            mlflow_sklearn.log_model(fitted_pipeline, "model")
        except Exception:
            pass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, auc, precision_score, recall_score
)

def plot_results_curves_and_metrics(y_true, probas_dict, summary_df=None, p_floor=None, thr_used=None):
    """
    probas_dict: {"ModelName": y_proba_array}
    summary_df:  optional; not required
    p_floor:     optional float to draw precision floor line
    thr_used:    optional decision threshold to mark on PR curve
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Precision–Recall + ROC for each model
    for name, y_prob in probas_dict.items():
        y_prob = np.asarray(y_prob).ravel()

        P, R, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        axes[0].plot(R[:-1], P[:-1], label=f"{name} (AP={ap:.3f})")

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        # Mark the chosen threshold point on PR
        if thr_used is not None:
            y_hat_thr = (y_prob >= float(thr_used)).astype(int)
            p_at = precision_score(y_true, y_hat_thr, zero_division=0)
            r_at = recall_score(y_true, y_hat_thr)
            axes[0].scatter([r_at], [p_at], marker="o")
            axes[0].annotate(
                f"thr={float(thr_used):.3f}\nP={p_at:.2f}, R={r_at:.2f}",
                (r_at, p_at), xytext=(5, 5), textcoords="offset points", fontsize=8
            )

    if p_floor is not None:
        axes[0].axhline(float(p_floor), ls="--", label=f"P floor={float(p_floor):.2f}")

    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision"); axes[0].set_title("Precision–Recall")
    axes[0].legend(loc="lower left")

    axes[1].plot([0, 1], [0, 1], ls="--")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].set_title("ROC")
    axes[1].legend(loc="lower right")

    plt.tight_layout(); plt.show()

# Public API
__all__ = [
    "build_preprocessors",
    "BalanceReporter", "_insert_after_sampling", "final_fit_with_balance",
    "build_tuned_pipe",
    "build_curves", "plot_results_curves_and_metrics",
    "save_model", "load_model", "pick_best",
    "run_experiment", "RANDOM_STATE"
]
# ========================= end of file =========================
