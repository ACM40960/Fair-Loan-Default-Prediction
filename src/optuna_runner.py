# --- Optuna tuning for Recall @ Precision ≥ p_floor, GPU where possible ---
# Models covered (≥5 trials each recommended):
#   - SVM (RBF; ThunderSVM GPU if available, else sklearn CPU)
#   - XGBoost (gpu_hist if available, else hist)
#   - Logistic Regression (balanced, L2)
#   - Random Forest (CPU)
#   - KNN (probabilistic)
#
# Notes:
# - Use `pre_tree` for tree models (typically One-Hot).
# - Use `pre_linear` for SVM/LR/KNN (ideally One-Hot + StandardScaler).
# - Sampling during CV is controlled via `sampling_type_cv` and `target_share_cv`.
# - Objective is recall at the best threshold that satisfies Precision ≥ p_floor.

# ============================
# Optuna Tuning (Macro Recall @ Precision Floor) + ThunderSVM support
# ============================
import numpy as np
import pandas as pd
import optuna
from typing import Dict, Tuple, Any
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, recall_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import builtins, threading
from optuna.trial import TrialState

# ----- ThunderSVM (GPU/CPU) fallback to sklearn SVC -----
try:
    from thundersvm import SVC as TSVC
    HAVE_TSVMC = True
except Exception:
    TSVC = None
    HAVE_TSVMC = False

# --- GPU capability checks ---
USE_GPU = True  # set False to force CPU

# cuML (GPU RF & LR)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    HAVE_CUML_RF = True
except Exception:
    cuRF = None
    HAVE_CUML_RF = False

try:
    from cuml.linear_model import LogisticRegression as cuLR
    HAVE_CUML_LR = True
except Exception:
    cuLR = None
    HAVE_CUML_LR = False


# Persist across cell re-runs / imports
if not hasattr(builtins, "_PRINTED_FLAGS"):
    builtins._PRINTED_FLAGS = set()
if not hasattr(builtins, "_PRINTED_FLAGS_LOCK"):
    builtins._PRINTED_FLAGS_LOCK = threading.Lock()


def _log_once(tag: str, msg: str, level: str = "info"):
    """
    Print/log `msg` only once per unique `tag`.
    - De-duplicates across notebook cell re-runs.
    - Flushes stdout so you see it during trials.
    - Also logs to Optuna's logger.
    """
    t = str(tag)
    with builtins._PRINTED_FLAGS_LOCK:
        if t in builtins._PRINTED_FLAGS:
            return
        builtins._PRINTED_FLAGS.add(t)

    # Stdout (immediate)
    print(msg, flush=True)

    # Optuna logger (non-fatal if Optuna not initialized)
    try:
        logger = optuna.logging.get_logger("optuna")
        log_fn = getattr(logger, level, logger.info)
        log_fn(msg)
    except Exception:
        pass


# Models
from sklearn.svm import SVC as SKSVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ---------- Core metric helpers ----------
def recall_at_precision_floor(y_true, y_prob, p_floor: float) -> Tuple[float, float]:
    """
    Find the highest threshold whose precision >= p_floor on (y_true, y_prob).
    Returns (pos_recall_at_that_threshold, threshold).
    Note: We use this threshold for *evaluation*, but for macro recall we recompute recalls at that threshold.
    """
    p, r, thr = precision_recall_curve(y_true, y_prob)
    p, r = p[:-1], r[:-1]  # align with thresholds
    idxs = np.where(p >= p_floor)[0]
    if len(idxs) == 0:
        return 0.0, 1.01
    idx = idxs[0]  #least conservative threshold meeting the floor
    return float(r[idx]), float(thr[idx])

"""
    Cross-validated evaluation that:
      1) Fits (preprocess -> [undersample] -> model) on each fold,
      2) Chooses the lowest threshold meeting precision >= p_floor on that fold,
      3) Computes MACRO recall at that threshold,
      4) Returns (mean_macro_recall_across_folds, median_threshold).
    """
def _cv_macro_recall_and_threshold(
    model, X, y, preprocessor, p_floor, folds, seed,
    sampling_type_cv="under_only", target_share_cv=0.20,
):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    macro_recalls, thresholds = [], []

    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        steps = [("preprocess", preprocessor)]
        if sampling_type_cv == "under_only":
            r = target_share_cv / (1.0 - target_share_cv)
            steps.append(("sampler", RandomUnderSampler(sampling_strategy=r, random_state=seed)))

        steps.append(("model", model))
        pipe = ImbPipeline(steps)
        pipe.fit(Xtr, ytr)

        # --- score vector: proba if available, else decision_function, else labels ---
        yscore, out = None, None
        if hasattr(pipe, "predict_proba"):
            try:
                out = pipe.predict_proba(Xva)
            except Exception:
                out = None
        if out is not None:
            yscore = out[:, 1]

        if yscore is None:
            if hasattr(pipe, "decision_function"):
                yscore = pipe.decision_function(Xva)
            else:
                # last resort (weak, but avoids crash)
                yscore = pipe.predict(Xva)

        # CuPy -> NumPy if needed
        try:
            import cupy as cp  # noqa: F401
            if isinstance(yscore, cp.ndarray):
                yscore = cp.asnumpy(yscore)
        except Exception:
            pass
        yscore = np.asarray(yscore).ravel()

        _, thr = recall_at_precision_floor(yva, yscore, p_floor=p_floor)
        yhat = (yscore >= thr).astype(int)
        macro_recalls.append(recall_score(yva, yhat, average="macro"))
        thresholds.append(thr)

    return float(np.mean(macro_recalls)), float(np.median(thresholds))


# ---------- Objectives (optimizing MACRO recall) ----------
def objective_svm(trial, X, y, pre_linear, p_floor, folds, seed, sampling_type_cv, target_share_cv):
    from sklearn.svm import SVC
    params = {
        "C": trial.suggest_float("C", 0.1, 50.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        "kernel": "rbf",
        "probability": False,     # ← faster; we’ll use decision_function
        "cache_size": 500,        
        "random_state": seed,
        "class_weight": None,     
    }
    model = SVC(**params)
    mean_macro, med_thr = _cv_macro_recall_and_threshold(
        model, X, y, pre_linear, p_floor, folds, seed, sampling_type_cv, target_share_cv
    )
    trial.set_user_attr("thr_median", med_thr)
    return mean_macro



def objective_xgb(trial, X, y, pre_tree, p_floor, folds, seed, sampling_type_cv, target_share_cv):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 400, 3000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_float("min_child_weight", 1.0, 20.0),
        "gamma":             trial.suggest_float("gamma", 0.0, 10.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "random_state": seed, "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }
    # GPU tree method (falls back automatically if not available)
    if USE_GPU:
        params["tree_method"] = "gpu_hist"
        params["max_bin"] = trial.suggest_int("max_bin", 63, 511)
        _log_once("XGB_GPU", "[XGB] Using GPU (tree_method='gpu_hist').")

    model = XGBClassifier(**params)
    mean_macro, med_thr = _cv_macro_recall_and_threshold(
        model, X, y, pre_tree, p_floor, folds, seed, sampling_type_cv, target_share_cv
    )
    trial.set_user_attr("thr_median", med_thr)
    return mean_macro


# optional GPU SVD if available; fallback to sklearn (CPU) SVD

try:
    from cuml.decomposition import TruncatedSVD as cuTSVD
    HAVE_CU_TSVD = True
except Exception:
    HAVE_CU_TSVD = False
from sklearn.decomposition import TruncatedSVD as SKTSVD  # CPU fallback (OK, reduces width before GPU)
from sklearn.base import BaseEstimator, TransformerMixin

class AutoSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=128, random_state=42, use_gpu=True):
        self.n_components = int(n_components)
        self.random_state = random_state
        self.use_gpu = bool(use_gpu)
        self._svd = None

    def fit(self, X, y=None):
        n_features = X.shape[1]
        k = max(1, min(int(self.n_components), int(n_features) - 1))  # cap safely
        if self.use_gpu and HAVE_CU_TSVD:
            self._svd = cuTSVD(n_components=k, random_state=self.random_state)
        else:
            self._svd = SKTSVD(n_components=k, random_state=self.random_state)
        self._svd.fit(X, y)
        return self

    def transform(self, X):
        return self._svd.transform(X)
# device-aware cast to float32 (handles NumPy & CuPy)
def _to_float32(Z):
    try:
        import cupy as cp
        if isinstance(Z, cp.ndarray):
            return Z.astype(cp.float32)
    except Exception:
        pass
    return Z.astype(np.float32)

def objective_rf(trial, X, y, pre_tree, p_floor, folds, seed, sampling_type_cv, target_share_cv):
    svd_k = trial.suggest_int("svd_components", 96, 160)
    svd = AutoSVD(n_components=svd_k, random_state=seed, use_gpu=True)

    rf_params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 400),
        "max_depth":    trial.suggest_int("max_depth", 8, 15),
        # rename param to avoid Optuna distribution conflicts
        "max_features": trial.suggest_float("rf_max_features_frac", 0.3, 0.5),
        "bootstrap":    True,
        "n_bins":       trial.suggest_int("n_bins", 64, 120),
        "random_state": seed,
    }

    base = cuRF(**rf_params)
    cast32 = FunctionTransformer(_to_float32, accept_sparse=True, feature_names_out="one-to-one")
    model = make_pipeline(svd, cast32, base)
    _log_once("RF_GPU", "[RF] Using cuML RandomForest (GPU) + AutoSVD.")

    try:
        mean_macro, med_thr = _cv_macro_recall_and_threshold(
            model, X, y, pre_tree, p_floor, folds, seed,
            sampling_type_cv=sampling_type_cv, target_share_cv=target_share_cv
        )
    except Exception as e:
        em = str(e).lower()
        if any(t in em for t in ["cuda", "cuml", "bad_alloc", "out of memory"]):
            _log_once("RF_GPU_OOM", "[RF] GPU OOM/bad_alloc → pruning trial.")
            raise optuna.exceptions.TrialPruned()
        raise

    trial.set_user_attr("thr_median", med_thr)
    return mean_macro





def objective_lr(trial, X, y, pre_linear, p_floor, folds, seed, sampling_type_cv, target_share_cv):
    if USE_GPU and HAVE_CUML_LR:
        # cuML LR (GPU) — L2 regularized logistic
        C = trial.suggest_float("C", 1e-4, 1e3, log=True)
        params = {"C": C, "max_iter": 5000, "fit_intercept": True}
        model = cuLR(**params)
        _log_once("LR_GPU", "[LR] Using cuML LogisticRegression (GPU).")
    else:
        # sklearn LR (CPU)
        params = {
            "penalty": "l2",
            "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
            "solver": "lbfgs",
            "max_iter": 5000,
            "n_jobs": -1,
            "class_weight": None,
            "random_state": seed,
        }
        model = LogisticRegression(**params)
        _log_once("LR_CPU", "[LR] Using sklearn LogisticRegression (CPU).")

    mean_macro, med_thr = _cv_macro_recall_and_threshold(
        model, X, y, pre_linear, p_floor, folds, seed, sampling_type_cv, target_share_cv
    )
    trial.set_user_attr("thr_median", med_thr)
    return mean_macro



# ---------- Public API ----------
from optuna.trial import TrialState

def run_all(
    X, y,
    pre_linear, pre_tree,
    trials_cfg: Dict[str, int],
    seed: int = 42,
    sampling_type_cv: str = "under_only",
    target_share_cv: float = 0.20,
    folds: int = 3,
    storage: str = None,
    p_floor: float = 0.35,
):
    """
    Runs Optuna for selected models; resumes from storage and only adds the *delta*
    of trials needed. Returns:
        studies, winner_name, winner_params, winner_thr
    """
    # Only the 4 models you’re using now
    dispatch = {
        "SVM": (objective_svm,  pre_linear),
        "XGB": (objective_xgb,  pre_tree),
        "RF":  (objective_rf,   pre_tree),
        "LR":  (objective_lr,   pre_linear),
    }

    # If you changed RF param distributions (e.g., max_features categorical → float),
    # give RF a new study name to avoid distribution conflicts:
    study_name_map = {
        # "RF": "RF_macro_recall_atP_v2",   # ← uncomment if needed
    }

    studies, best_scores, best_params, best_thrs = {}, {}, {}, {}

    for name, n_trials in trials_cfg.items():
        if name not in dispatch or int(n_trials) <= 0:
            continue

        obj_fn, preproc = dispatch[name]
        study_name = study_name_map.get(name, f"{name}_macro_recall_atP")

        # Create/load study (with pruner if you like)
        if storage:
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
            )
        else:
            study = optuna.create_study(direction="maximize", study_name=study_name)

        def _wrapped(trial):
            return obj_fn(
                trial, X, y, preproc, p_floor, folds, seed, sampling_type_cv, target_share_cv
            )

        # Run only the needed additional trials (auto-skip when already complete)
        n_complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
        to_run = max(0, int(n_trials) - n_complete)
        if to_run == 0:
            _log_once(f"{name}_SKIP", f"[{name}] Already completed {n_complete} trials; skipping optimize().")
        else:
            _log_once(f"{name}_RESUME", f"[{name}] Resuming: {n_complete} done, running +{to_run} trials (target={int(n_trials)}).")
            study.optimize(_wrapped, n_trials=to_run)

        studies[name] = study

        if study.best_trial is not None:
            best_scores[name] = float(study.best_value)
            best_params[name] = study.best_params
            best_thrs[name] = float(study.best_trial.user_attrs.get("thr_median", 0.5))
        else:
            _log_once(f"{name}_EMPTY", f"[{name}] No completed trials found.")

    if not best_scores:
        raise ValueError("No studies were run or no completed trials; check trials_cfg and storage.")

    winner_name = max(best_scores, key=best_scores.get)
    winner_params = best_params[winner_name]
    winner_thr = best_thrs[winner_name]
    return studies, winner_name, winner_params, winner_thr



def summarize_optuna(studies, p_floor=0.35):
    metric_col = f"best_macro_recall_at_P≥{int(100*p_floor)}%"
    rows = []
    for name, st in studies.items():
        if not st or not st.trials or st.best_trial is None:
            continue
        bt = st.best_trial
        n_complete = sum(t.state == TrialState.COMPLETE for t in st.trials)
        rows.append({
            "model": name,
            metric_col: float(st.best_value),
            "best_thr": float(bt.user_attrs.get("thr_median", np.nan)),
            "n_trials": n_complete,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(metric_col, ascending=False)
        print(df.to_string(index=False))
    else:
        print("No completed trials found.")
    return df