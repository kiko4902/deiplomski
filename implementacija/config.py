import os

SMOTE_K_VALUES = [3, 5, 7, 10]
CV_FOLDS = 5
CV_REPEATS = 30
CV_SEED_START = 42

METRICS = ["f1", "g_mean", "auc_roc", "auc_pr", "balanced_accuracy", "mcc", "f2"]
CLASSIFIER_NAMES = ["dt", "rf", "xgboost", "lr", "svm", "knn", "gnb", "mlp"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "data")
REAL_DATA_DIR = os.path.join(DATA_DIR, "real")
SYNTH_DATA_DIR = os.path.join(DATA_DIR, "synthetic")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(SYNTH_DATA_DIR, exist_ok=True)
