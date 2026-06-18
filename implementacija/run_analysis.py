"""Glavna skripta za pokretanje svih eksperimenata."""

import sys
import numpy as np
from config import SMOTE_K_VALUES, METRICS
from evaluation.experiment_runner import run_experiment
from data.generate_synthetic import generate_synthetic_sets, load_all_datasets, print_dataset_summary


SMOTE_ALL = [
    "SMOTE", "Borderline-SMOTE1", "Borderline-SMOTE2", "ADASYN",
    "SafeLevel-SMOTE", "KMeans-SMOTE", "SVM-SMOTE",
    "SMOTE-ENN", "SMOTE-Tomek", "G-SMOTE", "Random-SMOTE", "PolynomFit-SMOTE",
]

CLASSIFIERS_ALL = ["dt", "rf", "lr", "svm", "knn", "gnb", "mlp"]


def main():
    use_smote = sys.argv[1:] if len(sys.argv) > 1 else ["SMOTE", "ADASYN"]
    use_classifiers = ["rf", "lr"]
    k_vals = [5] if len(sys.argv) > 2 else ([5] if len(use_smote) > 4 else SMOTE_K_VALUES)
    datasets_all = len(sys.argv) > 2

    print("=" * 60)
    print("  SMOTE Experimental Analysis")
    print("=" * 60)

    datasets = {}
    print("\n[1] Generating synthetic datasets...")
    synth = generate_synthetic_sets()
    datasets.update(synth)

    print("\n[2] Loading real datasets...")
    real = load_all_datasets()
    datasets.update(real)

    print("\n[3] Dataset summary:")
    print_dataset_summary(datasets)

    print(f"\n[4] Running experiments...")
    print(f"    SMOTE variants: {len(use_smote)}")
    print(f"    Classifiers:    {len(use_classifiers)}")
    print(f"    k values:       {k_vals}")
    print(f"    Datasets:       {len(datasets)}")
    print(f"    Expected rows:  {len(use_smote) * len(use_classifiers) * len(k_vals) * len(datasets) * len(METRICS)}")
    print()

    for name, (X, y, meta) in datasets.items():
        print(f"\n--- {name} ({meta['n_samples']} samples, IR={meta['ir']:.1f}) ---")
        run_experiment(
            X, y, name,
            classifier_names=use_classifiers,
            smote_names=use_smote,
            k_values=k_vals,
        )

    print("\n" + "=" * 60)
    print("  Done. Results saved in results/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()
