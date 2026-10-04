"""
main.py
-------
Point d'entrée du pipeline de churn prediction.

Deux usages :
    1. En CLI (Atelier 2) :
        python main.py --action all --data Churn_Modelling.csv
    2. En import (Atelier 3) : pipeline_prefect.py appelle les fonctions
       run_install(), run_prepare(), run_train(), run_save(), run_load(),
       run_evaluate().
"""

import argparse
import subprocess
import sys
import os

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
REPO_URL = "https://github.com/nidhaltaboui/mlops_project.git"


def run_clone(repo_url=REPO_URL, dest="."):
    """Clone le dépôt, ou le met à jour (git pull) s'il existe déjà."""
    if os.path.isdir(os.path.join(dest, ".git")):
        cmd = ["git", "-C", dest, "pull", "--ff-only"]
    else:
        cmd = ["git", "clone", repo_url, dest]
    subprocess.run(cmd, check=True)


def run_install(requirements="requirements.txt"):
    """Installe les dépendances listées dans requirements.txt."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements],
        check=True,
    )


def run_prepare(data_path=DATA_PATH):
    """Prépare les données. Retourne x_train, x_test, y_train, y_test,
    encoder, scaler, feature_columns."""
    return prepare_data(data_path)


def run_train(x_train, y_train, n_estimators=100, learning_rate=0.1):
    """Entraîne le modèle et le retourne."""
    return train_model(
        x_train,
        y_train,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
    )


def run_save(model, model_path=MODEL_PATH):
    """Sauvegarde le modèle et retourne son chemin."""
    save_model(model, model_path)
    return model_path


def run_load(model_path=MODEL_PATH):
    """Charge et retourne un modèle sauvegardé."""
    return load_model(model_path)


def run_evaluate(model, x_test, y_test):
    """Évalue le modèle et retourne le dictionnaire de métriques."""
    return evaluate_model(model, x_test, y_test)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline modulaire de churn prediction (MLOps)."
    )
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["prepare", "train", "evaluate", "save", "load", "all"],
        help="Étape du pipeline à exécuter.",
    )
    parser.add_argument("--data", type=str, default=DATA_PATH)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    if args.action == "load":
        model = run_load(args.model_path)
        _, x_test, _, y_test, _, _, _ = run_prepare(args.data)
        run_evaluate(model, x_test, y_test)
        return

    x_train, x_test, y_train, y_test, encoder, scaler, feature_columns = run_prepare(
        args.data
    )

    if args.action == "prepare":
        print("Données préparées avec succès.")
        print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
        return

    model = run_train(x_train, y_train, args.n_estimators, args.learning_rate)

    if args.action == "train":
        print("Modèle entraîné avec succès.")
    if args.action in ("evaluate", "all"):
        run_evaluate(model, x_test, y_test)
    if args.action in ("save", "all"):
        run_save(model, args.model_path)
        import joblib

        joblib.dump(encoder, "gender_encoder.joblib")
        joblib.dump(scaler, "scaler.joblib")
        joblib.dump(feature_columns, "feature_columns.joblib")
        print("Encodeur, scaler et colonnes sauvegardés.")


if __name__ == "__main__":
    main()
