"""
model_pipeline.py
------------------
Pipeline modulaire pour le projet de prédiction du churn client (Churn_Modelling.csv).

Ce fichier contient les fonctions réutilisables demandées dans l'Atelier 2 :
    - prepare_data()   : charge et prétraite les données
    - train_model()    : entraîne le modèle (GradientBoostingClassifier)
    - evaluate_model()  : évalue les performances du modèle
    - save_model()      : sauvegarde le modèle entraîné (joblib)
    - load_model()      : charge un modèle sauvegardé

Modèle utilisé : GradientBoostingClassifier (remplace le RandomForestClassifier
du notebook d'origine).
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data(data_path: str):
    """
    Charge et prétraite les données du fichier CSV.

    Étapes effectuées :
        - Chargement du CSV.
        - Encodage de la colonne 'Gender' (LabelEncoder).
        - Suppression des colonnes non pertinentes
          ('Surname', 'Geography', 'RowNumber', 'CustomerId').
        - Séparation features (X) / cible (y).
        - Split train/test (80/20).
        - Standardisation des features numériques (StandardScaler).

    Parameters
    ----------
    data_path : str
        Chemin vers le fichier CSV contenant les données brutes.

    Returns
    -------
    x_train_scaled, x_test_scaled, y_train, y_test : np.ndarray / pd.Series
        Données prêtes pour l'entraînement et le test.
    """
    df = pd.read_csv(data_path)

    # Encodage de la variable catégorielle 'Gender'
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Suppression des colonnes non pertinentes pour l'entraînement
    columns_to_drop = ["Surname", "Geography"]
    df = df.drop(columns_to_drop, axis=1)

    # Séparation features / cible
    x = df.drop(["Exited"], axis=1)
    y = df["Exited"]
    x = x.drop(columns=["RowNumber", "CustomerId"])

    # Split train / test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )

    # Standardisation
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(
        x_test
    )  # transform (pas fit_transform) sur le test

    return x_train_scaled, x_test_scaled, y_train, y_test


def train_model(
    x_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
):
    """
    Entraîne un modèle GradientBoostingClassifier.

    Parameters
    ----------
    x_train : array-like
        Features d'entraînement.
    y_train : array-like
        Labels d'entraînement.
    n_estimators : int
        Nombre d'arbres (étapes de boosting).
    learning_rate : float
        Taux d'apprentissage (contribution de chaque arbre).
    max_depth : int
        Profondeur maximale de chaque arbre.
    random_state : int
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    model : GradientBoostingClassifier
        Le modèle entraîné.
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """
    Évalue les performances du modèle sur le jeu de test.

    Parameters
    ----------
    model : estimator
        Modèle entraîné (doit implémenter .predict()).
    x_test : array-like
        Features de test.
    y_test : array-like
        Labels réels de test.

    Returns
    -------
    metrics : dict
        Dictionnaire contenant l'accuracy, le rapport de classification
        et la matrice de confusion.
    """
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy score: {accuracy * 100:.2f}%")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", matrix)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def save_model(model, filepath="classifier.joblib"):
    """
    Sauvegarde le modèle entraîné avec joblib.

    Parameters
    ----------
    model : estimator
        Modèle entraîné à sauvegarder.
    filepath : str
        Chemin de destination du fichier .joblib.
    """
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé dans : {filepath}")


def load_model(filepath="classifier.joblib"):
    """
    Charge un modèle préalablement sauvegardé.

    Parameters
    ----------
    filepath : str
        Chemin du fichier .joblib à charger.

    Returns
    -------
    model : estimator
        Le modèle chargé, prêt à être utilisé pour des prédictions.
    """
    model = joblib.load(filepath)
    print(f"Modèle chargé depuis : {filepath}")
    return model
