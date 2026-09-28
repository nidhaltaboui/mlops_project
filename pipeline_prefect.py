"""
pipeline_prefect.py
-------------------
Orchestration du pipeline ML avec Prefect (Atelier 3).
Chaque task appelle une fonction de main.py.
"""

import argparse
import subprocess
import sys

from prefect import flow, task, get_run_logger

import main as m

FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_cmd(cmd, check=True):
    """Exécute une commande, journalise la sortie via Prefect."""
    logger = get_run_logger()
    res = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(res.stdout)
    if res.returncode != 0:
        logger.warning(res.stderr)
        if check:
            raise RuntimeError(f"Échec : {' '.join(cmd)}")
    return res.returncode


@task(name="clone_repo")
def clone_task():
    m.run_clone()


@task(name="install_dependencies")
def install_task():
    m.run_install()


@task(name="format_code")
def format_task():
    run_cmd([sys.executable, "-m", "black", *FILES])


@task(name="code_quality")
def quality_task():
    run_cmd(
        [sys.executable, "-m", "flake8", "--max-line-length=100", *FILES],
        check=False,
    )


@task(name="code_security")
def security_task():
    run_cmd([sys.executable, "-m", "bandit", "-q", *FILES], check=False)


@task(name="unit_tests")
def test_task():
    run_cmd([sys.executable, "-m", "pytest", "-q", "test_pipeline.py"])


@task(name="prepare_data")
def prepare_task(data_path=m.DATA_PATH):
    return m.run_prepare(data_path)


@task(name="train_model")
def train_task(x_train, y_train):
    return m.run_train(x_train, y_train)


@task(name="save_model")
def save_task(model, model_path=m.MODEL_PATH):
    return m.run_save(model, model_path)


@task(name="load_model")
def load_task(model_path=m.MODEL_PATH):
    return m.run_load(model_path)


@task(name="evaluate_model")
def evaluate_task(model, x_test, y_test):
    return m.run_evaluate(model, x_test, y_test)


@flow(name="code")
def code_flow():
    format_task()
    quality_task()
    security_task()
    test_task()


@flow(name="all")
def all_flow():
    clone_task()
    install_task()
    code_flow()
    x_train, x_test, y_train, y_test = prepare_task()
    model = train_task(x_train, y_train)
    save_task(model)
    evaluate_task(model, x_test, y_test)


@flow(name="train")
def train_flow():
    x_train, x_test, y_train, y_test = prepare_task()
    return train_task(x_train, y_train)


@flow(name="evaluate")
def evaluate_flow():
    _, x_test, _, y_test = prepare_task()
    model = load_task()
    evaluate_task(model, x_test, y_test)


FLOWS = {
    "code": code_flow,
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancement des flows Prefect.")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
