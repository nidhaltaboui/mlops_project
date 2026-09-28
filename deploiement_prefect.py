"""
deploiement_prefect.py
----------------------
Enregistre les flows comme deployments Prefect et les sert (worker).
Le flow 'all' est planifié une fois par jour à 02h00.
"""

from prefect import serve

from pipeline_prefect import all_flow, train_flow, evaluate_flow, code_flow

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(
            name="ml-pipeline-train",
            tags=["training", "mlops"],
        ),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate",
            tags=["evaluation", "mlops"],
        ),
        code_flow.to_deployment(
            name="ml-pipeline-code",
            tags=["quality", "mlops"],
        ),
    )
