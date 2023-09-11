import mlflow

if __name__ == "__main__":
    print("started")
    experiment_id = mlflow.get_experiment_by_name("housing_price_prediction").experiment_id
    with mlflow.start_run(experiment_id=experiment_id, description="parent") as parent_run:
        mlflow.set_tag("mlflow.runName", "PARENT_RUN")
        mlflow.log_param("parent", "yes")
        mlflow.run(
            ".",
            "ingest_data",
            experiment_id=experiment_id,
            run_name="ingest_data",
            env_manager="local",
        )

        mlflow.run(
            ".",
            "model-train",
            experiment_id=experiment_id,
            run_name="model-train",
            env_manager="local",
        )
        mlflow.run(
            ".",
            "model-score",
            experiment_id=experiment_id,
            run_name="model-score",
            env_manager="local",
        )
