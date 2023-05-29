import argparse
import logging
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    """
    This script scores the trained model based on the user arguments given.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mp",
        "--model_path",
        help="Provide path to store output files",
        default="../../artifacts/model.pkl",
    )
    parser.add_argument(
        "-tp",
        "--test_data_path",
        help="Provide path for test dataset",
        default="../../data/processed/test/test.csv",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        choices=["DEBUG", "INFO", "ERROR"],
        type=lambda arg: {x.lower(): x for x in ["DEBUG", "INFO", "ERROR"]}[arg.lower()],
        help="Provide level of the log",
        default="info",
    )
    parser.add_argument(
        "-lp", "--log_path", help="Provide path to store log file", default="../../logs"
    )
    parser.add_argument(
        "-ncl",
        "--no_console_log",
        help="Logs will not be printed onto console when used",
        action="store_true",
    )

    args = parser.parse_args()
    test_data_path = args.test_data_path
    model_path = args.model_path
    log_level = args.log_level
    log_path = args.log_path
    console_log = args.no_console_log

    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    if log_level.upper() == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif log_level.upper() == "ERROR":
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    if not console_log:
        handler_cons = logging.StreamHandler(sys.stdout)
        handler_cons.setFormatter(formatter)
        logger.addHandler(handler_cons)

    if log_path:
        handler_file = logging.FileHandler(f"{log_path}/score.log")
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    logger.info(f"Loading and preparing test data from {test_data_path}...")
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop("median_house_value", axis=1).values
    y_test = test_data["median_house_value"].copy().values

    logger.info(f"Loading Trained RandomForest Model at {model_path}...")
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    logger.info("Scoring the model...")
    final_predictions = model.predict(X_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    logger.info(f"RMSE for the Test data provided on RandomForest Model is {final_rmse}.")

    logger.info("Execution completed without any errors.")
