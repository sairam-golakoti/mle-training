import argparse
import logging
import pickle
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class ModelTrain:
    """This class trains a RandomForest Regression model and returns the best estimated model

    Parameters
    ----------
    train_data : str
        Dataset to be used to train the model

    Returns
    -------
    RandomForestRegressor
        Trained regression model
    """

    PARAM_GRID = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    def __init__(self, train_data: str) -> None:
        self.train_data = train_data

    def read_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read the train dataset and separate X and y datasets

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple of X_train and y_train pandas dataframe (in the same order)
        """
        data = pd.read_csv(self.train_data)
        housing_prepared = data.drop("median_house_value", axis=1).values
        housing_labels = data["median_house_value"].copy().values

        return housing_prepared, housing_labels

    def model_train(
        self, housing_prepared: pd.DataFrame, housing_labels: pd.DataFrame
    ) -> RandomForestRegressor:
        """Train and select the best estimated RandomForest Regression model

        Parameters
        ----------
        housing_prepared : pd.DataFrame
            X_train dataset
        housing_labels : pd.DataFrame
            y_train dataset / labels

        Returns
        -------
        RandomForestRegressor
            Best Estimated RandomForest Regression model
        """
        forest_reg = RandomForestRegressor(random_state=42)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(
            forest_reg,
            self.PARAM_GRID,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        grid_search.fit(housing_prepared, housing_labels)

        grid_search.best_params_

        final_model = grid_search.best_estimator_
        return final_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-td",
        "--train_data",
        help="Provide path to read the input data",
        default="../../data/processed/train/train.csv",
    )
    parser.add_argument(
        "-op",
        "--output_data",
        help="Provide path to store output files",
        default="../../artifacts/model.pkl",
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
    train_data = args.train_data
    output_data = args.output_data
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
        handler_file = logging.FileHandler(f"{log_path}/train.log")
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    trainer = ModelTrain(train_data)

    logger.info(f"Loading the train data from {train_data}...")
    X_train, y_train = trainer.read_data()

    logger.info("Training RandomForest Regressor model...")
    model = trainer.model_train(X_train, y_train)

    logger.info(f"Saving the trained model at {output_data}...")
    with open(output_data, "wb") as file:
        pickle.dump(model, file)

    logger.info("Execution completed without any errors.")
