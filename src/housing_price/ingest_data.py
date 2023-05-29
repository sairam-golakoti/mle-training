import argparse
import logging
import os
import sys
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit


class HousingPriceData:
    """HousingPriceData class performs required data related actions
    Downloading raw data, Feature Transformation, Train-Test splitting, saving the processed dataset

    Parameters
    ----------
    raw_path : str
        Path to store raw dataset
    processed_path : str
        Path to store processed train and test datasets

    """

    DOWNLOAD_ROOT = r"https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_URL = DOWNLOAD_ROOT + r"datasets/housing/housing.csv"

    def __init__(self, raw_path: str, processed_path: str) -> None:
        self.raw_path = raw_path
        self.processed_path = processed_path

    def fetch_housing_data(self, housing_url: str = HOUSING_URL) -> None:
        """Download the raw tgz file from external URL, extract and store the dataset as csv

        Parameters
        ----------
        housing_url : str, optional
            url pointing to the dataset, by default HOUSING_URL
        """
        os.makedirs(self.raw_path, exist_ok=True)
        tgz_path = os.path.join(self.raw_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=self.raw_path)
        housing_tgz.close()

    def load_housing_data(self) -> pd.DataFrame:
        """Load the csv dataset into pandas dataframe

        Returns
        -------
        pd.DataFrame
            Input raw dataframe
        """
        csv_path = os.path.join(self.raw_path, "housing.csv")
        self.housing = pd.read_csv(csv_path)

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """This function splits the data into train and test datasets

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            tuple of train dataset and test dataset (in the same order)
        """
        self.housing["income_cat"] = pd.cut(
            self.housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(self.housing, self.housing["income_cat"]):
            strat_train_set = self.housing.loc[train_index]
            strat_test_set = self.housing.loc[test_index]

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        return strat_train_set, strat_test_set

    def feature_trasform(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """This function helps to derive and transform required features

        Parameters
        ----------
        train_data : pd.DataFrame
            Train dataset
        test_data : pd.DataFrame
            Test dataset

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple of transformed train and test datasets (in the same order)
        """
        housing_tr = train_data.drop("median_house_value", axis=1)
        housing_te = test_data.drop("median_house_value", axis=1)

        tr_labels = train_data["median_house_value"].copy()
        te_labels = test_data["median_house_value"].copy()

        housing_tr_num = housing_tr.drop("ocean_proximity", axis=1)
        housing_te_num = housing_te.drop("ocean_proximity", axis=1)

        imputer = SimpleImputer(strategy="median")
        imputer.fit(housing_tr_num)
        X_train = imputer.transform(housing_tr_num)
        X_test = imputer.transform(housing_te_num)

        housing_train = pd.DataFrame(
            X_train, columns=housing_tr_num.columns, index=housing_tr.index
        )
        housing_test = pd.DataFrame(X_test, columns=housing_te_num.columns, index=housing_te.index)

        housing_train = self.feature_eng(
            housing_train, "rooms_per_household", "total_rooms", "households"
        )
        housing_train = self.feature_eng(
            housing_train, "bedrooms_per_room", "total_bedrooms", "total_rooms"
        )
        housing_train = self.feature_eng(
            housing_train, "population_per_household", "population", "households"
        )

        housing_test = self.feature_eng(
            housing_test, "rooms_per_household", "total_rooms", "households"
        )
        housing_test = self.feature_eng(
            housing_test, "bedrooms_per_room", "total_bedrooms", "total_rooms"
        )
        housing_test = self.feature_eng(
            housing_test, "population_per_household", "population", "households"
        )

        housing_tr_cat = housing_tr[["ocean_proximity"]]
        housing_train = housing_train.join(pd.get_dummies(housing_tr_cat, drop_first=True))
        housing_te_cat = housing_te[["ocean_proximity"]]
        housing_test = housing_test.join(pd.get_dummies(housing_te_cat, drop_first=True))

        housing_train["median_house_value"] = tr_labels
        housing_test["median_house_value"] = te_labels

        return housing_train, housing_test

    def feature_eng(
        self, data: pd.DataFrame, out_col: str, in_col_1: str, in_col_2: str
    ) -> pd.DataFrame:
        """Derives a new column by dividing provided two columns

        Parameters
        ----------
        data : pd.DataFrame
            Pandas dataframe
        out_col : str
            Name of the output column
        in_col_1 : str
            Name of numerator column
        in_col_2 : str
            Name of denominator column

        Returns
        -------
        pd.DataFrame
            Dataframe with new column added
        """
        data[out_col] = data[in_col_1] / data[in_col_2]
        return data

    def save_data(self, data: pd.DataFrame, type: str) -> None:
        """Save the provided Pandas dataframe to csv

        Parameters
        ----------
        data : pd.DataFrame
            Dataset to be saved to csv file
        type : str
            Type of the dataset ["train", "test"]
        """
        os.makedirs(os.path.join(self.processed_path, type), exist_ok=True)
        data.to_csv(
            os.path.join(self.processed_path, type, f"{type}.csv"),
            index=False,
            index_label=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downlaod, process and store provided data")
    parser.add_argument(
        "-d",
        "--download_data",
        help="If used, raw data will be downloaded",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--raw_path",
        help="Provide path to store raw data",
        default="../../data/raw",
    )
    parser.add_argument(
        "-p",
        "--processed_path",
        help="Provide path to store processed data",
        default="../../data/processed",
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
    download = args.download_data
    raw_path = args.raw_path
    processed_path = args.processed_path
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
        handler_file = logging.FileHandler(f"{log_path}/ingest_data.log")
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    housing_price = HousingPriceData(raw_path, processed_path)

    if download:
        logger.info(f"Fetching the following dataset {housing_price.HOUSING_URL}...")
        housing_price.fetch_housing_data()

    logger.info("Reading the dataset...")
    housing_price.load_housing_data()
    logger.info("Splitting Housing Price data into Train and Test datasets...")
    strat_train_set, strat_test_set = housing_price.split_data()

    logger.info("Performing Feature Transformation...")
    train_data, test_data = housing_price.feature_trasform(strat_train_set, strat_test_set)

    logger.info("Saving Train and Test datasets...")
    housing_price.save_data(train_data, "train")
    housing_price.save_data(test_data, "test")

    logger.info("Execution completed without any errors.")
