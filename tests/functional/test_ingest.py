import unittest
from os.path import exists


class MyTest(unittest.TestCase):
    def test_ingest_data(self):
        train_exists = exists("../../data/processed/train/train.csv")
        self.assertTrue(train_exists, "Train data does not exist!")
        test_exists = exists("../../data/processed/test/test.csv")
        self.assertTrue(test_exists, "Test data does not exist!")


if __name__ == "__main__":
    unittest.main()
