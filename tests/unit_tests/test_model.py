import unittest
from os.path import exists


class MyTest(unittest.TestCase):
    def test_ingest_data(self):
        model_exists = exists("../../artifacts/model.pkl")
        self.assertTrue(model_exists, "Model does not exist!")


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()
