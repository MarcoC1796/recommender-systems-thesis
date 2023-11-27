import unittest
import os
import pandas as pd
import numpy as np
from utils.preprocessor import DataPreprocessor
from scipy.sparse import coo_matrix


class TestDataPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method will be run once before all tests
        cls.raw_data_path = os.path.join(
            "datasets", "raw", "movielens", "movielens_100k.csv"
        )  # Path to a test CSV file
        cls.preprocessed_data_path = os.path.join(
            "datasets", "preprocessed", "movielens", "test_preprocessed_data.csv"
        )

        cls.dummy_data = {
            "user_id": [1, 2, 1, 2, 3, 3],
            "item_id": [10, 10, 20, 20, 30, 10],
            "timestamp": [100000, 100001, 100002, 100003, 100004, 100005],
        }
        cls.dummy_df = pd.DataFrame(cls.dummy_data)
        cls.dummy_csv_path = "dummy_data.csv"

    @classmethod
    def tearDownClass(cls):
        # This method will be run once after all tests
        # Clean up the test CSV files
        if os.path.exists(cls.preprocessed_data_path):
            os.remove(cls.preprocessed_data_path)

        if os.path.exists(cls.dummy_csv_path):
            os.remove(cls.dummy_csv_path)

    def test_load_data(self):
        # Test that data is loaded correctly
        preprocessor = DataPreprocessor(self.raw_data_path)
        preprocessor.load_data()
        self.assertIsNotNone(preprocessor.data)
        self.assertIsInstance(preprocessor.data, pd.DataFrame)

    def test_remove_duplicates(self):
        # Test that duplicates are removed correctly
        preprocessor = DataPreprocessor(self.raw_data_path)
        preprocessor.load_data()

        duplicates = preprocessor.data.groupby(["user_id", "item_id"]).size()
        duplicates_true = duplicates[duplicates > 1]

        self.assertFalse(
            duplicates_true.empty,
            "There are no duplicates present before removing duplicates.",
        )

        preprocessor.remove_duplicates()

        duplicates = preprocessor.data.groupby(["user_id", "item_id"]).size()
        duplicates_false = duplicates[duplicates > 1]

        self.assertTrue(
            duplicates_false.empty,
            "There are duplicates present after removing duplicates.",
        )

    def test_create_indices(self):
        # Test that user and item indices are created correctly
        preprocessor = DataPreprocessor(self.raw_data_path)
        preprocessor.load_data()
        preprocessor.remove_duplicates()
        preprocessor.create_indices()

        # Check if the 'user_index' and 'item_index' columns have been created
        self.assertIn(
            "user_index", preprocessor.data.columns, "user_index column not created."
        )
        self.assertIn(
            "item_index", preprocessor.data.columns, "item_index column not created."
        )

        # Check if indices start from 1
        self.assertEqual(
            preprocessor.data["user_index"].min(),
            0,
            "user_index does not start from 0.",
        )
        self.assertEqual(
            preprocessor.data["item_index"].min(),
            0,
            "item_index does not start from 0.",
        )

        # Check if indices are continuous and match the count of unique users and items
        unique_users = preprocessor.data["user_id"].nunique()
        unique_items = preprocessor.data["item_id"].nunique()
        self.assertEqual(
            preprocessor.data["user_index"].max() + 1,
            unique_users,
            "user_index is not continuous.",
        )
        self.assertEqual(
            preprocessor.data["item_index"].max() + 1,
            unique_items,
            "item_index is not continuous.",
        )

    def test_save_preprocessed_data(self):
        # Test that preprocessed data is saved correctly
        preprocessor = DataPreprocessor(self.raw_data_path)
        preprocessor.preprocess()
        preprocessor.save_preprocessed_data(
            directory=os.path.join("datasets", "preprocessed", "movielens"),
            filename="test_preprocessed_data.csv",
        )
        self.assertTrue(os.path.exists(self.preprocessed_data_path))

    def test_create_adjacency_matrix(self):
        expected_matrix = [
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]

        wrong_matrix = [
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]

        self.dummy_df.to_csv(self.dummy_csv_path, index=False)
        preprocessor = DataPreprocessor(self.dummy_csv_path)
        preprocessor.preprocess()

        adjacency_matrix = preprocessor.create_adjacency_matrix()
        dense_adjacency_matrix = adjacency_matrix.toarray()

        self.assertIsInstance(
            adjacency_matrix, coo_matrix, "The adjacency matrix is not in COO format."
        )
        self.assertTrue(
            (adjacency_matrix != adjacency_matrix.T).nnz == 0,
            "Adjacency matrix is not symmetric.",
        )
        self.assertTrue(
            (dense_adjacency_matrix == expected_matrix).all(),
            "Created adjacency matrix does not match the expected matrix.",
        )
        self.assertFalse(
            (dense_adjacency_matrix == wrong_matrix).all(),
            "Created adjacency matrix matches wrong matrix.",
        )

    def test_normalize_adjacency_matrix(self):
        expected_normalized_matrix = [
            [0, 0, 0, 1 / np.sqrt(6), 1 / 2, 0],
            [0, 0, 0, 1 / np.sqrt(6), 1 / 2, 0],
            [0, 0, 0, 1 / np.sqrt(6), 0, 1 / np.sqrt(2)],
            [1 / np.sqrt(6), 1 / np.sqrt(6), 1 / np.sqrt(6), 0, 0, 0],
            [1 / 2, 1 / 2, 0, 0, 0, 0],
            [0, 0, 1 / np.sqrt(2), 0, 0, 0],
        ]

        self.dummy_df.to_csv(self.dummy_csv_path, index=False)
        preprocessor = DataPreprocessor(self.dummy_csv_path)
        preprocessor.preprocess()

        adjacency_matrix = preprocessor.create_adjacency_matrix()

        # Use the actual normalize_adjacency_matrix method
        normalized_matrix = preprocessor.normalize_adjacency_matrix(adjacency_matrix)

        # Convert COO to dense format for comparison
        normalized_dense = normalized_matrix.toarray()

        self.assertIsInstance(
            normalized_matrix, coo_matrix, "The normalized matrix is not in COO format."
        )
        self.assertTrue(
            np.allclose(normalized_dense, expected_normalized_matrix),
            "The normalized adjacency matrix does not match the expected matrix.",
        )

    def test_save_adjacency_matrix(self):
        preprocessor = DataPreprocessor(self.dummy_csv_path)
        preprocessor.preprocess()
        adjacency_matrix = preprocessor.create_adjacency_matrix()

        # Define directory and filename for saving
        save_dir = "test_adjacency_matrix"
        chunk_size = 2  # Define a small chunk size for testing

        # Run save_adjacency_matrix method
        preprocessor.save_adjacency_matrix(adjacency_matrix, save_dir, chunk_size)

        # Check if the directory is created
        self.assertTrue(os.path.exists(save_dir), "The save directory does not exist.")

        # Verify files are created and have correct data
        num_chunks = (adjacency_matrix.nnz + chunk_size - 1) // chunk_size
        for chunk_id in range(num_chunks):
            file_path = os.path.join(save_dir, f"adj_chunk_{chunk_id}.npz")
            self.assertTrue(
                os.path.exists(file_path), f"Chunk file {file_path} does not exist."
            )

            # Load the chunk and verify its contents
            with np.load(file_path, mmap_mode=None) as chunk:
                self.assertIn("row", chunk)
                self.assertIn("col", chunk)
                self.assertIn("data", chunk)
                self.assertIn("shape", chunk)

                # Verify that each chunk has the correct shape attribute
                chunk_shape = tuple(chunk["shape"])
                self.assertEqual(chunk_shape, adjacency_matrix.shape)

                # Check the data is correct (this is a simple check, for large matrices,
                # you would want to check that the data corresponds to the correct slice of the matrix)
                expected_data = adjacency_matrix.data[
                    chunk_id * chunk_size : (chunk_id + 1) * chunk_size
                ]
                np.testing.assert_array_equal(
                    chunk["data"], expected_data, "Data in saved chunk is incorrect."
                )

        # Clean up: remove the created directory and its files
        for chunk_id in range(num_chunks):
            file_path = os.path.join(save_dir, f"adj_chunk_{chunk_id}.npz")
            os.remove(file_path)
        os.rmdir(save_dir)


# This allows the test script to be run from the command line
if __name__ == "__main__":
    unittest.main()
