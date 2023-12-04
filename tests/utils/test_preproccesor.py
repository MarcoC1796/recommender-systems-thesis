import unittest
import os
import pandas as pd
import numpy as np
from utils.preprocessor import DataPreprocessor
from scipy.sparse import coo_matrix


class TestDataPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_data_path = os.path.join(
            "datasets", "raw", "movielens", "movielens_100k.csv"
        )
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
        preprocessor = DataPreprocessor(self.raw_data_path)
        preprocessor.load_data()
        preprocessor.remove_duplicates()
        preprocessor.create_indices()

        self.assertIn(
            "user_index", preprocessor.data.columns, "user_index column not created."
        )
        self.assertIn(
            "item_index", preprocessor.data.columns, "item_index column not created."
        )

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
        preprocessor = DataPreprocessor(self.raw_data_path)
        preprocessor.preprocess()
        preprocessor.save_preprocessed_data(
            directory=os.path.join("datasets", "preprocessed", "movielens"),
            filename="test_preprocessed_data",
            save_metadata=False,
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

        adjacency_matrix = preprocessor.create_adjacency_matrix(preprocessor.data)
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

        adjacency_matrix = preprocessor.create_adjacency_matrix(preprocessor.data)

        normalized_matrix = preprocessor.normalize_adjacency_matrix(adjacency_matrix)

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
        adjacency_matrix = preprocessor.create_adjacency_matrix(preprocessor.data)

        save_dir = "test_adjacency_matrix"
        chunk_size = 2

        preprocessor.save_adjacency_matrix(adjacency_matrix, save_dir, chunk_size)

        self.assertTrue(os.path.exists(save_dir), "The save directory does not exist.")

        num_folds = (adjacency_matrix.nnz + chunk_size - 1) // chunk_size
        for fold_id in range(num_folds):
            file_path = os.path.join(save_dir, f"adj_fold_{fold_id}.npz")
            self.assertTrue(
                os.path.exists(file_path), f"Chunk file {file_path} does not exist."
            )

            with np.load(file_path, mmap_mode=None) as fold:
                self.assertIn("row", fold)
                self.assertIn("col", fold)
                self.assertIn("data", fold)
                self.assertIn("shape", fold)

                fold_shape = tuple(fold["shape"])
                self.assertEqual(fold_shape, adjacency_matrix.shape)

                expected_data = adjacency_matrix.data[
                    fold_id * chunk_size : (fold_id + 1) * chunk_size
                ]
                np.testing.assert_array_equal(
                    fold["data"], expected_data, "Data in saved chunk is incorrect."
                )

        for fold_id in range(num_folds):
            file_path = os.path.join(save_dir, f"adj_fold_{fold_id}.npz")
            os.remove(file_path)
        os.rmdir(save_dir)

    def test_split_data(self):
        preprocessor = DataPreprocessor(self.raw_data_path)
        preprocessor.preprocess()

        train_df, test_df = preprocessor.split_data(test_size=0.2)

        total_interactions = len(preprocessor.data)
        self.assertEqual(
            len(train_df) + len(test_df),
            total_interactions,
            "The total number of interactions in train and test sets does not match the original dataset.",
        )

        merged_df = pd.merge(train_df, test_df, on=["user_id", "item_id"], how="inner")
        self.assertTrue(
            merged_df.empty,
            "There are overlapping interactions between train and test sets.",
        )

    def test_repeatable_results_with_fixed_seed(self):
        preprocessor = DataPreprocessor(self.raw_data_path)
        preprocessor.preprocess()

        train_df1, test_df1 = preprocessor.split_data(test_size=0.2, random_state=42)
        train_df2, test_df2 = preprocessor.split_data(test_size=0.2, random_state=42)

        self.assertTrue(
            train_df1.equals(train_df2) and test_df1.equals(test_df2),
            "Splits with the same random seed are not identical.",
        )

    def test_calculate_metadata(self):
        self.dummy_df.to_csv(self.dummy_csv_path, index=False)
        preprocessor = DataPreprocessor(self.dummy_csv_path)
        preprocessor.preprocess()

        metadata = preprocessor.calculate_metadata()

        expected_num_users = self.dummy_df["user_id"].nunique()
        expected_num_items = self.dummy_df["item_id"].nunique()
        expected_num_interactions = len(self.dummy_df)
        expected_dataset_density = expected_num_interactions / (
            expected_num_users * expected_num_items
        )

        self.assertEqual(
            metadata["num_users"], expected_num_users, "Number of users is incorrect."
        )
        self.assertEqual(
            metadata["num_items"], expected_num_items, "Number of items is incorrect."
        )
        self.assertEqual(
            metadata["num_interactions"],
            expected_num_interactions,
            "Number of interactions is incorrect.",
        )
        self.assertAlmostEqual(
            metadata["dataset_density"],
            expected_dataset_density,
            msg="Dataset density is incorrect.",
        )


# This allows the test script to be run from the command line
if __name__ == "__main__":
    unittest.main()
