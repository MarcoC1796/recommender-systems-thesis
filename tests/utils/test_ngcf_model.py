import tensorflow as tf
import numpy as np
import os
import tempfile
import unittest
from models.ngcf import NGCFModel


class TestNGCFModel(unittest.TestCase):
    def test_load_adjacency_submatrix(self):
        # Create a temporary directory to save mock adjacency matrix
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock adjacency matrix data
            mock_adj_data = {
                "row": np.array([0, 1, 2]),
                "col": np.array([1, 2, 0]),
                "data": np.array([1.0, 2.0, 3.0]),
                "shape": (3, 3),
            }

            # Save mock data in npz format
            np.savez(os.path.join(tmp_dir, "adj_fold_0.npz"), **mock_adj_data)

            # Instantiate NGCFModel with mock data directory
            model = NGCFModel(
                num_users=3,
                num_items=3,
                num_layers=1,
                embedding_size=10,
                n_fold=1,
                adj_save_dir=tmp_dir,
            )

            # Load the adjacency submatrix
            loaded_submatrix = model.load_adjacency_submatrix(0)

            # Convert loaded indices to a list of tuples for comparison
            loaded_indices_as_tuples = [
                tuple(idx) for idx in loaded_submatrix.indices.numpy().tolist()
            ]

            # Assertions to ensure the loaded data is correct
            self.assertIsInstance(
                loaded_submatrix,
                tf.SparseTensor,
                "Loaded data should be a SparseTensor",
            )
            self.assertEqual(
                loaded_indices_as_tuples,
                list(zip(mock_adj_data["row"], mock_adj_data["col"])),
            )
            self.assertEqual(
                loaded_submatrix.values.numpy().tolist(), mock_adj_data["data"].tolist()
            )
            self.assertEqual(
                tuple(loaded_submatrix.dense_shape.numpy()), mock_adj_data["shape"]
            )


# Run the test
if __name__ == "__main__":
    unittest.main()
