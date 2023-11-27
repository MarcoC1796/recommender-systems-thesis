import numpy as np
import os
import tensorflow as tf
from scipy import sparse


class AdjacencyMatrixHandler:
    def __init__(self, save_dir: str, n_fold: int):
        self.save_dir = save_dir
        self.n_fold = n_fold

    def create_and_save_submatrices(self, adj_matrix: sparse.spmatrix):
        # Process the adjacency matrix into submatrices and save
        pass  # Include the logic for saving the submatrices

    def load_submatrix(self, fold_idx: int) -> tf.SparseTensor:
        with np.load(os.path.join(self.save_dir, f"adj_fold_{fold_idx}.npz")) as loader:
            return tf.SparseTensor(
                indices=np.array(list(zip(loader["row"], loader["col"]))),
                values=loader["data"],
                dense_shape=loader["shape"],
            )
