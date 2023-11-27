import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, diags
import os
from typing import Optional


class DataPreprocessor:
    def __init__(self, file_path: str):
        self.file_path: str = file_path
        self.data: Optional[pd.DataFrame] = None
        self.num_users: Optional[int] = None
        self.num_items: Optional[int] = None

    def load_data(self) -> None:
        """Loads data from a CSV file into a pandas DataFrame."""
        self.data = pd.read_csv(self.file_path)

    def calculate_unique_counts(self) -> None:
        """Calculates the number of unique users and items."""
        if self.data is not None:
            self.num_users = self.data["user_id"].nunique()
            self.num_items = self.data["item_id"].nunique()
        else:
            raise ValueError(
                "Data not loaded. Please load the data before calculating unique counts."
            )

    def remove_duplicates(self) -> None:
        """Removes duplicate user-item interactions, keeping only the latest based on timestamp."""
        if self.data is not None:
            self.data.sort_values(
                by=["user_id", "item_id", "timestamp"],
                ascending=[True, True, False],
                inplace=True,
            )
            self.data.drop_duplicates(
                subset=["user_id", "item_id"], keep="first", inplace=True
            )

    def create_indices(self) -> None:
        """Creates continuous indices for user and item."""
        if self.data is not None:
            # Create user and item indices starting from 1
            self.data["user_index"], _ = pd.factorize(self.data["user_id"], sort=True)
            self.data["item_index"], _ = pd.factorize(self.data["item_id"], sort=True)

    def preprocess(self) -> None:
        """Preprocess the loaded data."""
        self.load_data()
        self.calculate_unique_counts()
        self.remove_duplicates()
        self.create_indices()

    def save_preprocessed_data(
        self,
        directory: str,
        filename: str,
    ) -> None:
        """Saves the preprocessed data to a specified directory."""
        if self.data is not None:
            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            save_path = os.path.join(directory, filename)
            self.data.to_csv(save_path, index=False)
            print(f"Preprocessed data saved to {save_path}")
        else:
            print("No data to save. Please run preprocess first.")

    def create_adjacency_matrix(self) -> coo_matrix:
        """Creates a symmetric adjacency matrix from the preprocessed data."""
        if self.data is None:
            raise ValueError(
                "Data not loaded. Call preprocess() before creating adjacency matrix."
            )

        user_indices = self.data["user_index"].values
        item_indices = self.data["item_index"].values
        num_users = user_indices.max() + 1
        num_items = item_indices.max() + 1

        # Interaction matrix R (user-item interactions)
        interactions = np.ones(len(user_indices))

        # Create the interaction part of the matrix (users-items interactions)
        interaction_part = coo_matrix(
            (interactions, (user_indices, item_indices + num_users)),
            shape=(num_users + num_items, num_users + num_items),
        )

        # The adjacency matrix is the sum of the interaction part and its transpose
        adjacency_matrix = interaction_part + interaction_part.T

        return adjacency_matrix.tocoo()

    def normalize_adjacency_matrix(self, adjacency_matrix: coo_matrix) -> coo_matrix:
        """Normalizes the adjacency matrix."""
        # Sum the interactions for each user/item
        d = np.array(adjacency_matrix.sum(axis=1)).flatten()

        # Inverse square root of the sum
        d_inv_sqrt = 1.0 / np.sqrt(d)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        # Create the diagonal matrix for D^-1/2 using the inverse square root of the sum
        d_mat_inv_sqrt = diags(d_inv_sqrt)

        # Compute the normalized adjacency matrix
        normalized_adjacency_matrix = d_mat_inv_sqrt.dot(adjacency_matrix).dot(
            d_mat_inv_sqrt
        )

        return normalized_adjacency_matrix.tocoo()

    def save_adjacency_matrix(
        self, adjacency_matrix: coo_matrix, save_dir: str, chunk_size: int
    ) -> None:
        """Saves the adjacency matrix in chunks."""
        try:
            self.adjacency_matrix_chunk_size = chunk_size
            os.makedirs(save_dir, exist_ok=True)

            row, col, data = (
                adjacency_matrix.row,
                adjacency_matrix.col,
                adjacency_matrix.data,
            )
            num_chunks = (len(data) + chunk_size - 1) // chunk_size
            self.adjacency_matrix_num_chunks = num_chunks

            for chunk_id in range(num_chunks):
                start = chunk_id * chunk_size
                end = min(start + chunk_size, len(data))
                file_path = os.path.join(save_dir, f"adj_chunk_{chunk_id}.npz")

                with open(file_path, "wb") as f:
                    np.savez_compressed(
                        f,
                        row=row[start:end],
                        col=col[start:end],
                        data=data[start:end],
                        shape=adjacency_matrix.shape,
                    )

            print(f"Saved chunk {chunk_id} of the normalized adjacency matrix.")
        except OSError as e:
            print(f"An error occurred while saving the adjacency matrix: {e.strerror}")


if __name__ == "__main__":
    # Obtain the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change the current working directory to the script's directory for consistent file handling
    os.chdir(script_dir)

    # Confirm the change of the current working directory
    print(
        f"Changed the current working directory to the script's location: {os.getcwd()}"
    )

    # Define a path to the 'raw' datasets directory
    raw_data_dir = os.path.join(script_dir, "..", "datasets", "raw")
    raw_data_dir = os.path.normpath(raw_data_dir)

    # Output the normalized path
    print(f"The path to the 'raw' datasets directory is: {raw_data_dir}")

    # List the contents of the 'raw' datasets directory
    print(f"Listing contents of '{raw_data_dir}':")
    print(os.listdir(raw_data_dir))

    # The file path for the raw data
    file_path = os.path.join(raw_data_dir, "movielens", "movielens_100k.csv")

    # Create an instance of the DataPreprocessor
    preprocessor = DataPreprocessor(file_path)

    # Run preprocessing
    preprocessor.preprocess()

    # Optionally, retrieve the processed data if needed
    processed_data = preprocessor.get_data()

    # Define the path to save the preprocessed data
    save_dir = os.path.join(script_dir, "..", "datasets", "preprocessed", "movielens")
    save_dir = os.path.normpath(save_dir)

    # Save the processed data
    preprocessor.save_preprocessed_data(
        directory=save_dir, filename="p_movielens_100k.csv"
    )
