import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, diags
import os
import json
from typing import Optional, Tuple


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
        save_metadata: bool = True,
        save_split_datasets: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Saves the preprocessed data to a specified directory.
        If save_split_datasets is True, it saves train and test datasets separately.
        """
        if self.data is not None:
            os.makedirs(directory, exist_ok=True)
            if save_split_datasets:
                train_df, test_df = self.split_data(
                    test_size=test_size, random_state=random_state
                )
                train_df.to_csv(
                    os.path.join(directory, f"{filename}_train.csv"), index=False
                )
                test_df.to_csv(
                    os.path.join(directory, f"{filename}_test.csv"), index=False
                )
                print(f"Train and test datasets saved in {directory}")
            else:
                save_path = os.path.join(directory, f"{filename}.csv")
                self.data.to_csv(save_path, index=False)
                print(f"Preprocessed data saved to {save_path}")

            if save_metadata:
                metadata = self.calculate_metadata()
                metadata_path = os.path.join(directory, f"{filename}_metadata.json")
                with open(metadata_path, "w") as file:
                    json.dump(metadata, file, indent=4)
                print(f"Metadata saved to {metadata_path}")

        else:
            print("No data to save. Please run preprocess first.")

    def create_adjacency_matrix(self, subset_df: pd.DataFrame) -> coo_matrix:
        """
        Creates a symmetric adjacency matrix from the specified subset of data.
        The matrix size is based on the total number of users and items in the whole dataset.
        """
        if self.data is None or self.num_users is None or self.num_items is None:
            raise ValueError(
                "Data not loaded or unique counts not calculated. Call preprocess() before creating adjacency matrix."
            )

        if subset_df is None:
            raise ValueError("Subset DataFrame is not provided.")

        user_indices = subset_df["user_index"].values
        item_indices = subset_df["item_index"].values

        interactions = np.ones(len(user_indices))

        interaction_part = coo_matrix(
            (interactions, (user_indices, item_indices + self.num_users)),
            shape=(self.num_users + self.num_items, self.num_users + self.num_items),
        )

        adjacency_matrix = interaction_part + interaction_part.T

        return adjacency_matrix.tocoo()

    def normalize_adjacency_matrix(self, adjacency_matrix: coo_matrix) -> coo_matrix:
        """Normalizes the adjacency matrix."""
        d = np.array(adjacency_matrix.sum(axis=1)).flatten()

        d_inv_sqrt = 1.0 / np.sqrt(d)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        d_mat_inv_sqrt = diags(d_inv_sqrt)

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

            for fold_id in range(num_chunks):
                start = fold_id * chunk_size
                end = min(start + chunk_size, len(data))
                file_path = os.path.join(save_dir, f"adj_fold_{fold_id}.npz")

                with open(file_path, "wb") as f:
                    np.savez_compressed(
                        f,
                        row=row[start:end],
                        col=col[start:end],
                        data=data[start:end],
                        shape=adjacency_matrix.shape,
                    )

            print(f"Saved chunk {fold_id} of the normalized adjacency matrix.")
        except OSError as e:
            print(f"An error occurred while saving the adjacency matrix: {e.strerror}")

    def split_data(
        self, test_size: float = 0.2, random_state=42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing sets.
        """
        if self.data is None:
            raise ValueError(
                "Data not loaded. Call preprocess() before splitting data."
            )

        shuffled_data = self.data.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )

        num_test = int(len(shuffled_data) * test_size)

        test_df = shuffled_data[:num_test]
        train_df = shuffled_data[num_test:]

        return train_df, test_df

    def calculate_metadata(self) -> dict:
        """
        Calculate and return metadata.
        """
        num_users = (
            self.num_users
            if self.num_users is not None
            else self.data["user_id"].nunique()
        )
        num_items = (
            self.num_items
            if self.num_items is not None
            else self.data["item_id"].nunique()
        )
        num_interactions = len(self.data)
        dataset_density = num_interactions / (num_users * num_items)

        return {
            "num_users": num_users,
            "num_items": num_items,
            "num_interactions": num_interactions,
            "dataset_density": dataset_density,
        }


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
