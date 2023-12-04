import tensorflow as tf
from typing import Tuple


class DataHandler:
    def __init__(
        self, dataset_path: str, batch_size: int = 8192, buffer_size: int = 100_000
    ) -> None:
        """
        Initialize the DataHandler.

        :param dataset_path: Path to the dataset file.
        :param batch_size: Size of batches for processing the dataset.
        """
        self.dataset_path: str = dataset_path
        self.batch_size: int = batch_size
        self.buffer_size: int = buffer_size

    def load_data(self) -> tf.data.Dataset:
        """
        Loads data from the dataset path and formats it for training.

        :return: A TensorFlow dataset object.
        """
        dataset = tf.data.experimental.make_csv_dataset(
            self.dataset_path,
            batch_size=1,
            select_columns=["rating", "user_index", "item_index"],
            label_name="rating",
            num_epochs=1,
        ).unbatch()
        dataset = dataset.map(self._format_features_and_label)
        self.dataset_size = dataset.reduce(0, lambda x, _: x + 1).numpy()
        dataset = dataset.shuffle(buffer_size=self.buffer_size).batch(
            batch_size=self.batch_size
        )

        self.steps_per_epoch = self.dataset_size // self.batch_size
        if self.dataset_size % self.batch_size != 0:
            self.steps_per_epoch += 1

        return dataset

    def _format_features_and_label(
        self, features: dict, label: tf.Tensor
    ) -> Tuple[dict, tf.Tensor]:
        """
        Formats the features and label for the dataset.

        :param features: Dictionary containing feature columns.
        :param label: The label for each data entry.
        :return: Tuple of formatted features and label.
        """
        user_indices = features["user_index"]
        item_indices = features["item_index"]
        return {"user_index": user_indices, "item_index": item_indices}, label

    def split_data(self, dataset):
        total_size = self.dataset_size
        self.test_size = int(total_size * self.test_split)
        self.train_size = total_size - self.test_size

        train_dataset = dataset.take(self.train_size)
        test_dataset = dataset.skip(self.train_size).take(self.test_size)

        return train_dataset, test_dataset

    def get_train_test_datasets(self):
        dataset = self.load_data()
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.seed)

        train_dataset, test_dataset = self.split_data(dataset)

        self.train_steps_per_epoch = self.train_size // self.batch_size
        if self.train_size % self.batch_size != 0:
            self.train_steps_per_epoch += 1

        self.test_steps_per_epoch = self.test_size // self.batch_size
        if self.test_size % self.batch_size != 0:
            self.test_steps_per_epoch += 1

        train_dataset = (
            train_dataset.shuffle(self.shuffle_buffer_size)
            .batch(self.batch_size)
            .repeat()
        )
        test_dataset = test_dataset.batch(self.batch_size).cache()

        return train_dataset, test_dataset
