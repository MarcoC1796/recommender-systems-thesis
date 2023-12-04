import numpy as np
import os
import tensorflow as tf
from keras import layers, Model
from typing import Optional, Tuple


class NGCFLayer(layers.Layer):
    def __init__(
        self, embedding_size: int, activation: Optional[callable] = None, **kwargs
    ):
        super(NGCFLayer, self).__init__(**kwargs)
        self.embedding_size: int = embedding_size
        self.W1 = self.add_weight(
            shape=(embedding_size, embedding_size),
            initializer="glorot_uniform",
            name="W1",
        )
        self.W2 = self.add_weight(
            shape=(embedding_size, embedding_size),
            initializer="glorot_uniform",
            name="W2",
        )

    def call(
        self,
        embeddings: tf.Tensor,
        adj_submatrix: tf.SparseTensor = None,
        include_self_messages: bool = False,
    ) -> tf.Tensor:
        include_self_messages = tf.constant(include_self_messages)

        def true_self_messages() -> tf.Tensor:
            self_messages = tf.matmul(embeddings, self.W1)

            relevant_embeddings = tf.sparse.sparse_dense_matmul(
                adj_submatrix, embeddings
            )

            non_interactive_messages = tf.matmul(relevant_embeddings, self.W1)
            interactive_messages = tf.matmul(relevant_embeddings * embeddings, self.W2)

            return self_messages + non_interactive_messages + interactive_messages

        def false_self_messages() -> tf.Tensor:
            relevant_embeddings = tf.sparse.sparse_dense_matmul(
                adj_submatrix, embeddings
            )

            non_interactive_messages = tf.matmul(relevant_embeddings, self.W1)
            interactive_messages = tf.matmul(relevant_embeddings * embeddings, self.W2)

            return non_interactive_messages + interactive_messages

        return tf.cond(include_self_messages, true_self_messages, false_self_messages)


class NGCFModel(Model):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_layers: int,
        embedding_size: int,
        n_fold: int,
        adj_save_dir: str,
        **kwargs,
    ):
        super(NGCFModel, self).__init__(**kwargs)
        self.num_users: int = num_users
        self.num_items: int = num_items
        self.embedding_size: int = embedding_size
        self.n_fold: int = n_fold
        self.adj_save_dir: str = adj_save_dir

        self.user_embeddings = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            embeddings_initializer="glorot_uniform",
            name="user_embeddings",
        )
        self.item_embeddings = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            embeddings_initializer="glorot_uniform",
            name="item_embeddings",
        )

        self.ngcf_layers = [NGCFLayer(embedding_size) for _ in range(num_layers)]

    def call(self, inputs) -> tf.Tensor:
        user_indices, item_indices = inputs["user_index"], inputs["item_index"]

        initial_u_embeddings = self.user_embeddings(user_indices)
        initial_i_embeddings = self.item_embeddings(item_indices)

        (
            propagated_u_embeddings,
            propagated_i_embeddings,
        ) = self.compute_propagated_embeddings(user_indices, item_indices)

        final_u_embeddings = tf.concat(
            [initial_u_embeddings, propagated_u_embeddings], axis=1
        )
        final_i_embeddings = tf.concat(
            [initial_i_embeddings, propagated_i_embeddings], axis=1
        )

        user_item_scores = tf.reduce_sum(
            final_u_embeddings * final_i_embeddings, axis=1
        )

        return user_item_scores

    def compute_propagated_embeddings(
        self, user_indices: tf.Tensor, item_indices: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        original_u_embeddings = self.user_embeddings.weights[0]
        original_i_embeddings = self.item_embeddings.weights[0]

        input_embeddings = tf.concat(
            [original_u_embeddings, original_i_embeddings], axis=0
        )

        all_propagated_embeddings = []

        for layer in self.ngcf_layers:
            output_embeddings = tf.zeros_like(input_embeddings)

            for i_fold in range(self.n_fold):
                adj_submatrix = self.load_adjacency_submatrix(i_fold)
                include_self_messages = i_fold == 0
                output_embeddings += layer(
                    input_embeddings,
                    adj_submatrix,
                    include_self_messages=include_self_messages,
                )

            output_embeddings = tf.nn.leaky_relu(output_embeddings)
            all_propagated_embeddings.append(output_embeddings)
            input_embeddings = output_embeddings

        all_propagated_embeddings = tf.concat(all_propagated_embeddings, axis=1)

        propagated_u_embeddings, propagated_i_embeddings = tf.split(
            all_propagated_embeddings, [self.num_users, self.num_items], axis=0
        )

        selected_u_embeddings = tf.nn.embedding_lookup(
            propagated_u_embeddings, user_indices
        )
        selected_i_embeddings = tf.nn.embedding_lookup(
            propagated_i_embeddings, item_indices
        )

        return (
            selected_u_embeddings,
            selected_i_embeddings,
        )

    def load_adjacency_submatrix(self, fold_idx: int) -> tf.SparseTensor:
        with np.load(
            os.path.join(self.adj_save_dir, f"adj_fold_{fold_idx}.npz")
        ) as loader:
            return tf.SparseTensor(
                indices=np.array(list(zip(loader["row"], loader["col"]))),
                values=tf.cast(loader["data"], tf.float32),
                dense_shape=loader["shape"],
            )


if __name__ == "__main__":
    import os
    from utils.datahandler import DataHandler

    dummy_train = (
        {
            "user_index": tf.constant([0, 1, 3, 1, 3, 2], dtype=tf.int64),
            "item_index": tf.constant([2, 3, 5, 0, 3, 2], dtype=tf.int64),
        },
        tf.constant([1, 4, 9, 6, 6, 7], dtype=tf.int64),
    )

    datahandler = DataHandler(
        dataset_path=os.path.join(
            "datasets", "preprocessed", "movielens", "p_movielens_100k.csv"
        ),
        test_split=0.2,
    )

    # After you call get_train_test_datasets
    train_dataset, test_dataset = datahandler.get_train_test_datasets()

    # Take a single batch from the train dataset and inspect its structure
    for features, labels in train_dataset.take(1):
        print("Features:", features)
        print("Labels:", labels)
        print(
            "Shapes:",
            {key: value.shape for key, value in features.items()},
            labels.shape,
        )

    users, items = (943, 1664)

    save_dir = os.path.join(
        "datasets", "preprocessed", "movielens", "p_movielens_100k_norm_adj_mat"
    )

    model = NGCFModel(
        num_users=users,
        num_items=items,
        num_layers=3,
        embedding_size=32,
        n_fold=1,
        adj_save_dir=save_dir,
    )

    # Compile the model
    model.compile(
        optimizer="adam",  # Adjust the optimizer and learning rate as needed
        loss="mean_squared_error",  # You can also use a custom loss function here
        metrics=[
            tf.keras.metrics.RootMeanSquaredError()
        ],  # Add any additional metrics you want to track
    )

    result = model(dummy_train[0])

    print("\nRESULT:", result, "\n")

    history = model.fit(
        train_dataset,
        epochs=25,
        steps_per_epoch=datahandler.train_steps_per_epoch,
        verbose=1,
    )
