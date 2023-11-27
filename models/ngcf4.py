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
        self.activation: Optional[callable] = activation
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
        adj_submatrix: Optional[tf.SparseTensor] = None,
        self_messages: bool = False,
    ) -> tf.Tensor:
        if self_messages:
            embeddings_updated: tf.Tensor = tf.matmul(embeddings, self.W1)
        elif adj_submatrix is not None:
            relevant_embeddings: tf.Tensor = tf.sparse.sparse_dense_matmul(
                adj_submatrix, embeddings
            )
            non_interactive_messages: tf.Tensor = tf.matmul(
                relevant_embeddings, self.W1
            )
            interactive_messages: tf.Tensor = tf.matmul(
                relevant_embeddings * embeddings, self.W2
            )
            embeddings_updated: tf.Tensor = (
                non_interactive_messages + interactive_messages
            )
        else:
            raise ValueError(
                "An adjacency submatrix must be provided if not using only self_messages."
            )

        if self.activation:
            embeddings_updated = self.activation(embeddings_updated)

        return embeddings_updated


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

    def call(self, user_indices: tf.Tensor, item_indices: tf.Tensor) -> tf.Tensor:
        final_u_embeddings, final_i_embeddings = self.compute_final_embeddings(
            user_indices, item_indices
        )

        user_item_scores: tf.Tensor = tf.reduce_sum(
            final_u_embeddings * final_i_embeddings, axis=1
        )

        return user_item_scores

    def compute_final_embeddings(
        self, user_indices: tf.Tensor, item_indices: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        original_u_embeddings: tf.Tensor = self.user_embeddings(user_indices)
        original_i_embeddings: tf.Tensor = self.item_embeddings(item_indices)

        all_u_embeddings: list = [original_u_embeddings]
        all_i_embeddings: list = [original_i_embeddings]

        all_embeddings: tf.Tensor = tf.concat(
            [original_u_embeddings, original_i_embeddings], axis=0
        )

        for layer in self.ngcf_layers:
            layer_embeddings: tf.Tensor = tf.zeros_like(all_embeddings)

            for i_fold in range(self.n_fold):
                adj_submatrix: tf.SparseTensor = self.load_adjacency_submatrix(i_fold)
                layer_embeddings += layer(all_embeddings, adj_submatrix)

            all_embeddings = tf.nn.leaky_relu(layer_embeddings)

            u_embeddings, i_embeddings = tf.split(
                all_embeddings, [self.num_users, self.num_items], axis=0
            )
            all_u_embeddings.append(u_embeddings)
            all_i_embeddings.append(i_embeddings)

        final_u_embeddings: tf.Tensor = tf.concat(all_u_embeddings, axis=1)
        final_i_embeddings: tf.Tensor = tf.concat(all_i_embeddings, axis=1)

        return final_u_embeddings, final_i_embeddings

    def load_adjacency_submatrix(self, fold_idx: int) -> tf.SparseTensor:
        with np.load(
            os.path.join(self.adj_save_dir, f"adj_fold_{fold_idx}.npz")
        ) as loader:
            return tf.SparseTensor(
                indices=np.array(list(zip(loader["row"], loader["col"]))),
                values=loader["data"],
                dense_shape=loader["shape"],
            )
