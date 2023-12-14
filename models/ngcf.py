"""
Neural Graph Collaborative Filtering (NGCF) Model Implementation

This file contains a modern implementation of the NGCF model using TensorFlow and Keras.
NGCF, originally introduced in the paper "Neural Graph Collaborative Filtering" by Xiang Wang et al.,
is a recommendation system approach based on graph neural networks. It leverages user-item interaction 
data for predictive tasks, integrating user-item graph structure into the embedding learning process.

Original Paper: "Neural Graph Collaborative Filtering" by Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019)
Link to paper: https://arxiv.org/abs/1905.08108

The implementation includes two main classes: 
1. NGCFLayer - A custom TensorFlow layer representing a single layer in the NGCF architecture.
2. NGCFModel - The complete NGCF model, incorporating multiple NGCFLayer instances for 
   learning user-item interactions.

Key Features:
- Modern implementation using TensorFlow 2.x and Keras.
- Utilizes sparse matrix operations for efficient handling of large, sparse user-item graphs.
- Supports inclusion of self-interactions in the NGCFLayer to capture individual preferences.
- Implements the propagation of embeddings through the graph structure for enhanced recommendations.

Usage:
This script is intended to be imported as a module in a larger recommendation system pipeline where
user-item interaction data is available.

Dependencies: 
- numpy
- tensorflow
- keras
- os

Example: 

"""

import numpy as np
import os
import tensorflow as tf
from keras import layers, Model
from typing import Optional, Tuple


class NGCFLayer(layers.Layer):
    """
    A custom layer for the NGCF model representing a single layer of the network.

    Attributes:
        embedding_size (int): The size of the embeddings used in the layer.
        W1 (tf.Variable): Weight matrix for self-interaction in the layer.
        W2 (tf.Variable): Weight matrix for the interaction between a node and its neighbors.
    """

    def __init__(
        self, embedding_size: int, activation: Optional[callable] = None, **kwargs
    ):
        """
        Initialize the NGCFLayer.

        Args:
            embedding_size (int): The size of the embeddings.
            activation (callable, optional): Activation function to use. Defaults to None.
        """
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
        """
        Forward pass for the NGCFLayer.

        Args:
            embeddings (tf.Tensor): The input embeddings.
            adj_submatrix (tf.SparseTensor, optional): The adjacency submatrix for the graph.
            include_self_messages (bool, optional): Flag to include self-interaction messages. Defaults to False.

        Returns:
            tf.Tensor: The output embeddings after the layer's transformations.
        """

        include_self_messages = tf.constant(include_self_messages)

        def true_self_messages() -> tf.Tensor:
            # E@W1 term
            self_interaction = tf.matmul(embeddings, self.W1)

            # L@E term
            neighbor_interaction = tf.sparse.sparse_dense_matmul(
                adj_submatrix, embeddings
            )

            # L@E@W1 term
            neighbor_messages = tf.matmul(neighbor_interaction, self.W1)

            # L@E element-wise multiplied by E, then applying W2
            elementwise_product = tf.matmul(neighbor_interaction * embeddings, self.W2)

            return self_interaction + neighbor_messages + elementwise_product

        def false_self_messages() -> tf.Tensor:
            # L@E term
            neighbor_interaction = tf.sparse.sparse_dense_matmul(
                adj_submatrix, embeddings
            )

            # L@E@W1 term
            neighbor_messages = tf.matmul(neighbor_interaction, self.W1)

            # L@E element-wise multiplied by E, then apply W2
            elementwise_product = tf.matmul(neighbor_interaction * embeddings, self.W2)

            return neighbor_messages + elementwise_product

        return tf.cond(include_self_messages, true_self_messages, false_self_messages)


class NGCFModel(Model):
    """
    The NGCF model, a graph-based neural network for collaborative filtering.

    Attributes:
        num_users (int): Number of users in the dataset.
        num_items (int): Number of items in the dataset.
        embedding_size (int): Size of the embeddings for users and items.
        n_fold (int): Number of folds to split the adjacency matrix for parallel computation.
        adj_save_dir (str): Directory where adjacency matrices are saved.
        user_embeddings (keras.layers.Embedding): Embedding layer for users.
        item_embeddings (keras.layers.Embedding): Embedding layer for items.
        ngcf_layers (List[NGCFLayer]): List of NGCF layers in the model.
    """

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
        """
        Forward pass for the NGCFModel.

        Args:
            inputs (dict): Dictionary with user and item indices.

        Returns:
            tf.Tensor: The predicted user-item interaction scores.
        """
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
        """
        Compute the propagated embeddings for users and items.

        Args:
            user_indices (tf.Tensor): Indices of users.
            item_indices (tf.Tensor): Indices of items.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Tuple of tensors for user and item embeddings.
        """

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
        """
        Load a submatrix of the adjacency matrix from the saved files.

        Args:
            fold_idx (int): Index of the fold.

        Returns:
            tf.SparseTensor: The loaded adjacency submatrix.
        """
        with np.load(
            os.path.join(self.adj_save_dir, f"adj_fold_{fold_idx}.npz")
        ) as loader:
            return tf.SparseTensor(
                indices=np.array(list(zip(loader["row"], loader["col"]))),
                values=tf.cast(loader["data"], tf.float32),
                dense_shape=loader["shape"],
            )


if __name__ == "__main__":
    pass
