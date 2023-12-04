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
        include_self_messages: bool = False,
    ) -> tf.Tensor:
        self_messages = tf.constant(include_self_messages)

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

        return tf.cond(self_messages, true_self_messages, false_self_messages)
