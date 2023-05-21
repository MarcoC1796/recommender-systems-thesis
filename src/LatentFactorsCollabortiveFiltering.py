import numpy as np
from tqdm.auto import trange
from .aux_functions import standardize_interactions


class LatentFactorsCollaborativeFiltering:
    def __init__(self, num_users, num_items, standardize=True):
        self.num_users = num_users
        self.num_items = num_items
        self.user_embeddings = None
        self.item_embeddings = None
        self.standardize = standardize
        self.mean_train = None
        self.std_train = None

    def fit(
        self,
        train_interactions,
        validation_interactions=None,
        num_factors=10,
        epochs=10,
        batch_size=128,
        learning_rate=0.01,
    ):
        if self.standardize:
            train_interactions, mean_train, std_train = standardize_interactions(
                train_interactions
            )
            self.mean_train = mean_train
            self.std_train = std_train

        self.user_embeddings = np.random.normal(
            loc=0, scale=1, size=(self.num_users, num_factors)
        )
        self.item_embeddings = np.random.normal(
            loc=0, scale=1, size=(self.num_items, num_factors)
        )

        train_errors = []
        validation_errors = []

        pbar_outer = trange(epochs, desc="Current Error: None | Training Progress: ")

        for epoch in pbar_outer:
            np.random.shuffle(train_interactions)
            train_error = 0.0
            validation_error = 0.0
            num_batches = len(train_interactions) // batch_size
            pbar_inner = trange(
                num_batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False
            )

            for batch_idx in pbar_inner:
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_interactions = train_interactions[start_idx:end_idx]
                users, items, ratings = np.split(batch_interactions, 3, axis=1)
                users = users.flatten().astype(int)
                items = items.flatten().astype(int)
                ratings = ratings.flatten()
                errors = self.update_params(users, items, ratings, learning_rate)
                train_error += np.sum(errors**2)

            train_errors.append(np.sqrt(train_error / len(train_interactions)))
            if validation_interactions is not None:
                validation_error = self.evaluateRMSE(validation_interactions)
                validation_errors.append(validation_error)
            pbar_outer.set_description(
                f"Current Error: {train_errors[-1]:.2e} | Traning Progress"
            )

        return train_errors, validation_errors

    def predict(self, user, item):
        user_embedding = self.user_embeddings[user, :]
        item_embedding = self.item_embeddings[item, :]
        return np.dot(user_embedding, item_embedding)

    def predict_batch(self, users, items):
        user_embeddings_batch = self.user_embeddings[users, :]
        item_embeddings_batch = self.item_embeddings[items, :]
        return np.sum(user_embeddings_batch * item_embeddings_batch, axis=1)

    def recommend_items(self, user, top_k=5):
        user_embedding = self.user_embeddings[user, :]
        scores = np.dot(self.item_embeddings, user_embedding)
        top_item_indices = np.argsort(scores)[::-1][:top_k]
        return top_item_indices

    def update_params(self, users, items, ratings, learning_rate):
        errors = ratings - self.predict_batch(users, items)
        user_gradients = errors[:, np.newaxis] * self.item_embeddings[items, :]
        item_gradients = errors[:, np.newaxis] * self.user_embeddings[users, :]
        self.user_embeddings[users, :] += learning_rate * user_gradients
        self.item_embeddings[items, :] += learning_rate * item_gradients
        return errors

    def evaluateRMSE(self, test_interactions):
        test_users, test_items, test_ratings = np.split(test_interactions, 3, axis=1)
        test_users = test_users.flatten().astype(int)
        test_items = test_items.flatten().astype(int)
        test_ratings = test_ratings.flatten()
        test_predictions = self.predict_batch(test_users, test_items)
        if self.standardize:
            test_predictions = test_predictions * self.std_train + self.mean_train
        errors = test_ratings - test_predictions
        test_rmse = np.sqrt(np.mean(errors**2))
        return test_rmse
