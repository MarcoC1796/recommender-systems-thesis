import numpy as np
from tqdm.auto import trange
from .aux_functions import standardize_interactions


class LatentFactorsCollaborativeFiltering:
    def __init__(
        self,
        num_users,
        num_items,
        num_factors=10,
        include_biases=True,
        standardize=True,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.include_biases = include_biases
        self.user_biases = None
        self.item_biases = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.standardize = standardize
        self.mean_train = None
        self.std_train = None

    def fit(
        self,
        train_interactions,
        validation_interactions=None,
        num_factors=None,
        reg_strength=0,
        epochs=10,
        batch_size=128,
        learning_rate=0.01,
        tolerance=None,
        max_batches_per_epoch=None,
        compute_detailed_errors=False,
    ):
        if num_factors is not None:
            self.num_factors = num_factors

        if self.standardize:
            train_interactions, mean_train, std_train = standardize_interactions(
                train_interactions
            )
            self.mean_train = mean_train
            self.std_train = std_train
        else:
            _, mean_train, std_train = standardize_interactions(train_interactions)
            self.mean_train = mean_train
            self.std_train = std_train

        self.initililize_embeddings()

        initial_train_RMSE = self.evaluate_RMSE(train_interactions)
        train_errors = [initial_train_RMSE]

        validation_errors = []
        if validation_interactions is not None:
            initial_validation_RMSE = self.evaluate_RMSE(
                validation_interactions, test=True
            )
            validation_errors.append(initial_validation_RMSE)

        pbar_outer = trange(epochs, desc="RMSE: None | Progress: ", leave=False)

        for epoch in pbar_outer:
            np.random.shuffle(train_interactions)

            num_batches = (
                int(np.ceil(len(train_interactions) / batch_size))
                if max_batches_per_epoch is None
                else max_batches_per_epoch
            )

            pbar_inner = trange(
                num_batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False
            )

            for batch_idx in pbar_inner:
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_interactions = train_interactions[start_idx:end_idx]

                if reg_strength > 0:
                    self.update_reg_params(
                        batch_interactions, learning_rate, reg_strength
                    )
                else:
                    self.update_params(batch_interactions, learning_rate)

                if compute_detailed_errors:
                    train_RMSE = self.evaluate_RMSE(train_interactions)
                    train_errors.append(train_RMSE)

            if not compute_detailed_errors:
                train_RMSE = self.evaluate_RMSE(train_interactions)
                train_errors.append(train_RMSE)

            if validation_interactions is not None:
                validation_RMSE = self.evaluate_RMSE(validation_interactions, test=True)
                validation_errors.append(validation_RMSE)

            if tolerance is not None and epoch > 0:
                absolute_improvement = np.abs(train_errors[-2] - train_errors[-1])
                if absolute_improvement / train_errors[-2] < tolerance:
                    pbar_outer.set_description(
                        f"Early Stopping at Epoch {epoch+1} due to tolerance reached | RMSE = {train_errors[-1]:.2e}  | Progress"
                    )
                    break

            pbar_outer.set_description(f"RMSE: {train_errors[-1]:.2e} | Progress")

        pbar_outer.close()
        return train_errors, validation_errors

    def predict(self, users, items):
        user_embeddings_batch = self.user_embeddings[users]
        item_embeddings_batch = self.item_embeddings[items]
        return np.sum(user_embeddings_batch * item_embeddings_batch, axis=1)

    def recommend_items(self, user, top_k=5):
        user_embedding = self.user_embeddings[user]
        scores = np.dot(self.item_embeddings, user_embedding)
        top_item_indices = np.argsort(scores)[::-1][:top_k]
        return top_item_indices

    def update_params(self, interactions, learning_rate):
        errors, users, items = self.evaluate_errors(interactions)

        user_gradients = errors[:, np.newaxis] * self.item_embeddings[items]
        item_gradients = errors[:, np.newaxis] * self.user_embeddings[users]

        np.add.at(self.user_embeddings, users, learning_rate * user_gradients)
        np.add.at(self.item_embeddings, items, learning_rate * item_gradients)

        if self.include_biases:
            self.user_embeddings[:, -1] = 1
            self.item_embeddings[:, -2] = 1

    def update_reg_params(self, interactions, learning_rate, reg_strength):
        (
            unique_users,
            user_gradients,
            unique_items,
            item_gradients,
        ) = self.compute_gradients(interactions, reg_strength)

        self.user_embeddings[unique_users] += learning_rate * user_gradients
        self.item_embeddings[unique_items] += learning_rate * item_gradients

        if self.include_biases:
            self.user_embeddings[:, -1] = 1
            self.item_embeddings[:, -2] = 1

    def compute_gradients(self, interactions, reg_strength):
        errors, users, items = self.evaluate_errors(interactions)

        user_gradients = errors[:, np.newaxis] * self.item_embeddings[items]
        item_gradients = errors[:, np.newaxis] * self.user_embeddings[users]

        unique_users, accumulated_user_gradients = self.accumulate_gradients(
            users, user_gradients
        )
        unique_items, accumulated_item_gradients = self.accumulate_gradients(
            items, item_gradients
        )

        accumulated_user_gradients -= reg_strength * self.user_embeddings[unique_users]
        accumulated_item_gradients -= reg_strength * self.item_embeddings[unique_items]

        return (
            unique_users,
            accumulated_user_gradients,
            unique_items,
            accumulated_item_gradients,
        )

    def accumulate_gradients(self, indices, gradients):
        unique_indices, inverse = np.unique(indices, return_inverse=True)
        accumulated_gradients = np.zeros((unique_indices.shape[0], self.num_factors))
        np.add.at(accumulated_gradients, inverse, gradients)
        return unique_indices, accumulated_gradients

    def evaluate_errors(self, interactions, test=False):
        users, items, ratings = self.split_interactions(interactions)
        predictions = self.predict(users, items)
        if test and self.standardize:
            predictions = predictions * self.std_train + self.mean_train
        errors = ratings - predictions
        return errors, users, items

    def evaluate_RMSE(self, interactions, test=False):
        errors, _, _ = self.evaluate_errors(interactions, test=test)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse

    def initililize_embeddings(self):
        self.num_factors = (
            self.num_factors + 2 if self.include_biases else self.num_factors
        )

        loc = 0 if self.standardize else self.mean_train
        scale = 1 if self.standardize else self.std_train

        self.user_embeddings = np.random.default_rng().normal(
            loc=loc, scale=scale, size=(self.num_users, self.num_factors)
        )
        self.item_embeddings = np.random.default_rng().normal(
            loc=loc, scale=scale, size=(self.num_items, self.num_factors)
        )

        if self.include_biases:
            self.user_embeddings[:, -1] = 1
            self.item_embeddings[:, -2] = 1

    def split_interactions(self, interactions):
        users, items, ratings = np.split(interactions, 3, axis=1)
        users = users.flatten().astype(int)
        items = items.flatten().astype(int)
        ratings = ratings.flatten()
        return users, items, ratings
