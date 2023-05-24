import numpy as np
import pandas as pd
from copy import deepcopy


def initializeQ(m, f):
    return np.random.randn(m, f)


def initializeP(n, f):
    return np.random.randn(n, f)


def get_phi_and_Dphi(func, Dfunc, x_k, p_k):
    def phi(alpha):
        return func(x_k + alpha * p_k)

    def Dphi(alpha):
        return Dfunc(x_k + alpha * p_k).T @ p_k

    return phi, Dphi


def standardize_interactions(
    interactions, mean_interactions=None, std_interactions=None
):
    if mean_interactions is None:
        mean_interactions = interactions[:, 2].mean()
    if std_interactions is None:
        std_interactions = interactions[:, 2].std()
    interactions_standardized = deepcopy(interactions)
    interactions_standardized[:, 2] = (
        interactions_standardized[:, 2] - mean_interactions
    ) / std_interactions
    return interactions_standardized, mean_interactions, std_interactions


def exclude_unknown_users_items_in_test(train_set, test_set):
    """
    Filter a test set to exclude unknown users and items that do not appear in the training set.

    Parameters
    ----------
    train_set : np.array
        A 2D numpy array where the first column represents users, the second column represents items,
        and possibly additional columns represent other information (e.g., ratings).

    test_set : np.array
        A 2D numpy array with the same structure as train_set, which will be filtered to exclude unknown users and items.

    Returns
    -------
    filtered_test_set: np.array
        A filtered version of the test set, excluding rows with users or items not appearing in the training set.
    """
    train_users = set(train_set[:, 0])
    train_items = set(train_set[:, 1])

    mask = np.isin(test_set[:, 0], list(train_users)) & np.isin(
        test_set[:, 1], list(train_items)
    )
    filtered_test_set = test_set[mask]

    return filtered_test_set


def reindex_interactions(interactions):
    """
    Reindex users and items in a set of interactions.

    The reindexing process assigns new indices to each unique user and item. These new indices
    are contiguous integers starting from 0. The number of unique users and items determines
    the range of these indices. After reindexing, users will have indices from 0 to the number
    of unique users - 1, and items will have indices from 0 to the number of unique items - 1.

    Parameters
    ----------
    interactions : np.array
        A 2D numpy array where the first column represents users, the second column represents items,
        and additional columns represent other information (e.g., ratings).

    Returns
    -------
    tuple
        - reindexed_interactions : np.array
            The interactions with users and items reindexed. The new indices for users range from 0 to
            the number of unique users - 1. Similarly, for items, they range from 0 to the number of unique
            items - 1.
        - user_mapping : dict
            A dictionary mapping the original users to the new indices.
        - item_mapping : dict
            A dictionary mapping the original items to the new indices.
    """
    # Create a dataframe from the array
    df = pd.DataFrame(interactions, columns=["user", "item", "rating"])

    # Reindex users and items
    df["user"], user_index = pd.factorize(df["user"])
    df["item"], item_index = pd.factorize(df["item"])

    # Create user and item mapping dictionaries
    user_mapping = {value: index for index, value in enumerate(user_index)}
    item_mapping = {value: index for index, value in enumerate(item_index)}

    # Return the reindexed interactions and the mapping dictionaries
    return df.to_numpy(), user_mapping, item_mapping


def reindex_interactions_based_on_mapping(interactions, user_mapping, item_mapping):
    """
    Reindex users and items in a set of interactions according to a provided mapping.

    Parameters
    ----------
    interactions : np.array
        A 2D numpy array where the first column represents users, the second column represents items,
        and additional columns represent other information (e.g., ratings).

    user_mapping : dict
        A dictionary mapping the original users to the new indices.

    item_mapping : dict
        A dictionary mapping the original items to the new indices.

    Returns
    -------
    np.array
        The reindexed interactions.
    """
    # Create a dataframe from the array
    df = pd.DataFrame(interactions, columns=["user", "item", "rating"])

    # Apply the provided user and item mapping to the interactions
    df["user"] = df["user"].map(user_mapping)
    df["item"] = df["item"].map(item_mapping)

    # Return the reindexed interactions
    return df.to_numpy()
