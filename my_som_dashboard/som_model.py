# som_model.py

"""
Train a MiniSom and compute its U-Matrix (unit distance map) for the dashboard.
"""

from minisom import MiniSom
import numpy as np
from typing import Tuple
import pandas as pd

def train_som(
    data_df: pd.DataFrame,
    x_dim: int = 10,
    y_dim: int = 10,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    iterations: int = 1000,
    random_seed: int = 42
) -> MiniSom:
    """
    Initialize and train a Self-Organizing Map on the given numeric DataFrame.

    Parameters:
        data_df: DataFrame containing only feature columns (no geometry).
        x_dim, y_dim: dimensions of the SOM grid.
        sigma: initial spread of the neighborhood function.
        learning_rate: initial learning rate.
        iterations: number of training steps.
        random_seed: for reproducibility.

    Returns:
        A trained MiniSom instance.
    """
    values = data_df.drop(columns=["hex_x", "hex_y"], errors="ignore").values

    som = MiniSom(
        x_dim, y_dim,
        input_len=values.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=random_seed
    )
    som.random_weights_init(values)
    som.train_random(values, iterations)
    return som


def compute_umatrix(som: MiniSom) -> np.ndarray:
    """
    Compute the flattened U-Matrix (inter-node distances) from the trained SOM.

    Parameters:
        som: a trained MiniSom object.

    Returns:
        A 1D numpy array of length (x_dim * y_dim), containing each unit’s mean distance to neighbors.
    """
    um = som.distance_map()  # shape (x_dim, y_dim)
    return um.flatten()
