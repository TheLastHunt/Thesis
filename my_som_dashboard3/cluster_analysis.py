# cluster_analysis.py

"""
Perform hierarchical clustering on SOM nodes and assign cluster labels to each observation.
"""

import pandas as pd
from minisom import MiniSom
from sklearn.cluster import AgglomerativeClustering


def assign_clusters(
    som: MiniSom,
    data_df: pd.DataFrame,
    n_clusters: int = 5
) -> pd.DataFrame:
    """
    Perform agglomerative clustering on the trained SOM's node weights,
    then assign each data observation to the cluster of its Best Matching Unit.

    Parameters:
        som: trained MiniSom instance.
        data_df: DataFrame used for SOM training (observations × features).
        n_clusters: number of clusters to form on the SOM grid.

    Returns:
        DataFrame: original data_df with three new columns:
            - 'bmu_x', 'bmu_y': coordinates of each observation's BMU node.
            - 'hc_cluster': cluster label for each observation.
    """
    # 1) Cluster the SOM's nodes
    weights = som.get_weights()  # shape (x_dim, y_dim, features)
    x_dim, y_dim, _ = weights.shape
    flat_weights = weights.reshape(x_dim * y_dim, -1)
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    node_labels = hc.fit_predict(flat_weights)

    # 2) For each observation, find its Best Matching Unit (BMU)
    features = data_df.drop(columns=["hex_x", "hex_y"], errors="ignore")
    values = features.values
    bmus = [som.winner(obs) for obs in values]
    bmu_x = [pt[0] for pt in bmus]
    bmu_y = [pt[1] for pt in bmus]
    flat_idx = [i * y_dim + j for i, j in bmus]

    # 3) Assign cluster label based on BMU's node label
    clusters = [int(node_labels[idx]) for idx in flat_idx]

    # 4) Return augmented DataFrame
    df_out = data_df.copy()
    df_out['bmu_x'] = bmu_x
    df_out['bmu_y'] = bmu_y
    df_out['hc_cluster'] = clusters
    return df_out


def compute_cluster_means(
    hex_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute mean of each numeric variable for each cluster.

    Parameters:
        hex_df: DataFrame including 'hc_cluster' and numeric feature columns.

    Returns:
        A DataFrame with 'hc_cluster' and the mean of each numeric column.
    """
    numeric_cols = (
        hex_df
        .select_dtypes(include=[float, int])
        .columns
    )
    numeric_cols = [
        c for c in numeric_cols
        if c not in {"hc_cluster", "bmu_x", "bmu_y", "hex_x", "hex_y"}
    ]
    cluster_means = (
        hex_df
        .groupby('hc_cluster')[numeric_cols]
        .mean()
        .reset_index()
    )
    return cluster_means
