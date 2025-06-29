# data_loader.py

"""
Load geographic data and scale numeric variables for the SOM dashboard.
"""

import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

def load_data(path: str) -> gpd.GeoDataFrame:
    """
    Read a GeoPackage (or other file supported by GeoPandas) into a GeoDataFrame.

    Parameters:
        path: filesystem path or URL to the geospatial data.

    Returns:
        A GeoDataFrame containing all layers/columns in the file.
    """
    return gpd.read_file(path)


def scale_data(
    gdf: gpd.GeoDataFrame,
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Standardize all numeric columns (except any in exclude_cols) and return a DataFrame
    for SOM training plus the original GeoDataFrame for mapping.

    Parameters:
        gdf: input GeoDataFrame with numeric and geometry columns.
        exclude_cols: list of column names to skip during scaling (e.g. identifiers).

    Returns:
        scaled_df: pandas DataFrame of standardized numeric features, plus any spatial keys.
        geo_df: the original GeoDataFrame (unchanged).
    """
    exclude_cols = exclude_cols or []

    # select numeric columns, minus any excludes
    numeric = gdf.select_dtypes(include=["number"]).copy()
    numeric.drop(columns=[c for c in exclude_cols if c in numeric.columns], inplace=True, errors='ignore')

    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(numeric.values)
    scaled_df = pd.DataFrame(scaled_arr, columns=numeric.columns, index=gdf.index)

    # carry through any grid coordinates
    for coord in ("hex_x", "hex_y"):
        if coord in gdf.columns:
            scaled_df[coord] = gdf[coord].values

    return scaled_df, gdf
