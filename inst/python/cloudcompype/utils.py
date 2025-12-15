import numpy as np


import pandas as pd
from dataclasses import dataclass


def voxelize_coords(cloud_coords, res, record_n=False):
    """
    Homogenize point cloud density to a custom resolution using voxelization,
    with the option to record the number of original points per voxel.

    Args:
        cloud_coords (np.ndarray): Array of points (N x 3).
        res (float): Voxel resolution. Must be > 0.
        record_n (bool): If True, also return the count of original points per
                         voxel.

    Returns:
        np.ndarray: If record_n=False, returns voxelized points with duplicates
                    removed (N x 3).
                    If record_n=True, returns voxelized points with point
                    counts as fourth column (N x 4).

    Raises:
        ValueError: If resolution is not positive
    """
    if res <= 0:
        raise ValueError("Voxel resolution must be positive")

    if cloud_coords.size == 0:
        return np.empty((0, 3)) if not record_n else np.empty((0, 4))

    # Calculate voxel indices for all points
    voxel_indices = np.round(cloud_coords / res).astype(int)

    if record_n:
        df = pd.DataFrame({
            'x': voxel_indices[:, 0],
            'y': voxel_indices[:, 1],
            'z': voxel_indices[:, 2]
        })

        # Group by voxel coordinates and count occurrences
        grouped = df.groupby(['x', 'y', 'z']).size().reset_index(name='count')

        # Convert back to coordinate format and add counts as fourth column
        voxel_coords = grouped[['x', 'y', 'z']].values.astype(float) * res
        point_counts = grouped['n'].values

        return np.column_stack((voxel_coords, point_counts))
    else:
        # Original behavior: just return unique voxelized coordinates
        cloud_coords_vox = res * np.round(cloud_coords / res)
        cloud_coords_vox = np.unique(cloud_coords_vox, axis=0)
        return cloud_coords_vox


def gini(array):
    """
    Calculate the Gini coefficient of a 1D numpy array to measure inequality in
    a distribution.

    Args:
        array (np.ndarray): Input array of values (N x 1)

    Returns:
        float: Gini coefficient between 0 and 1.

    Note:
        Adapted from https://github.com/oliviaguest/gini

    Raises:
        ValueError: If array is not one-dimensional.
    """
    if array.ndim != 1:
        raise ValueError("Input array must be 1D")

    # Work on a copy to avoid modifying the input array
    array = array.copy()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = array + 0.0000001  # I disabled this to not skew the data
    array = np.sort(array)
    index = np.arange(1, array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


@dataclass
class dtm_config:
    """
    Configuration parameters for Digital Terrain Model (DTM) extraction.

    Attributes:
        resolution (float): Grid resolution for rasterization (default: 0.3)
        n_pts_sor (int): Number of sample points for SOR filtering
                         (default: 50)
        n_sigma_sor (float): Sigma threshold for SOR filtering (default: 0.01)
        i_smooth (int): Number of Laplacian smoothing iterations (default: 20)
        f_smooth (float): Laplacian smoothing factor (default: 0.2)
        n_pts_sampling (int): Number of points for mesh sampling
                              (default: 10000)
        dtm_res (float): Final DTM resolution (default: 0.01)
    """
    resolution: float = 0.3
    n_pts_sor: int = 50
    n_sigma_sor: float = 0.01
    i_smooth: int = 20
    f_smooth: float = 0.2
    n_pts_sampling: int = 10000
    dtm_res: float = 0.01
