from utils import Folder
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def morans_i(x, y, values):
    """
    Compute Moran's I using inverse distance weights.
    """
    n = len(values)
    coords = np.column_stack((x, y))

    # Distance matrix
    dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    
    # Avoid division by zero (self-distance)
    with np.errstate(divide='ignore'):
        weights = 1 / dist_matrix
    np.fill_diagonal(weights, 0)

    # Normalise weights
    W = weights.sum()

    mean_val = values.mean()
    diff = values - mean_val

    numerator = np.sum(weights * np.outer(diff, diff))
    denominator = np.sum(diff ** 2)

    return (n / W) * (numerator / denominator)


target_directory = Folder('data')

for file in target_directory.files:
    if file.type == 'csv' and 'analysed' not in file.path:
        df = pd.read_csv(file.path)

        coords = df[['X', 'Y']].values
        tree = cKDTree(coords)

        # Query k=6 because the first neighbour is the point itself
        dists, indices = tree.query(coords, k=6)

        # Remove self (first column)
        dists = dists[:, 1:]
        indices = indices[:, 1:]

        # Nearest neighbour (k=1 after removing self)
        df['Nearest neighbour ID'] = indices[:, 0]
        df['Distance to nearest neighbour'] = dists[:, 0]

        # Top 5 neighbours
        df['5 nearest neighbour IDs'] = list(indices)
        df['Mean distance to 5 nearest neighbours'] = dists.mean(axis=1)

        # Global mean
        global_mean_5nn = df['Mean distance to 5 nearest neighbours'].mean()

        # Optional: mean pairwise distance (vectorised)
        diff = coords[:, None, :] - coords[None, :, :]
        pairwise_dists = np.sqrt((diff ** 2).sum(axis=2))
        mean_pairwise = pairwise_dists.sum(axis=1) / (len(df) - 1)
        df['Mean pairwise distance'] = mean_pairwise

        # Moran’s I (using X as "value" placeholder — replace if needed)
        # If you actually have a measurement per cell, use that instead
        df['value_for_moran'] = df['Mean']  # or replace with real measurement
        moran_value = morans_i(df['X'].values, df['Y'].values, df['value_for_moran'].values)

        print(f"{file.name} - Global mean (5NN): {global_mean_5nn:.4f}, Moran's I: {moran_value:.4f}")

        # Save output
        df.to_csv(file.path[:-4] + '_analysed.csv', index=False)