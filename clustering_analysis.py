from utils import Folder

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union
import random

import matplotlib.pyplot as plt

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

def local_counts(tree, points, r):
    """
    Count neighbors within radius r for each point (excluding self).
    """
    counts = tree.query_ball_point(points, r, return_length=True)
    return counts - 1  # remove self-count


def simulate_csr(bounds, n_points):
    """
    Generate uniform random points in bounding box.
    bounds: [(xmin, xmax), (ymin, ymax), ...]
    """
    dim = len(bounds)
    pts = np.zeros((n_points, dim))
    for d, (low, high) in enumerate(bounds):
        pts[:, d] = np.random.uniform(low, high, size=n_points)
    return pts


def local_clustering_significance(points, r, n_sim=100):
    """
    Compute local clustering significance vs CSR.

    Returns:
        observed_counts
        z_scores
        p_values (two-sided)
    """
    n = len(points)
    dim = points.shape[1]

    # Build tree for observed data
    tree = cKDTree(points)
    observed = local_counts(tree, points, r)

    # Bounding box
    bounds = [(points[:, d].min(), points[:, d].max()) for d in range(dim)]

    # Null simulations
    sim_counts = np.zeros((n_sim, n))

    for i in range(n_sim):
        sim_pts = simulate_csr(bounds, n)
        sim_tree = cKDTree(sim_pts)

        # IMPORTANT: evaluate at observed locations
        sim_counts[i] = local_counts(sim_tree, points, r)

    # Mean and std under null
    mu = sim_counts.mean(axis=0)
    sigma = sim_counts.std(axis=0, ddof=1)

    # Avoid divide-by-zero
    sigma[sigma == 0] = np.nan

    z = (observed - mu) / sigma

    # Two-sided empirical p-values
    p = np.mean(np.abs(sim_counts - mu) >= np.abs(observed - mu), axis=0)

    return observed, z, p

def ripley_k_l(points, radii, area, tree=None, edge_correction=True):
    """
    Compute Ripley's K and L functions.

    Parameters:
        points (ndarray): Nx2 array of (x, y)
        radii (ndarray): radii at which to evaluate K
        area (float): total observation window area
        tree (cKDTree): optional precomputed tree
        edge_correction (bool): apply border correction

    Returns:
        K (ndarray), L (ndarray)
    """
    n = len(points)
    intensity = n / area  # lambda

    if tree is None:
        tree = cKDTree(points)

    # Bounding box (for edge correction)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)

    K = np.zeros_like(radii, dtype=float)

    for i, r in enumerate(radii):
        counts = []

        for idx, p in enumerate(points):
            if edge_correction:
                # Skip points too close to boundary
                if (p[0] - xmin < r or xmax - p[0] < r or
                    p[1] - ymin < r or ymax - p[1] < r):
                    continue

            # Count neighbors within r (exclude self)
            neighbors = tree.query_ball_point(p, r)
            counts.append(len(neighbors) - 1)

        if len(counts) == 0:
            K[i] = np.nan
        else:
            K[i] = np.mean(counts) / intensity

    # L-function
    L = np.sqrt(K / np.pi)

    return K, L


def load_points_from_csv(path):
    df = pd.read_csv(path)
    return df[['X', 'Y']].values


def compute_area(points):
    # Simple bounding box area (replace if you have a mask/window)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    return (xmax - xmin) * (ymax - ymin)

# -------------------------
# Load data
# -------------------------
def load_points(path):
    df = pd.read_csv(path)
    return df[['X', 'Y']].values


# -------------------------
# 5-NN distance
# -------------------------
def mean_kNN_distance(points, k=5):
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k+1)  # includes self
    return np.mean(dists[:, 1:])  # drop self


# -------------------------
# Build spatial mask
# -------------------------
def build_mask(points, buffer_radius=5.0):
    circles = [Point(p).buffer(buffer_radius) for p in points]
    region = unary_union(circles)
    return region


# -------------------------
# Sample uniformly in mask
# -------------------------
def sample_points_in_region(region, n_points):
    minx, miny, maxx, maxy = region.bounds
    samples = []

    while len(samples) < n_points:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)

        if region.contains(Point(x, y)):
            samples.append([x, y])

    return np.array(samples)


# -------------------------
# Run CSR simulation
# -------------------------
def csr_simulation(points, n_sim=10, k=5, buffer_radius=5.0, fname=None):
    n = len(points)

    # observed statistic
    obs = mean_kNN_distance(points, k=k)

    # build support
    region = build_mask(points, buffer_radius)

    sims = []
    for _ in range(n_sim):
        sim_pts = sample_points_in_region(region, n)
        sims.append(mean_kNN_distance(sim_pts, k=k))

    sims = np.array(sims)

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=5, label="Real")
    plt.scatter(sim_pts[:, 0], sim_pts[:, 1], s=5, label="CSR", alpha=0.6)
    plt.legend()
    plt.title("Real vs CSR point patterns")
    plt.axis("equal")
    plt.savefig(fname[:-4] + '_simulated.png')
    plt.close()

    # standardization
    mean_sim = sims.mean()
    std_sim = sims.std()

    z = (obs - mean_sim) / std_sim if std_sim > 0 else np.nan

    return {
        "observed": obs,
        "random_mean": mean_sim,
        "random_std": std_sim,
        "z_score": z,
        "sim_values": sims
    }


target_directory = Folder('data')

for file in target_directory.files:
    if file.type == 'csv' and 'analysed' not in file.path:
        df = pd.read_csv(file.path)
        df.index = df[' ']
        df.drop(columns=[' '], inplace=True)

        coords = df[['X', 'Y']].values
        tree = cKDTree(coords)

        # Query k=6 because the first neighbour is the point itself
        dists, indices = tree.query(coords, k=6)

        # Remove self (first column)
        dists = dists[:, 1:]
        indices = indices[:, 1:] + min(df.index) # Also start counting from 1 as in the data

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

        points = load_points(file.path)
    
        print(file.name)

        result = csr_simulation(
            points,
            n_sim=30,
            k=2,
            buffer_radius=15.0,
            fname=file.path
        )

        for k, v in result.items():
            result[k] = np.round(v, 2)

        print(result, '\n')

        # Save output
        df.to_csv(file.path[:-4] + '_analysed.csv', index=False)
