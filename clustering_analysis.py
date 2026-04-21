from utils import Folder

import pandas as pd
# import json
# import operator
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.cluster import KMeans
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
def sample_points_in_region(region, n_points, sep_min, max_attempts=100000):
    minx, miny, maxx, maxy = region.bounds
    samples = []

    tree = None
    attempts = 0

    while len(samples) < n_points and attempts < max_attempts:
        attempts += 1

        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)

        if not region.contains(Point(x, y)):
            continue

        if tree is not None:
            # Query nearest existing point
            dist, _ = tree.query([x, y], k=1)
            if dist < sep_min:
                continue

        # Accept point
        samples.append([x, y])

        # Rebuild tree periodically (cheap enough for moderate sizes)
        tree = cKDTree(samples)

    if len(samples) < n_points:
        print(f"Warning: only placed {len(samples)} points after {attempts} attempts")

    return np.array(samples)


# -------------------------
# Run CSR simulation
# -------------------------
def csr_simulation(points, n_sim=10, k=5, buffer_radius=5.0, sep_min=0, fname=None):
    n = len(points)

    # observed statistic
    obs = mean_kNN_distance(points, k=k)

    # build support
    region = build_mask(points, buffer_radius)

    sims = []
    for _ in range(n_sim):
        sim_pts = sample_points_in_region(region, n, sep_min=sep_min)
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

# # -------------------------
# # Load points
# # -------------------------
# def load_points(path):
#     df = pd.read_csv(path)
#     return df[['X', 'Y']].values


# # -------------------------
# # Compute bounding area
# # -------------------------
# def compute_area(points):
#     xmin, ymin = points.min(axis=0)
#     xmax, ymax = points.max(axis=0)
#     return (xmax - xmin) * (ymax - ymin)


# -------------------------
# Clark–Evans index
# -------------------------
def clark_evans(points):
    n = len(points)
    area = compute_area(points)
    density = n / area

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)  # k=2 includes self
    nn = dists[:, 1]

    r_obs = nn.mean()
    r_exp = 1 / (2 * np.sqrt(density))

    R = r_obs / r_exp
    return R, r_obs, r_exp


# -------------------------
# Density proxy per point
# -------------------------
def local_density(points, k=5):
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k+1)
    mean_knn = dists[:, 1:].mean(axis=1)
    return 1 / mean_knn  # inverse distance ≈ density


# -------------------------
# Split into density groups
# -------------------------
def split_density_regions(points, n_regions=3):
    dens = local_density(points)

    # cluster by density only (not spatial position)
    km = KMeans(n_clusters=n_regions, n_init=10)
    labels = km.fit_predict(dens.reshape(-1, 1))

    regions = []
    for i in range(n_regions):
        region_pts = points[labels == i]
        if len(region_pts) > 5:  # avoid tiny groups
            regions.append(region_pts)

    return regions


# -------------------------
# Full analysis
# -------------------------
def analyze_clark_evans(points, n_regions=3):
    results = {}

    # overall
    R, r_obs, r_exp = clark_evans(points)
    results["overall"] = {"R": R, "r_obs": r_obs, "r_exp": r_exp}

    # regions
    regions = split_density_regions(points, n_regions)

    region_results = []
    for i, pts in enumerate(regions):
        R, r_obs, r_exp = clark_evans(pts)
        region_results.append({
            "region": i,
            "n_points": len(pts),
            "R": R,
            "r_obs": r_obs,
            "r_exp": r_exp
        })

    results["regions"] = region_results
    return results

# -------------------------
# Build grid
# -------------------------
def make_grid(points, n_cells=100):
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)

    x_edges = np.linspace(xmin, xmax, n_cells + 1)
    y_edges = np.linspace(ymin, ymax, n_cells + 1)

    return x_edges, y_edges


# -------------------------
# Pixel centers
# -------------------------
def grid_centers(x_edges, y_edges):
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2

    xv, yv = np.meshgrid(xc, yc)
    centers = np.column_stack([xv.ravel(), yv.ravel()])

    return centers, xv.shape


# -------------------------
# Density calculation
# -------------------------
def compute_density(points, centers, tol=None):
    tree = cKDTree(points)

    if tol is None:
        # simple binning (count points per nearest pixel)
        # assign each point to nearest center
        _, idx = tree.query(centers, k=1)
        # NOTE: this is not ideal for binning; better use histogram2d below
        raise ValueError("Use histogram2d for simple counts")
    else:
        # count points within radius of each pixel center
        counts = []
        for c in centers:
            neighbors = tree.query_ball_point(c, tol)
            counts.append(len(neighbors))

        return np.array(counts)


# -------------------------
# Fast raw counts per pixel
# -------------------------
def compute_counts_hist(points, x_edges, y_edges):
    H, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=[x_edges, y_edges]
    )
    return H.T  # match meshgrid orientation


# -------------------------
# Save to CSV
# -------------------------
def save_grid(x_edges, y_edges, values, out_path):
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2

    rows = []
    for i, y in enumerate(yc):
        for j, x in enumerate(xc):
            rows.append({
                "X_center": x,
                "Y_center": y,
                "density": values[i, j]
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    plt.figure()

    # extent ensures axes match real coordinates
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    plt.imshow(
        df['density'].to_numpy().reshape((len(x_edges) - 1, len(y_edges) - 1)),
        origin='lower',
        extent=extent,
        aspect='auto'
    )

    plt.colorbar(label="Density")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Density heatmap")

    plt.savefig('images' + out_path[4:-4] + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyse_files(target_directory):
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
            df['Mean distance to 2 nearest neighbours'] = dists[:,0:2].mean(axis=1)
            df['Mean distance to 5 nearest neighbours'] = dists.mean(axis=1)
            df['Density based on 5nn'] = 1/df['Mean distance to 5 nearest neighbours']

            # # Global mean
            # global_mean_2nn = df['Mean distance to 2 nearest neighbours'].mean()
            # global_mean_5nn = df['Mean distance to 5 nearest neighbours'].mean()

            # Optional: mean pairwise distance (vectorised)
            diff = coords[:, None, :] - coords[None, :, :]
            pairwise_dists = np.sqrt((diff ** 2).sum(axis=2))
            min_dist = pairwise_dists.min()
            mean_pairwise = pairwise_dists.sum(axis=1) / (len(df) - 1)
            std_pairwise = None
            df['Mean pairwise distance'] = mean_pairwise

            points = load_points(file.path)
        
            print(file.name)

            result = csr_simulation(
                points,
                n_sim=30,
                k=2,
                buffer_radius=15.0,
                sep_min=min_dist,
                fname=file.path
            )

            for k, v in result.items():
                result[k] = np.round(v, 2)

            print(result, '\n')

            # Save output
            df.to_csv(file.path[:-4] + '_analysed.csv', index=False)


            # Heatmap of density
            n_cells = 50  # 100x100 = 10,000 pixels
            tol = 50.0      # tolerance radius (set None for raw counts)

            x_edges, y_edges = make_grid(points, n_cells)
            centers, shape = grid_centers(x_edges, y_edges)

            if tol is None:
                density = compute_counts_hist(points, x_edges, y_edges)
            else:
                counts = compute_density(points, centers, tol)
                density = counts.reshape(shape)

            save_grid(x_edges, y_edges, density, f"{file.path[:-4]}_analysed_density_map.csv")


def make_summary_table(target_directory):
    data_dict_list = []
    for file in target_directory.files:
        if file.type == 'csv' and 'analysed.' in file.path:
            df = pd.read_csv(file.path)
            global_mean_2nn = round(df['Mean distance to 2 nearest neighbours'].mean(),2)
            global_mean_5nn = round(df['Mean distance to 5 nearest neighbours'].mean(),2)
            # data_dict.update({file.name[:-4]: global_mean_5nn})
            data_dict_list.append({'name': file.name.split('annotation')[0][:-1] + ('_' + file.name[-14] if str.isdigit(file.name[-14]) else ''), 'Mean 2nn distance': global_mean_2nn, 'Mean 5nn distance': global_mean_5nn})
    
            # -------------------------
            # Clark-Evans index analysis of clustering
            # -------------------------

            points = load_points(file.path)
            results = analyze_clark_evans(points, n_regions=3)

            print("\nOverall Clark-Evans:")
            print(results["overall"])
            data_dict_list[-1].update({'Overall Clark-Evans': results['overall']})

            print("\nRegions:")
            for r in results["regions"]:
                print(r)
                data_dict_list[-1].update({f'Region {r["region"]} Clark-Evans': r['R'], f'Region {r["region"]} density': 1/r['r_obs'], f'Number of points in region {r["region"]}': r['n_points']})
            
            density_data = pd.read_csv(file.path[:-12] + 'analysed_density_map.csv')
            
            density_data['density class'] = density_data['density'] < 1E-8
            density_data = density_data.loc[~density_data['density class']]
            mean_density = 10 #density_data['density'].mean() # Seems to be a good middling density to divide regions at
            density_data['density class'] = density_data['density'] > mean_density # WILL ALWAYS GIVE 50:50 if you compare to mean!

            data_dict_list[-1].update({'Percentage coverage high density pixels': round(100*density_data['density class'].mean(), 1)})
            
    # sorted_dict = dict(sorted(data_dict.items(), key=operator.itemgetter(0)))
    # with open('analysed_data_summary.json', 'w') as f:
    #     json.dump(sorted_dict, f)
    pd.DataFrame(data_dict_list).sort_values('name').to_csv('analysed_data_summary.csv')


if __name__ == '__main__':
    target_directory = Folder('data')
    # analyse_files(target_directory)
    make_summary_table(target_directory)

