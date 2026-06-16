import os
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt; import matplotlib.cm as cm; import matplotlib.ticker as ticker
import copy
from scipy.spatial import cKDTree
from skimage.segmentation import flood, flood_fill
from skimage import io
from skimage.draw import line


from scipy.ndimage import label, convolve, distance_transform_edt
from skimage.morphology import (
    binary_dilation,
    remove_small_objects,
    disk,
    binary_closing
)


from utils import Folder
from typing import Tuple

def sample_windows(a, window_size, n_samples, allow_overlap=True, seed=None):
    """
    Generate random sampling windows.

    Parameters
    ----------
    a : 2D array
        Binary occupancy mask.

    window_size : int
        Size of square sampling window in pixels.

    n_samples : int
        Number of random samples.

    allow_overlap : bool
        Whether windows may overlap.

    seed : int or None
        Random seed.

    Returns
    -------
    windows : list
        List of tuples:
        (sample_id, y0, y1, x0, x1)
    """

    rng = np.random.default_rng(seed)

    h, w = a.shape

    windows = []

    occupied = []

    max_attempts = 1000
    attempts = 0

    while len(windows) < n_samples and attempts < max_attempts:

        y0 = rng.integers(0, h - window_size)
        x0 = rng.integers(0, w - window_size)

        y1 = y0 + window_size
        x1 = x0 + window_size

        if not allow_overlap:

            overlaps = False

            for oy0, oy1, ox0, ox1 in occupied:

                if not (
                    x1 < ox0 or
                    x0 > ox1 or
                    y1 < oy0 or
                    y0 > oy1
                ):
                    overlaps = True
                    break

            if overlaps:
                attempts += 1
                continue

            occupied.append((y0, y1, x0, x1))

        windows.append(
            (len(windows), y0, y1, x0, x1)
        )

        attempts += 1

    return windows


def connect_cells_to_neighbours(
    seed_mask: np.ndarray,
    k: int = 3,
    max_distance: float = 300
):
    """
    Connect each connected cell object
    to its k nearest neighbours.

    Parameters
    ----------
    seed_mask : np.ndarray
        Binary mask of filtered GFP-positive regions.

    k : int
        Number of nearest neighbours.

    Returns
    -------
    graph_mask : np.ndarray
        Binary mask containing:
            - original cells
            - neighbour connections
    """

    labelled, n_cells = label(seed_mask)

    if n_cells == 0:
        return np.zeros_like(seed_mask, dtype=bool)

    # --------------------------------------------------------
    # Cell centroids
    # --------------------------------------------------------

    centroids = []

    for i in range(1, n_cells + 1):

        coords = np.column_stack(
            np.where(labelled == i)
        )

        centroid = coords.mean(axis=0)

        centroids.append(centroid)

    centroids = np.array(centroids)

    # --------------------------------------------------------
    # Nearest-neighbour graph
    # --------------------------------------------------------

    tree = cKDTree(centroids)

    graph_mask = seed_mask.copy()

    for i, pt in enumerate(centroids):

        distances, neighbours = tree.query(
            pt,
            k=min(k + 1, len(centroids))
        )

        if type(neighbours) is int: # If no neighbours
            continue
        for j, neighbour_idx in enumerate(neighbours[1:]):

            if distances[j] > max_distance:
                continue

            pt1 = centroids[i]
            pt2 = centroids[neighbour_idx]

            rr, cc = line(
                int(pt1[0]),
                int(pt1[1]),
                int(pt2[0]),
                int(pt2[1])
            )

            graph_mask[rr, cc] = True

    return graph_mask


def compute_cell_density(
    a: np.ndarray,
    pixel_area: float = 1.0,
    min_seed_pixels: int = 5,
    locale_radius: int = 8,
    dilation_radius: int = 20,
    min_cell_size: int = 10
):
    """
    Reconstruct biologically plausible GFP-positive tissue
    from sparse/disconnected binary GFP signal.

    IMPORTANT:
        Assumes:
            0 = GFP-positive
            1 = background

    Steps:
    -------
    1. Extract GFP-positive pixels
    2. Remove isolated sparse pixels
    3. Generate effective tissue mask by dilation
    4. Compute connected-cell density inside tissue

    Parameters
    ----------
    a : np.ndarray
        Binary image.

    pixel_area : float
        Area represented by one pixel.

    min_seed_pixels : int
        Minimum neighbouring GFP pixels required locally.

    locale_radius : int
        Radius used to determine whether a pixel belongs
        to a real cell body.

    dilation_radius : int
        Radius for tissue reconstruction.

    min_cell_size : int
        Minimum connected component size after reconstruction.

    Returns
    -------
    dict
    """

    # --------------------------------------------------------
    # GFP-positive pixels
    # --------------------------------------------------------

    gfp = (a == 0)

    coords = np.column_stack(np.where(gfp))

    if len(coords) == 0:

        return {
            "n_cells": 0,
            "cell_density": np.nan,
            "mean_cell_area": np.nan,
            "tissue_mask": np.zeros_like(a, dtype=bool)
        }

    # --------------------------------------------------------
    # Remove isolated pixels using local neighbour count
    # --------------------------------------------------------

    tree = cKDTree(coords)

    keep = np.zeros(len(coords), dtype=bool)

    for i, pt in enumerate(coords):

        neighbours = tree.query_ball_point(pt, r=locale_radius)

        if len(neighbours) >= min_seed_pixels:
            keep[i] = True

    filtered_coords = coords[keep]

    seed_mask = np.zeros_like(a, dtype=bool)

    seed_mask[
        filtered_coords[:, 0],
        filtered_coords[:, 1]
    ] = True

    # --------------------------------------------------------
    # Remove tiny connected components
    # --------------------------------------------------------

    seed_mask = remove_small_objects(
        seed_mask,
        min_size=min_cell_size
    )

    # --------------------------------------------------------
    # Reconstruct tissue occupancy field
    # --------------------------------------------------------

    tissue_mask = binary_dilation(
        seed_mask,
        disk(dilation_radius)
    )

    # --------------------------------------------------------
    # Connected component analysis
    # --------------------------------------------------------

    labelled_cells, n_cells = label(seed_mask)

    cell_sizes = np.bincount(labelled_cells.ravel())[1:]

    valid_cells = cell_sizes >= min_cell_size

    n_valid_cells = np.sum(valid_cells)

    tissue_area = np.sum(tissue_mask) * pixel_area

    density = (
        n_valid_cells / tissue_area
        if tissue_area > 0
        else np.nan
    )

    mean_cell_area = (
        np.mean(cell_sizes[valid_cells]) * pixel_area
        if n_valid_cells > 0
        else np.nan
    )

    return {
        "n_cells": n_valid_cells,
        "cell_density": density,
        "mean_cell_area": mean_cell_area,
        "tissue_area": tissue_area,
        "tissue_mask": tissue_mask
    }
    
def compute_hole_statistics(labelled_holes, pixel_area=1.0):

    hole_sizes = np.bincount(labelled_holes.ravel())[1:]

    if len(hole_sizes) == 0:

        return {
            "mean_hole_area": 0,
            "largest_hole_area": 0,
            "std_hole_area": 0,
            "cv_hole_area": np.nan
        }

    hole_sizes = hole_sizes * pixel_area

    mean_area = np.mean(hole_sizes)
    std_area = np.std(hole_sizes)

    return {
        "mean_hole_area": mean_area,
        "largest_hole_area": np.max(hole_sizes),
        "std_hole_area": std_area,
        "cv_hole_area": (
            std_area / mean_area
            if mean_area > 0
            else np.nan
        )
    }


def hole_density_from_array_extended(
    a: np.ndarray, # Binary file of an image, normalisation not necessary. Everything nonzero is converted to 1.
    name: str, # Base name to save troubleshooting images and final CSV data with
    xrange: Tuple[float, float], # Width of image or section in micrometres
    yrange: Tuple[float, float], # Height of image or section in micrometres

    # GFP seed filtering
    locale_pixel_radius: int = 8, # How big a region to look in (convolve over) for neighbour pixels
    min_seed_pixels: int = 5, # Erase GFP+ regions with fewer neighbour pixels than this

    # tissue reconstruction
    cell_dilation_radius: float = 20, # how many pixels around each GFP+ pixel to dilate - should be big
    network_dilation_radius: float = 5,

    # For sampling - don't sample largely empty windows
    min_gfp_pixels: int = int(5E4), # For whole image. Prevents being too close to an edge.

    # object filtering
    min_hole_size: int = 10, # Don't count holes below this number of pixels towards the total count
    min_cell_size: float = 10, # Distance to dilate background by. Min surviving cell size is essentially 2x this
    min_cell_radius: float = 5,

    image_folder: str = 'troubleshoot_images',
    resolution: float = 1.0, # Convert micrometre lengths to pixels
    mode: str = 'whole_image'
):
    """
    Compute porosity statistics from sparse GFP binary images.

    IMPORTANT
    ---------
    Assumes:
        0 = GFP-positive
        1 = background

    Pipeline
    --------
    1. Identify GFP-positive seed pixels
    2. Remove isolated sparse pixels
    3. Remove tiny connected components
    4. Morphologically reconstruct tissue
    5. Detect enclosed holes
    6. Compute hole + cell statistics

    Returns
    -------
    dict
    """
    # Normalising all length parameters to the resolution
    cell_dilation_radius = cell_dilation_radius * resolution
    network_dilation_radius = network_dilation_radius * resolution
    min_cell_radius = min_cell_radius * resolution


    # ========================================================
    # PIXEL SCALING
    # ========================================================

    dx = (xrange[1] - xrange[0]) / a.shape[1]
    dy = (yrange[1] - yrange[0]) / a.shape[0]

    pixel_area = dx * dy

    blank_output = {
            "name": name,
            "n_holes": 0,
            "hole_density": np.nan,
            "hole_area": 0,
            "hole_coverage": 0,
            "filled_area": 0,
            "mean_hole_area": np.nan,
            "largest_hole_area": np.nan,
            "std_hole_area": np.nan,
            "cv_hole_area": np.nan,
            "n_cells": 0,
            "cell_density": np.nan,
            "mean_cell_area": np.nan
        }

    # ========================================================
    # GFP-POSITIVE PIXELS
    # ========================================================

    gfp = (a == 0)
    if 1.0*(gfp.sum()) < (~gfp).sum(): # Consider boosting background weighting a little
        gfp = ~gfp # Some files appear to be stored with GFP signal as >1 (255) and some as 0, annoyingly.

    if gfp.sum() < min_gfp_pixels:
        return blank_output

    coords = np.column_stack(np.where(gfp))

    if len(coords) == 0:
        return blank_output

    # ========================================================
    # FAST LOCAL DENSITY FILTERING
    # ========================================================

    kernel_size = 2 * locale_pixel_radius + 1

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8
    )

    local_counts = convolve(
        gfp.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0
    )

    seed_mask = (
        gfp &
        (local_counts >= min_seed_pixels)
    )

    fig, ax = plt.subplots()
    ax.imshow(seed_mask, cmap='viridis')
    ax.set_title(f'{name} gfp mask')
    ax.set_xlabel('X / um')
    ax.set_ylabel('Y / um')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / resolution:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / resolution:.2f}'))

    plt.savefig(f'{image_folder}/' + name + f'_seed_mask_{locale_pixel_radius}.png', bbox_inches='tight')
    plt.close(fig)

    if (~seed_mask).sum() < 1E4: # Some sparse images still getting this far annoyingly
        return blank_output

    # ========================================================
    # REMOVE TINY CELL FRAGMENTS
    # ========================================================

    # seed_mask = remove_small_objects(
    #     seed_mask,
    #     min_size=min_cell_size
    # ) # Doesn't seem to do anything

    seed_mask = binary_dilation(
        seed_mask,
        disk(min_cell_radius//2)
    ) # trying this instead to expand non-GFP region a little

    fig, ax = plt.subplots()
    ax.imshow(seed_mask, cmap='viridis')
    ax.set_title(f'{name} filtered cell mask')
    ax.set_xlabel('X / um')
    ax.set_ylabel('Y / um')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / resolution:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / resolution:.2f}'))

    plt.savefig(f'{image_folder}/' + name + f'_seed_mask_noiseless_{round(min_cell_radius/resolution,1)}.png', bbox_inches='tight')
    plt.close(fig)


    # ========================================================
    # Expand remaining regions to be cell-sized, in case they were previously fragmented
    # ========================================================
    
    seed_mask = ~(binary_dilation(
        ~seed_mask,
        disk(cell_dilation_radius)
    )) # merge remaining fragments of cells

    fig, ax = plt.subplots()
    ax.imshow(seed_mask, cmap='viridis')
    ax.set_title(f'{name} re-dilated cells')
    ax.set_xlabel('X / um')
    ax.set_ylabel('Y / um')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / resolution:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / resolution:.2f}'))

    plt.savefig(f'{image_folder}/' + name + f'_seed_mask_expanded_cells_{round(cell_dilation_radius/resolution,1)}.png', bbox_inches='tight')
    plt.close(fig)

    # ========================================================
    # Draw lines between cells and their 5 nearest neighbours
    # ========================================================

    seed_mask = connect_cells_to_neighbours(
        ~seed_mask,
        k=10,
        max_distance=400
    )

    seed_mask = ~(binary_dilation(
            seed_mask,
            disk(network_dilation_radius)
    ))

    fig, ax = plt.subplots()
    ax.imshow(seed_mask, cmap='viridis')
    ax.set_title(f'{name} cell network')
    ax.set_xlabel('X / um')
    ax.set_ylabel('Y / um')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / resolution:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / resolution:.2f}'))

    plt.savefig(f'{image_folder}/' + name + f'_seed_mask_connected{round(network_dilation_radius,1)}.png', bbox_inches='tight')
    plt.close(fig)

    # ========================================================
    # CELL STATISTICS
    # ========================================================

    labelled_cells, _ = label(seed_mask)

    cell_sizes = np.bincount(
        labelled_cells.ravel()
    )[1:]

    valid_cells = cell_sizes >= min_cell_size

    n_cells = np.sum(valid_cells)
    if int(n_cells) < 50:
        return blank_output

    # ========================================================
    # RECONSTRUCT EFFECTIVE TISSUE
    # ========================================================

    # tissue_mask = binary_dilation(
    #     ~seed_mask,
    #     disk(dilation_radius)
    # ) # Binary dilation expands bright regions. i.e. 1s so we need to invert an image where the signal is 0
    # Really slow when radius is large. Replacing with a distance calculation

    # Distance to nearest GFP-positive seed
    # dist = distance_transform_edt(seed_mask)

    # Tissue occupancy field
    # tissue_mask = dist <= dilation_radius

    # tissue_mask = ~tissue_mask

    # plt.imsave(f'{image_folder}/' + name + f'_tissue_mask_{dilation_radius}.png', tissue_mask)

    if mode == 'windows':
        tissue_mask = io.imread(f"final tif files/{name.split('/')[-1].split('_sample')[0]}_roi.tif")
        tissue_mask = tissue_mask[yrange[0]:yrange[1]].T[xrange[0]:xrange[1]].T
    else:
        tissue_mask = io.imread(f"final tif files/{name.split('/')[-1][:-4]}_roi.tif")
    pixel_area = resolution**-2 # Square micrometres per pixel

    tissue_area = np.sum(tissue_mask) * pixel_area

    cell_density = (
        n_cells / tissue_area
        if tissue_area > 0
        else np.nan
    )

    mean_cell_area = (
        np.mean(cell_sizes[valid_cells]) * pixel_area
        if n_cells > 0
        else np.nan
    )

    # ========================================================
    # HOLE DETECTION
    # ========================================================

    # dist = distance_transform_edt(~seed_mask)
    # holes_mask = dist <= network_dilation_radius

    labelled_holes, n_components = label(seed_mask)

    # Reshuffle colours for visibility
    shuffled = np.random.permutation(np.arange(1, n_components + 1))

    # Prepend 0 so background (label 0) maps to 0
    mapping = np.concatenate([[0], shuffled])

    # --- Apply the mapping to the labelled array ---
    shuffled_labels = mapping[labelled_holes]

    fig, ax = plt.subplots()
    ax.imshow(shuffled_labels, cmap='viridis')
    ax.set_title(f'{name} identified holes')
    ax.set_xlabel('X / um')
    ax.set_ylabel('Y / um')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / resolution:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / resolution:.2f}'))

    plt.savefig(f'{image_folder}/' + name + f'_labelled_holes_{min_hole_size}.png', bbox_inches='tight')
    plt.close(fig)

    retained_holes = np.zeros_like(
        labelled_holes,
        dtype=np.int32
    )

    holes = 0
    hole_area_pixels = 0

    current_label = 1

    for i in range(1, n_components + 1):

        component = (labelled_holes == i)

        touches_edge = (
            component[0, :].any() or
            component[-1, :].any() or
            component[:, 0].any() or
            component[:, -1].any()
        )

        n_pixels = component.sum()

        if (
            (not touches_edge and
            n_pixels >= min_hole_size) and
            n_pixels <= 1E5 # Max hole size to ignore background etc
        ):

            retained_holes[component] = current_label

            current_label += 1

            holes += 1
            hole_area_pixels += n_pixels

    # ========================================================
    # HOLE STATISTICS
    # ========================================================

    hole_sizes = np.bincount(
        retained_holes.ravel()
    )[1:]

    hole_sizes_scaled = hole_sizes * pixel_area

    hole_area = hole_area_pixels * pixel_area

    hole_fraction = (
        hole_area /
        (hole_area + tissue_area)
        if (hole_area + tissue_area) > 0
        else np.nan
    )

    if len(hole_sizes_scaled) > 0:

        mean_hole_area = np.mean(hole_sizes_scaled)

        largest_hole_area = np.max(hole_sizes_scaled)

        std_hole_area = np.std(hole_sizes_scaled)

        cv_hole_area = (
            std_hole_area / mean_hole_area
            if mean_hole_area > 0
            else np.nan
        )

    else:

        mean_hole_area = 0
        largest_hole_area = 0
        std_hole_area = 0
        cv_hole_area = np.nan

    # ========================================================
    # OUTPUT
    # ========================================================

    return {
        "name": name,

        # hole statistics
        "n_holes": holes,
        "hole_density": (
            holes / tissue_area
            if tissue_area > 0
            else np.nan
        ),
        "hole_area": hole_area,
        "hole_coverage": hole_fraction,

        "mean_hole_area": mean_hole_area,
        "largest_hole_area": largest_hole_area,
        "std_hole_area": std_hole_area,
        "cv_hole_area": cv_hole_area,

        # tissue statistics
        "filled_area": tissue_area,
        "n_cells": int(n_cells),
        "cell_density": cell_density,
        "mean_cell_area": mean_cell_area,

        # optional debugging output
        # "tissue_mask": tissue_mask,
        # "seed_mask": seed_mask
    }

def get_adjacent_indices(ij: Tuple[int, int], m: int, n: int):
    adjacent_indices = []
    if ij[0] > 0:
        adjacent_indices.append((ij[0]-1,ij[1]))
    if ij[0]+1 < m:
        adjacent_indices.append((ij[0]+1,ij[1]))
    if ij[1] > 0:
        adjacent_indices.append((ij[0],ij[1]-1))
    if ij[1]+1 < n:
        adjacent_indices.append((ij[0],ij[1]+1))
    return set(adjacent_indices)

# def set_bg_to_nan(labelled):
#     edge_points = {(0,0)}
#     m, n = labelled.shape
#     while len(edge_points) > 0:
#         pt = edge_points.pop()
#         if labelled[pt] == 1:
#             labelled[pt] = 2
#             edge_points = edge_points.union(get_adjacent_indices(pt, labelled.shape[0], labelled.shape[1]))
#     return labelled

def flood_fill_bg(a, old_val, new_val):
    for pt in [(0,0), a.shape, (0,a.shape[1]), (a.shape[0],0)]:
        if a[pt] == old_val:
            filled_a = flood_fill(a, pt, new_val)
            break
    return filled_a


def hole_density_from_array(a, name, xrange, yrange):
    """
    Computes:
        (# of NaN holes inside region) / (area of region)
    """

    # -------------------------
    # Masks
    # -------------------------
    iszero = a == 0
    valid = ~iszero

    # -------------------------
    # Label NaN regions
    # -------------------------
    labelled, n_components = label(iszero)

    holes = 0
    hole_area = 0.0
    # labelled = flood_fill_bg(labelled, 1, 2)

    for i in range(1, n_components + 1):
        component = (labelled == i)

        # check if touches boundary
        touches_edge = (
            component[0, :].any() or
            component[-1, :].any() or
            component[:, 0].any() or
            component[:, -1].any()
        )

        n_pixels = component.sum()

        if not touches_edge and n_pixels > 10:
            holes += 1
            hole_area += n_pixels

    # -------------------------
    # Area of valid region
    # -------------------------
    # pixel area (assumes regular grid)
    # x_vals = grid.columns.values
    # y_vals = grid.index.values
    dx, dy = (xrange[1]-xrange[0])/a.shape[0], (yrange[1]-yrange[0])/a.shape[1]  # Set scale based on input data in future

    # dx = np.abs(x_vals[1] - x_vals[0])
    # dy = np.abs(y_vals[1] - y_vals[0])
    pixel_area_unit = dx * dy
    # filled_area = np.sum(a)

    filled_area = valid.sum() * pixel_area_unit
    hole_area = hole_area * pixel_area_unit
    hole_density = holes / filled_area if filled_area > 0 else np.nan

    return {
        "name": name,
        "n_holes": holes,
        "area": filled_area,
        "hole_density": hole_density,
        "hole_area": hole_area,
        "hole_coverage": hole_area/(hole_area+filled_area)
    }

def make_grid(points, n_cells=100, margin=0):
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)

    x_edges = np.linspace(xmin-margin, xmax+margin, n_cells + 1)
    y_edges = np.linspace(ymin-margin, ymax+margin, n_cells + 1)
    # Adding a 20 unit margin so features don't get cut off

    return x_edges, y_edges

def grid_centers(x_edges, y_edges):
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2

    xv, yv = np.meshgrid(xc, yc)
    centers = np.column_stack([xv.ravel(), yv.ravel()])

    return centers, xv.shape

def compute_counts_hist(points, x_edges, y_edges):
    H, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=[x_edges, y_edges]
    )
    return H.T  # match meshgrid orientation

def compute_density(points, centers, tol, edge_tol):
    tree = cKDTree(points)
    # count points within radius tol of each pixel center
    counts = []
    for c in centers:
        neighbours = tree.query_ball_point(c, tol)
        if len(neighbours)==0:
            neighbours = tree.query_ball_point(c, edge_tol)
            if len(neighbours)==0:
                counts.append(np.nan)
            else:
                counts.append(0.0)
        else:
            counts.append(len(neighbours)/(np.pi*tol**2/100))

    return np.array(counts)

def save_grid(x_edges, y_edges, values, out_path):
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2

    rows = []
    for i, y in enumerate(yc):
        for j, x in enumerate(xc):
            rows.append({
                "X_center": x,
                "Y_center": y,
                "Density (cells per 100 um^2)": values[i, j]
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    plt.figure()

    # extent ensures axes match real coordinates
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])

    a = df['Density (cells per 100 um^2)'].to_numpy().reshape((len(x_edges) - 1, len(y_edges) - 1))
    
    masked_a = np.ma.array(a, mask=a>0)
    # plt.imshow(
    #     masked_a,
    #     origin='upper',
    #     extent=extent,
    #     aspect='equal',
    #     cmap='Greys',
    #     interpolation='nearest',
    # )

    # plt.colorbar(label=r"Density (cells per 100 $\mu m^2$)")
    # plt.xlabel(r"$X \ (\mu m)$")
    # plt.ylabel(r"$Y \ (\mu m)$")
    # plt.title("CXCL12 cell density heatmap")

    # plt.savefig('images for album' + out_path[4:-4] + '.png', dpi=300, bbox_inches='tight')
    # plt.close()

    out_a = np.where(a>0, 1, 0)
    return out_a

def get_porosities(target_directory, dilation_radius, tissue_tol):
    out_data = []
    for file in target_directory.files:
        if (file.type == 'csv' and 'analysed' not in file.path) and ('Slide' in file.path and 'duplicates_removed' in file.path) and not 'mask' in file.path:    
            points = file.read()[['X', 'Y']].values
            xrange, yrange = abs(points[-1,0]-points[0,0]), abs(points[-1,1]-points[0,1])


            target_n_cells = 300  # resulting map will have ~target_n_cells^2 pixels.

            x_edges, y_edges = make_grid(points, target_n_cells, margin=0)
            centers, shape = grid_centers(x_edges, y_edges)


            counts = compute_density(points, centers, tol=dilation_radius, edge_tol=tissue_tol)
            density = counts.reshape(shape)

            # Pixelated heatmap of the density data from the original CSV
            grid = save_grid(x_edges, y_edges, density, f"{file.path[:-4]}_dilated_tissue_mask.csv")


            out_data.append(hole_density_from_array(grid, file.name, xrange, yrange))
    return pd.DataFrame(out_data)

def get_porosities_whole_directory(target_directory, dilation_radius, tissue_tol):
    out_data = []
    for file in target_directory.files:
        if (file.type == 'csv') and not ('dilated_tissue_mask' in file.name) and not ('hole_densities' in file.name):    
            points = file.read()[['X', 'Y']].values
            xrange, yrange = abs(points[-1,0]-points[0,0]), abs(points[-1,1]-points[0,1])

            target_n_cells = 300  # resulting map will have ~target_n_cells^2 pixels.

            x_edges, y_edges = make_grid(points, target_n_cells, margin=0)
            centers, shape = grid_centers(x_edges, y_edges)


            counts = compute_density(points, centers, tol=dilation_radius, edge_tol=tissue_tol)
            density = counts.reshape(shape)

            # Pixelated heatmap of the density data from the original CSV
            grid = save_grid(x_edges, y_edges, density, f"{file.path[:-4]}_dilated_tissue_mask.csv")


            out_data.append(hole_density_from_array(grid, file.name, xrange, yrange))
    return pd.DataFrame(out_data)

def analyze_image(
    a: np.ndarray,
    xrange: Tuple[float, float],
    yrange: Tuple[float, float],
    sample_windows: bool = True,
    window_size: int = 1000,
    n_samples: int = 30,
    allow_overlap: bool = True,
    filename: str = 'image',
    image_folder: str = 'troubleshoot_images',
    resolution: float = 1   # Pixels per micrometre
):
    if sample_windows:
        return analyze_resampled_windows(
            a,
            xrange,
            yrange,
            window_size=window_size, # Micrometres
            n_samples=n_samples,
            allow_overlap=allow_overlap,
            file_name=filename,
            locale_pixel_radius=5,
            min_seed_pixels=20,
            cell_dilation_radius=4.0,
            network_dilation_radius=0.5,
            min_gfp_pixels=int(7E6),
            min_hole_size=int(5E3), # Trying to remove normal spacing from counts
            min_cell_size=10,
            min_cell_radius=3,
            image_folder=image_folder,
            resolution=resolution,
        )
    else:
        return pd.DataFrame([
            hole_density_from_array_extended(
                a,
                name=filename,
                xrange=xrange,
                yrange=yrange,
                locale_pixel_radius=4,
                min_seed_pixels=20,
                cell_dilation_radius=2.0,
                network_dilation_radius=0.5,
                min_gfp_pixels=int(5E4),
                min_hole_size=50,
                min_cell_size=10,
                min_cell_radius=3,
                image_folder=image_folder,
                resolution=resolution
            )
        ])

def analyze_resampled_windows(
        a: np.ndarray, # 2D binary occupancy mask
        xrange: Tuple[float, float], # width of whole image
        yrange: Tuple[float, float], # height of whole image
        window_size=100, # Entered in Micrometres, converted to pixels
        n_samples=50,
        allow_overlap=True,
        file_name='Image',
        locale_pixel_radius=4,
        min_seed_pixels=20,
        cell_dilation_radius=2.0,
        network_dilation_radius=0.5,
        min_gfp_pixels=int(5E4),
        min_hole_size=50,
        min_cell_size=10,
        min_cell_radius=3,
        image_folder='troubleshoot_images',
    resolution=1.0
    ):
    window_size = int(window_size * resolution) # resolution in pixels per um, so window_size in pixels after here
    results = []
    while len(results) < 5:
        windows = sample_windows(
            a,
            window_size,
            n_samples,
            allow_overlap=allow_overlap
        )

        for sample_id, y0, y1, x0, x1 in windows:

            sub_a = a[y0:y1, x0:x1]

            x_scale = (xrange[1] - xrange[0]) * (window_size / a.shape[1])
            y_scale = (yrange[1] - yrange[0]) * (window_size / a.shape[0])

            out = hole_density_from_array_extended(
                sub_a,
                name=f"{file_name[:-4]}_sample_{sample_id}",
                xrange=(x0,x1),
                yrange=(y0,y1),
                locale_pixel_radius=locale_pixel_radius,
                min_seed_pixels=min_seed_pixels,
                cell_dilation_radius=cell_dilation_radius,
                network_dilation_radius=network_dilation_radius,
                min_gfp_pixels=min_gfp_pixels,
                min_hole_size=min_hole_size,
                min_cell_size=min_cell_size,
                min_cell_radius=min_cell_radius,
                image_folder=image_folder,
                resolution=resolution,
                mode='windows'
            )

            out["sample_id"] = sample_id
            out["x0_px"] = x0
            out["y0_px"] = y0
            out["x1_px"] = x1
            out["y1_px"] = y1
            for coord in ("x0", "y0", "x1", "y1"):
                out[f"{coord}_um"] = round(out[f"{coord}_px"]/resolution, 2)

            if pd.DataFrame([out]).isna().sum().sum() > 1:
                continue
            elif out['n_cells'] < 50:
                continue
            else:
                results.append(out)

    

    return pd.concat([pd.DataFrame(results)])


from PIL import Image
from PIL.TiffTags import TAGS
Image.MAX_IMAGE_PIXELS = 933120000
import warnings

if __name__ == '__main__':
    dilation_radius = 55.0
    tissue_edge_tol = 60.0

    # target_directory = Folder('data')
    # out_data = get_porosities(target_directory, dilation_radius=dilation_radius, tissue_tol = tissue_edge_tol) # Units of micrometres
    # out_data.to_csv(f'hole_densities_{dilation_radius}_{tissue_edge_tol}.csv')

    # target_directory = Folder('mask particle csvs from Clara')
    # out_data = get_porosities_whole_directory(target_directory, dilation_radius=dilation_radius, tissue_tol = tissue_edge_tol) # Units of micrometres
    # out_data.to_csv(f'mask particle csvs from Clara/hole_densities_{dilation_radius}_{tissue_edge_tol}.csv')

    target_directory = Folder('mask tif files from Clara')
    hold_list = ['M5hip_right_bin.tif']
    for file in target_directory.files:
        if file.type == 'tif':
            if file.name.split('_')[-1] == 'bin.tif' and not file.name in hold_list:
                image = io.imread(file.path)

                with Image.open(file.path) as img:
                    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
                resolution = meta_dict['XResolution'][0]
                if resolution != meta_dict['YResolution'][0]:
                    raise ValueError('Axes not set equal in Tif image.')
                if not 'micron' in meta_dict['ImageDescription'][0]:
                    warnings.warn("Warning - image length not in units of Micrometres.")
                resolution = resolution[0]/resolution[1]

                df = analyze_image(
                    a=image,
                    xrange=(0, image.shape[0]),
                    yrange=(0, image.shape[1]),
                    sample_windows=True, # Process full image or not - full image is usually too big!
                    filename=file.name,
                    image_folder='troubleshoot_images',
                    resolution=resolution,
                    window_size=1500,
                    n_samples=5,
                    allow_overlap=True
                )
                df.dropna().to_csv(file.path[:-4] + '_resampled_analysed.csv')
        # except:
        #     print(f'Error on {file.path}, skipping to next.')