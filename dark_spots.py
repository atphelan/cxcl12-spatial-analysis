import os
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt, cm, ticker, colors
from scipy.spatial import cKDTree
from scipy.ndimage import label, convolve
from skimage import io
from skimage.draw import line
from skimage.morphology import binary_dilation, disk
import copy

from utils import Folder
from typing import Tuple

def sample_windows(a, window_size, n_samples, window_ids=None, allow_overlap=True, seed=None):
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

    window_ids : List(int)
        List of numbers to label windows with.

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
    if window_ids is None:
        window_ids = list(range(1,n_samples+1))
    elif len(window_ids) < n_samples:
        n_samples = len(window_ids)

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
            (window_ids.pop(), y0, y1, x0, x1)
        )

        attempts += 1

    return windows


def sample_windows_modes(
    a,
    window_size,
    n_samples,
    window_ids=None,
    mode='random',          # 'random' | 'no_overlap' | 'tiled'
    overlap_fraction=0.05,  # Only used in 'tiled' mode — max overlap as fraction of window_size
    seed=None
):
    """
    Generate sampling windows across a 2D binary occupancy mask.

    Parameters
    ----------
    a : 2D array
        Binary occupancy mask.

    window_size : int
        Size of square sampling window in pixels.

    n_samples : int
        Number of windows to generate. Ignored in 'tiled' mode,
        which fills the image as best as possible.

    window_ids : list of int or None
        Labels for windows. Defaults to 1..n_samples.
        Ignored in 'tiled' mode (auto-labelled).

    mode : str
        Placement strategy:
            'random'     — fully random placement, overlaps allowed.
            'no_overlap' — random placement, no overlaps permitted.
            'tiled'      — systematic grid fill with up to
                           overlap_fraction * window_size overlap
                           allowed at each edge.

    overlap_fraction : float
        Maximum overlap as a fraction of window_size on each edge.
        Only used in 'tiled' mode. E.g. 0.05 = 5% of window_size.

    seed : int or None
        Random seed (used in 'random' and 'no_overlap' modes).

    Returns
    -------
    windows : list of tuples
        Each tuple: (sample_id, y0, y1, x0, x1)

    report : dict
        Placement summary:
            'mode'              : mode used
            'requested'         : n_samples requested (None if tiled)
            'placed'            : number of windows placed
            'max_attempts'      : attempt limit (random/no_overlap only)
            'attempts_used'     : attempts made  (random/no_overlap only)
            'coverage_fraction' : fraction of image area covered
            'warning'           : warning string or None
    """

    h, w = a.shape
    windows = []
    report = {
        'mode':              mode,
        'requested':         n_samples if mode != 'tiled' else None,
        'placed':            0,
        'max_attempts':      None,
        'attempts_used':     None,
        'coverage_fraction': 0.0,
        'warning':           None
    }

    # --------------------------------------------------------
    # TILED MODE
    # --------------------------------------------------------

    if mode == 'tiled':

        max_overlap_px = int(overlap_fraction * window_size)
        step = window_size - max_overlap_px  # minimum step between window origins

        if step <= 0:
            raise ValueError(
                f"overlap_fraction={overlap_fraction} is too large — "
                f"step size would be <= 0 for window_size={window_size}."
            )

        sample_id = 1

        y_starts = list(range(0, h - window_size + 1, step))
        x_starts = list(range(0, w - window_size + 1, step))

        if not y_starts or not x_starts:
            report['warning'] = (
                f"window_size={window_size} is larger than the image "
                f"({h}x{w}) — no windows placed."
            )
            return windows, report

        for y0 in y_starts:
            for x0 in x_starts:

                y1 = y0 + window_size
                x1 = x0 + window_size

                windows.append((sample_id, y0, y1, x0, x1))
                sample_id += 1

        # Report uncovered border strips
        y_gap = h - (y_starts[-1] + window_size)
        x_gap = w - (x_starts[-1] + window_size)

        if y_gap > 0 or x_gap > 0:
            report['warning'] = (
                f"Tiled placement left uncovered border strips: "
                f"bottom={y_gap}px, right={x_gap}px. "
                f"Consider adjusting window_size or overlap_fraction."
            )

    # --------------------------------------------------------
    # RANDOM / NO_OVERLAP MODES
    # --------------------------------------------------------

    else:

        if window_ids is None:
            window_ids = list(range(1, n_samples + 1))
        elif len(window_ids) < n_samples:
            n_samples = len(window_ids)

        rng = np.random.default_rng(seed)
        occupied = []
        max_attempts = 1000
        attempts = 0

        report['max_attempts'] = max_attempts

        while len(windows) < n_samples and attempts < max_attempts:

            y0 = rng.integers(0, h - window_size)
            x0 = rng.integers(0, w - window_size)
            y1 = y0 + window_size
            x1 = x0 + window_size

            if mode == 'no_overlap':

                overlaps = any(
                    not (x1 < ox0 or x0 > ox1 or y1 < oy0 or y0 > oy1)
                    for oy0, oy1, ox0, ox1 in occupied
                )

                if overlaps:
                    attempts += 1
                    continue

                occupied.append((y0, y1, x0, x1))

            windows.append((window_ids.pop(), y0, y1, x0, x1))
            attempts += 1

        report['attempts_used'] = attempts

        if len(windows) < n_samples:
            report['warning'] = (
                f"Only placed {len(windows)}/{n_samples} windows "
                f"after {max_attempts} attempts. "
                f"Image may be too small or too densely packed for "
                f"the requested number of non-overlapping windows."
            )

    # --------------------------------------------------------
    # SHARED REPORTING
    # --------------------------------------------------------

    report['placed'] = len(windows)

    total_window_area = len(windows) * (window_size ** 2)
    report['coverage_fraction'] = round(
        min(total_window_area / (h * w), 1.0), 4
    )

    # Print report
    print(f"\n--- sample_windows report ---")
    print(f"Mode            : {report['mode']}")
    if report['requested'] is not None:
        print(f"Requested       : {report['requested']}")
    print(f"  Placed          : {report['placed']}")
    if report['attempts_used'] is not None:
        print(f"Attempts used   : {report['attempts_used']} / {report['max_attempts']}")
    print(f"  Coverage        : {report['coverage_fraction']*100:.1f}% of image area")
    if report['warning']:
        print(f"WARNING       : {report['warning']}")
    print(f"-----------------------------\n")

    return windows, report


def connect_cells_to_neighbours(
    seed_mask: np.ndarray,
    k: int = 3,
    max_distance: float = 300,
    window_border_distance: float = None  # If not None, connect nearby cells to window edge
):
    """
    Connect each connected cell object to its k nearest neighbours,
    and optionally connect cells near the window border to the edge.

    Parameters
    ----------
    seed_mask : np.ndarray
        Binary mask of filtered GFP-positive regions.

    k : int
        Number of nearest neighbours.

    max_distance : float
        Maximum distance in pixels to connect neighbours.

    window_border_distance : float or None
        Cells with centroids within this many pixels of any window
        edge are connected to the nearest point on that edge.
        Should be comparable to max_distance.

    Returns
    -------
    graph_mask : np.ndarray
        Binary mask containing:
            - original cells
            - neighbour connections
            - edge connections (if window_border_distance is set)
    """

    labelled, n_cells = label(seed_mask)

    if n_cells == 0:
        return np.zeros_like(seed_mask, dtype=bool)

    h, w = seed_mask.shape

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
    connections_made = 0

    for i, pt in enumerate(centroids):

        # --- Connect to k nearest cell neighbours ---
        distances, neighbours = tree.query(
            pt,
            k=min(k + 1, len(centroids))
        )
        connections_made += len(neighbours) - 1

        if type(neighbours) is int:
            continue

        for j, neighbour_idx in enumerate(neighbours[1:]):

            if distances[j] > max_distance:
                continue

            pt1 = centroids[i]
            pt2 = centroids[neighbour_idx]

            rr, cc = line(
                int(pt1[0]), int(pt1[1]),
                int(pt2[0]), int(pt2[1])
            )

            graph_mask[rr, cc] = True

        # --- Connect to window border if close enough ---
        if window_border_distance is not None:

            y, x = pt[0], pt[1]

            # Distance to each of the 4 edges
            dist_to_edges = {
                'top':    y,
                'bottom': h - 1 - y,
                'left':   x,
                'right':  w - 1 - x
            }

            for edge, dist in dist_to_edges.items():

                if dist <= window_border_distance:

                    # Nearest point on this edge to the centroid
                    if edge == 'top':
                        border_pt = (0,          int(x))
                    elif edge == 'bottom':
                        border_pt = (h - 1,      int(x))
                    elif edge == 'left':
                        border_pt = (int(y),     0)
                    elif edge == 'right':
                        border_pt = (int(y),     w - 1)

                    rr, cc = line(
                        int(y), int(x),
                        border_pt[0], border_pt[1]
                    )

                    # Clip to valid array bounds just in case
                    rr = np.clip(rr, 0, h - 1)
                    cc = np.clip(cc, 0, w - 1)

                    graph_mask[rr, cc] = True

    return graph_mask, connections_made/len(centroids)

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
    network_dilation_radius: float = 0.5,

    # For sampling - don't sample largely empty windows
    min_tissue_area_frac: float = 0.4, # Skips if region is too empty, useful when sampling.

    # object filtering
    min_hole_area: float = 1000, # Don't count holes below this number of pixels towards the total count
    max_hole_area: float = 1E5,
    
    min_cell_size: float = 10, # Distance to dilate background by. Min surviving cell size is essentially 2x this
    min_cell_radius: float = 3,

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
    # Normalising all length parameters to the resolution, from um to px
    cell_dilation_radius = cell_dilation_radius * resolution
    network_dilation_radius = network_dilation_radius * resolution
    min_cell_radius = min_cell_radius * resolution

    dense_polygon_area = 400

    out_df = []
    blank_output = pd.DataFrame([{
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
        }])
    
    base_name = name.split('_bin')[0]
    if not os.path.exists(f'{image_folder}/{base_name}'):
        os.mkdir(f'{image_folder}/{base_name}')

    if mode == 'windows':
        tissue_mask = io.imread(f"final tif files/{name.split('/')[-1].split('_sample')[0]}_roi.tif")
        tissue_mask = tissue_mask[yrange[0]:yrange[1]].T[xrange[0]:xrange[1]].T.astype(bool) # This has to be in pixels
        fig, ax = plt.subplots()
        ax.imshow(~tissue_mask, cmap='binary')
        ax.set_title(f'{name} annotated tissue mask')
        ax.set_xlabel('X / um')
        ax.set_ylabel('Y / um')

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.2f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.2f}'))

        plt.savefig(f'{image_folder}/{base_name}/' + name + f'_tissue_mask_segment.png', bbox_inches='tight')
        plt.close(fig)
        if (~tissue_mask).sum()/np.prod(tissue_mask.shape) < min_tissue_area_frac:
            print(f'Skipped as region <{round(100*min_tissue_area_frac)}% occupied by tissue')
            return blank_output
    else:
        tissue_mask = io.imread(f"final tif files/{name.split('/')[-1][:-4]}_roi.tif").astype(bool)
    print('.', end='', flush=True)
    # ========================================================
    # PIXEL SCALING
    # ========================================================

    # dx = (xrange[1] - xrange[0]) / a.shape[1]
    # dy = (yrange[1] - yrange[0]) / a.shape[0]

    # pixel_area = dx * dy

    # ========================================================
    # GFP-POSITIVE PIXELS
    # ========================================================

    a = ~(a.astype(bool))
    if 1.0*(a.sum()) < (~a).sum(): # Consider boosting background weighting a little
        a = ~a # Some files appear to be stored with GFP signal as >1 (255) and some as 0, annoyingly.

    coords = np.column_stack(np.where(a))

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
        a.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0
    )

    seed_mask = (
        a &
        (local_counts >= min_seed_pixels)
    )

    fig, ax = plt.subplots()
    ax.imshow(seed_mask, cmap='binary')
    ax.set_title(f'{name} gfp mask')
    ax.set_xlabel('X / um')
    ax.set_ylabel('Y / um')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.2f}'))

    plt.savefig(f'{image_folder}/{base_name}/' + name + f'_seed_mask_{locale_pixel_radius}.png', bbox_inches='tight')
    plt.close(fig)
    print('.', end='', flush=True)

    if (~seed_mask).sum() < 1E4: # Some sparse images still getting this far annoyingly
        return blank_output

    # ========================================================
    # REMOVE TINY CELL FRAGMENTS
    # ========================================================

    seed_mask = binary_dilation(
        seed_mask,
        disk(min_cell_radius//2)
    ) # trying this instead to expand non-GFP region a little

    fig, ax = plt.subplots()
    ax.imshow(seed_mask, cmap='binary')
    ax.set_title(f'{name} filtered cell mask')
    ax.set_xlabel('X / um')
    ax.set_ylabel('Y / um')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.2f}'))

    plt.savefig(f'{image_folder}/{base_name}/' + name + f'_seed_mask_noiseless_{round(min_cell_radius/resolution,1)}.png', bbox_inches='tight')
    plt.close(fig)
    print('.', end='', flush=True)

    # ========================================================
    # Expand remaining regions to be cell-sized, in case they were previously fragmented
    # ========================================================
    
    seed_mask = ~(binary_dilation(
        ~seed_mask,
        disk(cell_dilation_radius)
    )) # merge remaining fragments of cells

    fig, ax = plt.subplots()
    ax.imshow(seed_mask, cmap='binary')
    ax.set_title(f'{name} re-dilated cells')
    ax.set_xlabel('X / um')
    ax.set_ylabel('Y / um')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.2f}'))

    plt.savefig(f'{image_folder}/{base_name}/' + name + f'_seed_mask_expanded_cells_{round(cell_dilation_radius/resolution,1)}.png', bbox_inches='tight')
    plt.close(fig)
    print('.', end='', flush=True)

    # ========================================================
    # CELL STATISTICS AFTER EXPANDING CELLS
    # ========================================================

    labelled_cells, _ = label(~seed_mask)

    cell_sizes = np.bincount(
        labelled_cells.ravel()
    )[1:]

    valid_cells = cell_sizes >= min_cell_size

    n_cells = np.sum(valid_cells)
    if int(n_cells) < 50:
        print(' Insufficient cells found.')
        return blank_output
    
    # ========================================================
    # Get area of effective tissue
    # ========================================================

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
    # Draw lines between cells and their 5 nearest neighbours
    # ========================================================
    network_masks = [np.array(np.nan), np.nan, np.nan, np.nan, np.nan]
    for i, max_dist in enumerate(np.array([10, 30, 50, 100, 150])*resolution):
        network_masks[i], avg_connections = connect_cells_to_neighbours(
            ~seed_mask,
            k=6, # In a close-packed 2D setting, this covers the nearest neighbours
            max_distance=max_dist*resolution, # Conversion from 150um to pixels
            window_border_distance=None
        )
        # Remove connections which extend across holes in the tissue marked on the roi file
        network_masks[i] = np.logical_and(network_masks[i], ~tissue_mask)
     
        network_masks[i] = ~(binary_dilation(
                network_masks[i],
                disk(network_dilation_radius)
        ))

        fig, ax = plt.subplots()
        ax.imshow(network_masks[i], cmap='binary')
        ax.set_title(f'{name} cell network')
        ax.set_xlabel('X / um')
        ax.set_ylabel('Y / um')

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.2f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.2f}'))

        plt.savefig(f'{image_folder}/{base_name}/' + name + f'_network_mask_{max_dist}_{round(network_dilation_radius,1)}.png', bbox_inches='tight')
        plt.close(fig)
    
        print('.', end='', flush=True)

        # mark all polygons
        labelled_holes, n_components = label(network_masks[i])

        # ========================================================
        # COLOUR-CODED HOLE VISUALISATION
        # ========================================================

        colour_map = np.zeros(labelled_holes.shape, dtype=np.uint8)

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

            if touches_edge:
                colour_map[component] = 0                      # grey - edge-touching
            elif n_pixels < dense_polygon_area * resolution**2:
                colour_map[component] = 1                      # blue - usual spacing in dense clusters
            elif n_pixels < min_hole_area * resolution**2 and n_pixels > dense_polygon_area * resolution**2:
                colour_map[component] = 2
            elif n_pixels > max_hole_area * resolution**2:
                colour_map[component] = 4                      # red - too large, probably a different tissue e.g. trabecular bone or blood vessel
            else:
                colour_map[component] = 3                      # green - a genuine gap in the target cells where they should be under equal distribution assumptions
                retained_holes[component] = current_label
                current_label += 1
                holes += 1
                hole_area_pixels += n_pixels

        # --- Colourmap: black, green, blue, red, grey ---
        cmap = colors.ListedColormap(
            ['lightgrey', 'steelblue', 'lightblue', 'limegreen', 'tomato']
        )
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        im = ax.imshow(colour_map, cmap=cmap, norm=norm)

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels([
            'Background', #1
            'Typical dense patch', #2
            'Typical semi-dense patch', #3
            'Significant CXCL12 gap', #4
            'Other tissues', #5
        ])

        ax.set_title(f'{name} hole classification')
        ax.set_xlabel('X / um')
        ax.set_ylabel('Y / um')

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.2f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.2f}'))

        plt.savefig(
            f'{image_folder}/{base_name}/' + name + f'_labelled_holes_{max_dist}_{min_hole_area}_{dense_polygon_area}.png',
            bbox_inches='tight'
        )
        plt.close(fig)
        print('.', end='', flush=True)

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

            # cv_hole_area = (
            #     std_hole_area / mean_hole_area
            #     if mean_hole_area > 0
            #     else np.nan
            # )


        else:

            mean_hole_area = 0
            largest_hole_area = 0
            std_hole_area = 0
            # cv_hole_area = np.nan

        out_df.append({
            "name": f"{name}_md={max_dist}",

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
            # "cv_hole_area": cv_hole_area,
            "average_connections_per_cell": avg_connections,

            # tissue statistics
            "filled_area": tissue_area,
            "n_cells": int(n_cells),
            "cell_density": cell_density,
            "mean_cell_area": mean_cell_area,

            # optional debugging output
            # "tissue_mask": tissue_mask,
            # "seed_mask": seed_mask
        })

    print('.')
    return pd.DataFrame(out_df)


def analyse_image(
    a: np.ndarray,
    xrange: Tuple[float, float],
    yrange: Tuple[float, float],
    window_mode: str = 'tiled',
    window_size: int = 1000,
    n_samples: int = 30,
    filename: str = 'image',
    image_folder: str = 'troubleshoot_images',
    resolution: float = 1   # Pixels per micrometre
):
    if type(window_mode) is str:
        return analyse_resampled_windows(
            a,
            xrange,
            yrange,
            window_size=window_size, # Micrometres
            n_samples=n_samples,
            window_mode=window_mode,
            file_name=filename,
            locale_pixel_radius=5,
            min_seed_pixels=20,
            cell_dilation_radius=4.0,
            network_dilation_radius=0.5,
            min_tissue_area_frac=0.4,
            min_hole_area=4000, # Trying to remove normal spacing from counts of holes
            max_hole_area=1E5,
            min_cell_size=50,
            min_cell_radius=2.0,
            image_folder=image_folder,
            resolution=resolution,
        )
    elif window_mode is None:
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
                min_tissue_area_frac=0.1, # Doesn't really matter, keep it kinda low for whole images
                min_hole_area=4000,
                max_hole_area=1E5,
                min_cell_size=10,
                min_cell_radius=2.0,
                image_folder=image_folder,
                resolution=resolution
            )
        ])
    else:
        print('Not a valid window mode.')

def analyse_resampled_windows(
        a: np.ndarray, # 2D binary occupancy mask
        xrange: Tuple[float, float], # width of whole image in um
        yrange: Tuple[float, float], # height of whole image in um
        window_size=100, # Entered in Micrometres, converted to pixels
        n_samples=50,
        window_mode='tiled',
        file_name='Image',
        locale_pixel_radius=4,
        min_seed_pixels=20,
        cell_dilation_radius=2.0,
        network_dilation_radius=0.5,
        min_tissue_area_frac=0.4,
        min_hole_area=4000, #um^2
        max_hole_area=1E5, #um^2
        min_cell_size=10,
        min_cell_radius=3.0,
        image_folder='troubleshoot_images',
        resolution=1.0
    ):
    window_size = int(window_size * resolution) # resolution in pixels per um, so window_size in pixels after here
    results = []
    print(f'Taking up to {n_samples} windows.')
    window_ids = [i for i in range(1, n_samples+1)]
    while len(results) < n_samples:
        windows, warnings = sample_windows_modes(
            a,
            window_size,
            n_samples,
            mode=window_mode,
            window_ids=copy.copy(window_ids),
            overlap_fraction=0.1
        )
        if window_mode == 'tiled':
            n_samples = len(windows)
            window_ids = [i for i in range(1, n_samples+1)]

        for sample_id, y0, y1, x0, x1 in windows:
            print(f'Processing sample {sample_id}', flush=True)

            sub_a = a[y0:y1, x0:x1]

            x_scale = (xrange[1] - xrange[0]) * (window_size / a.shape[1])
            y_scale = (yrange[1] - yrange[0]) * (window_size / a.shape[0])

            out_df = hole_density_from_array_extended(
                sub_a,
                name=f"{file_name[:-4]}_sample_{sample_id}",
                xrange=(x0,x1),
                yrange=(y0,y1),
                locale_pixel_radius=locale_pixel_radius,
                min_seed_pixels=min_seed_pixels,
                cell_dilation_radius=cell_dilation_radius,
                network_dilation_radius=network_dilation_radius,
                min_tissue_area_frac=min_tissue_area_frac,
                min_hole_area=min_hole_area,
                max_hole_area=max_hole_area,
                min_cell_size=min_cell_size,
                min_cell_radius=min_cell_radius,
                image_folder=image_folder,
                resolution=resolution,
                mode='windows'
            )

            if len(out_df) == 0:
                raise ValueError("Empty dataframe returned!")
            elif out_df.isna().sum().sum() > 1:
                continue
            elif out_df['n_cells'].max() < 50:
                continue
            else:
                window_ids.remove(sample_id)
                results.append(out_df)

        if len(results) < n_samples and window_mode == 'tiled':
            n_samples = 1 # To end while loop

    return pd.concat(results)


from PIL import Image
from PIL.TiffTags import TAGS
Image.MAX_IMAGE_PIXELS = 933120000
import warnings

if __name__ == '__main__':
    target_directory = Folder('mask tif files from Clara')
    specific_files = []
    hold_list = []
    for file in target_directory.files:
        if file.type == 'tif':
            if file.name.split('_')[-1] == 'bin.tif' and not file.name in hold_list and (file.name in specific_files if len(specific_files) > 0 else True):
                image = io.imread(file.path)
                with Image.open(file.path) as img:
                    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
                    resolution = float(img.tag_v2.get(282))  # 282 = XResolution tag, returns a clean float
                    if resolution != float(img.tag_v2.get(283)):
                        raise ValueError('Axes not set equal in Tif image.')
                del(img)
                if not 'micron' in meta_dict['ImageDescription'][0]:
                    warnings.warn("Warning - image length not in units of Micrometres.")
                print('Analysing ' + file.name + ':')
                df = analyse_image(
                    a=image,
                    xrange=(0, image.shape[1]),
                    yrange=(0, image.shape[0]),
                    window_mode='tiled', # Process full image or not - full image is usually too big! put None for full image.
                    filename=file.name,
                    image_folder='troubleshoot_images',
                    resolution=resolution,
                    window_size=1400,
                    n_samples=30,
                )
                df.dropna().to_csv(file.path[:-4] + '_resampled_analysed.csv')