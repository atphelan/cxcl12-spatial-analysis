import os
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt, cm, ticker, colors
from scipy.spatial import cKDTree
from scipy.ndimage import label, convolve
from scipy.spatial.distance import pdist, squareform
from skimage import io
from skimage.draw import line
from skimage.morphology import binary_dilation, disk
import copy

from utils import Folder
from typing import Tuple

black_to_green = colors.LinearSegmentedColormap.from_list(
    name="black_to_green",
    colors=["black", "lime", "lime", "lime", "lime"]  # "lime" = bright green (#00FF00), Saturated a fair bit
)

black_to_blue = colors.LinearSegmentedColormap.from_list(
    name="black_to_blue",
    colors=["black", "blue", "blue", "blue", "blue"]
)

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

class _TileAccepted(Exception):
    """Break all loops — excellent tile found."""
    pass

class _SkipRegion(Exception):
    """Break out of size loops — current origin is unpromising."""
    pass

def _integral_image(arr: np.ndarray) -> np.ndarray:
    """
    Summed area table with a zero-padded border, so that:
        integral[y, x] = sum of arr[0:y, 0:x]
    This means integral[0, :] == 0 and integral[:, 0] == 0,
    and a rectangle sum arr[y0:y1, x0:x1] is always:
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    with no edge case handling needed.
    """
    padded = np.zeros(
        (arr.shape[0] + 1, arr.shape[1] + 1),
        dtype=np.float32
    )
    padded[1:, 1:] = np.cumsum(np.cumsum(arr.astype(np.float32), axis=0), axis=1)
    return padded


def _rect_sum(integral: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> float:
    """
    Query a padded integral image for the sum of arr[y0:y1, x0:x1].
    No edge case handling needed due to the zero padding.
    """
    return float(
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )

def sample_windows_adaptive(
    a: np.ndarray,
    tissue_mask: np.ndarray,
    base_size: int,
    size_tolerance: float = 0.2,
    overlap_fraction: float = 0.05,
    min_tissue_fraction: float = 0.4,
    search_stride: int = 50,
    seed: int = None
) -> Tuple[list, dict]:
    """
    Adaptively tile a tissue mask with rectangles, greedily placing
    tiles in order of uncovered tissue density.

    Each tile size is chosen independently in y and x within
    [base_size * (1 - tolerance), base_size * (1 + tolerance)],
    maximising the ratio of uncovered tissue pixels to tile area.

    Parameters
    ----------
    a : np.ndarray
        Full binary image array (used for shape reference).

    tissue_mask : np.ndarray
        2D boolean array, True = valid tissue.
        Must be the same shape as a.

    base_size : int
        Base tile size in pixels.

    size_tolerance : float
        Fractional size tolerance. Tile height and width are each
        independently chosen from:
            [base_size * (1 - tolerance), base_size * (1 + tolerance)]

    overlap_fraction : float
        Maximum allowed overlap between adjacent tiles as a fraction
        of base_size, on each edge.

    min_tissue_fraction : float
        Minimum fraction of tile area that must be tissue for a tile
        to be accepted.

    search_stride : int
        Step size in pixels for the coarse candidate origin grid.
        Smaller = more thorough but slower.

    seed : int or None
        Random seed for tiebreaking between equal-density candidates.

    Returns
    -------
    windows : list of tuples
        Each tuple: (sample_id, y0, y1, x0, x1)

    report : dict
        Placement summary including coverage statistics and warnings.
    """

    rng = np.random.default_rng(seed)
    if tissue_mask.sum() > np.prod(tissue_mask.shape)/2:
        tissue_mask = ~tissue_mask

    if a.shape != tissue_mask.shape:
        raise ValueError(
            f"a.shape {a.shape} and tissue_mask.shape "
            f"{tissue_mask.shape} must match."
        )

    h, w = a.shape

    # --------------------------------------------------------
    # Size ranges
    # --------------------------------------------------------

    s_min = max(1, int(np.floor(base_size * (1 - size_tolerance))))
    s_max = int(np.ceil(base_size * (1 + size_tolerance)))
    max_overlap_px = int(np.floor(overlap_fraction * base_size))

    # Candidate sizes to try — sample a modest number to keep
    # the search tractable. More sizes = better fit, slower search.
    n_size_candidates = 8
    size_candidates = np.unique(np.linspace(s_min, s_max, n_size_candidates, dtype=int))

    # --------------------------------------------------------
    # Coarse candidate origin grid
    # --------------------------------------------------------

    y_origins = np.arange(0, h - s_min + 1, search_stride)
    x_origins = np.arange(0, w - s_min + 1, search_stride)

    if len(y_origins) == 0 or len(x_origins) == 0:
        report = {
            'mode':                  'adaptive',
            'placed':                0,
            'tissue_coverage':       0.0,
            'mean_tile_height':      None,
            'mean_tile_width':       None,
            'total_tissue_px':       int(tissue_mask.sum()),
            'covered_tissue_px':     0,
            'warning': (
                f"base_size={base_size} is larger than the image "
                f"({h}x{w}) — no tiles placed."
            )
        }
        print(_format_report(report))
        return [], report

    # --------------------------------------------------------
    # Uncovered tissue tracking
    # --------------------------------------------------------

    # Float mask so we can accumulate partial coverage tracking
    uncovered = (tissue_mask).astype(np.float32).copy()
    total_tissue_px = int(tissue_mask.sum())

    if total_tissue_px == 0:
        report = {
            'mode':              'adaptive',
            'placed':            0,
            'tissue_coverage':   0.0,
            'mean_tile_height':  None,
            'mean_tile_width':   None,
            'total_tissue_px':   0,
            'covered_tissue_px': 0,
            'warning':           "tissue_mask contains no True pixels."
        }
        print(_format_report(report))
        return [], report

    # --------------------------------------------------------
    # Placed tile tracking (for overlap enforcement)
    # --------------------------------------------------------

    placed_tiles = []   # list of (y0, y1, x0, x1)
    windows = []
    sample_id = 0

    # --------------------------------------------------------
    # Greedy placement loop
    # --------------------------------------------------------
    covered_mask = np.zeros_like(tissue_mask, dtype=np.float32)
    # max_jitter = max(1, search_stride // 4)
    # jitter_y_table = rng.integers(0, max_jitter, size=(1000, len(y_origins)))
    # jitter_x_table = rng.integers(0, max_jitter, size=(1000, len(x_origins)))
    while True:

        integral_tissue    = _integral_image(tissue_mask)
        integral_uncovered = _integral_image(uncovered)

        best_score = -1.0
        best_tile  = None

        # Shuffle origins slightly each pass for tiebreaking
        jitter_y = rng.integers(0, max(1, search_stride // 4), size=len(y_origins))
        jitter_x = rng.integers(0, max(1, search_stride // 4), size=len(x_origins))

        try:
            for iy, y0_base in enumerate(y_origins):
                for ix, x0_base in enumerate(x_origins):

                    y0 = int(np.clip(y0_base + jitter_y[iy], 0, h - s_min))
                    x0 = int(np.clip(x0_base + jitter_x[ix], 0, w - s_min))

                    # Quick check — is there any uncovered tissue near here?
                    local_patch = uncovered[
                        y0 : min(y0 + s_max, h),
                        x0 : min(x0 + s_max, w)
                    ]
                    if local_patch.sum()/(np.prod(local_patch.shape)) <= 0.05:
                        continue

                    try:
                        # Try all combinations of height and width candidates
                        for sh in size_candidates:
                            for sw in size_candidates:
                                
                                y1 = y0 + sh
                                x1 = x0 + sw

                                # Must fit within image
                                if y1 > h or x1 > w:
                                    continue

                                integral_covered = _integral_image(covered_mask)

                                tile_area         = sh * sw
                                tissue_in_tile    = _rect_sum(integral_tissue,    y0, x0, y1, x1)
                                uncovered_in_tile = _rect_sum(integral_uncovered, y0, x0, y1, x1)
                                integral_tissue = _integral_image(tissue_mask.astype(np.float32))
                                print(f"tissue_mask sum: {tissue_mask.sum()}")
                                print(f"integral_tissue max: {integral_tissue.max()}")  # should match tissue_mask.sum()
                                print(f"spot check: {_rect_sum(integral_tissue, 0, 0, h, w)}")  # should equal tissue_mask.sum()
                                filled_frac = tissue_in_tile / tile_area
                                
                                # Sanity check on first origin
                                y0_test = int(np.random.choice(y_origins))
                                x0_test = int(np.random.choice(x_origins))
                                y1_test = min(y0_test + base_size, h)
                                x1_test = min(x0_test + base_size, w)

                                print(f"Test tile: y={y0_test}:{y1_test}, x={x0_test}:{x1_test}")
                                print(f"  tissue_in_tile  : {_rect_sum(integral_tissue,    y0_test, x0_test, y1_test, x1_test)}")
                                print(f"  uncovered_in_tile: {_rect_sum(integral_uncovered, y0_test, x0_test, y1_test, x1_test)}")
                                print(f"  tile_area       : {(y1_test - y0_test) * (x1_test - x0_test)}")
                                print(f"  tissue fraction : {_rect_sum(integral_tissue, y0_test, x0_test, y1_test, x1_test) / ((y1_test-y0_test)*(x1_test-x0_test)):.4f}")

                                print(f"tissue_mask fraction: {tissue_mask.mean():.4f}")
                                print(f"y_origins range: {y_origins[0]} to {y_origins[-1]}, n={len(y_origins)}")
                                print(f"x_origins range: {x_origins[0]} to {x_origins[-1]}, n={len(x_origins)}")
                                
                                if filled_frac < 0.05:
                                    raise _SkipRegion
                                elif filled_frac < min_tissue_fraction:
                                    continue

                                score = uncovered_in_tile / tile_area

                                already_covered  = _rect_sum(integral_covered, y0, x0, y1, x1)
                                tile_area        = sh * sw
                                overlap_fraction_actual = already_covered / tile_area

                                if overlap_fraction_actual > overlap_fraction:   # e.g. 0.05
                                    continue

                                if score > best_score:
                                    best_score = score
                                    best_tile  = (y0, y1, x0, x1)
                                    if score > 0.8:
                                        raise _TileAccepted
                    
                    except _SkipRegion:
                        pass

        except _TileAccepted:
            pass

        # No valid tile found — we're done
        if best_tile is None:
            break

        y0, y1, x0, x1 = best_tile
        covered_mask[y0:y1, x0:x1] = 1.0

        # Mark tile region as covered
        uncovered[y0:y1, x0:x1] = 0.0

        placed_tiles.append(best_tile)
        windows.append((sample_id, y0, y1, x0, x1))
        sample_id += 1

        # Stop if tissue is well covered
        if uncovered.sum()/total_tissue_px <= 0.05:
            break

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    covered_tissue_px = int(sum(
        tissue_mask[y0:y1, x0:x1].sum()
        for y0, y1, x0, x1 in placed_tiles
    )) if placed_tiles else 0

    # Recompute properly — union of all tile masks
    coverage_mask = np.zeros_like(tissue_mask, dtype=bool)
    for y0, y1, x0, x1 in placed_tiles:
        coverage_mask[y0:y1, x0:x1] = True
    covered_tissue_px = int((coverage_mask & tissue_mask).sum())

    heights = [y1 - y0 for _, y0, y1, _, _ in [
        (s, *t) for s, *t in windows
    ]]
    widths  = [x1 - x0 for _, _, _, x0, x1 in [
        (s, *t) for s, *t in windows
    ]]

    tissue_coverage = covered_tissue_px / total_tissue_px

    warnings = []
    if tissue_coverage < 0.8:
        warnings.append(
            f"Only {tissue_coverage*100:.1f}% of tissue covered. "
            f"Consider reducing search_stride, size_tolerance, or "
            f"min_tissue_fraction."
        )
    if len(windows) == 0:
        warnings.append(
            "No tiles placed. Check that tissue_mask has sufficient "
            "coverage and base_size is appropriate for the image size."
        )

    report = {
        'mode':              'adaptive',
        'placed':            len(windows),
        'tissue_coverage':   round(tissue_coverage, 4),
        'mean_tile_height':  round(np.mean(heights), 1) if heights else None,
        'mean_tile_width':   round(np.mean(widths),  1) if widths  else None,
        'total_tissue_px':   total_tissue_px,
        'covered_tissue_px': covered_tissue_px,
        'warning':           '; '.join(warnings) if warnings else None
    }

    print(_format_report(report))
    return windows, report


def _format_report(report: dict) -> str:
    """
    Format a window placement report dict for printing.
    Shared between sample_windows and sample_windows_adaptive.
    """
    lines = [
        "\n--- sample_windows report ---",
        f"  Mode              : {report['mode']}",
        f"  Placed            : {report['placed']}",
    ]

    if 'requested' in report and report['requested'] is not None:
        lines.insert(2, f"  Requested         : {report['requested']}")

    if report.get('tissue_coverage') is not None:
        lines.append(
            f"  Tissue coverage   : {report['tissue_coverage']*100:.1f}%"
        )
    if report.get('coverage_fraction') is not None:
        lines.append(
            f"  Image coverage    : {report['coverage_fraction']*100:.1f}%"
        )
    if report.get('mean_tile_height') is not None:
        lines.append(
            f"  Mean tile size    : "
            f"{report['mean_tile_height']} x {report['mean_tile_width']} px"
        )
    if report.get('attempts_used') is not None:
        lines.append(
            f"  Attempts used     : "
            f"{report['attempts_used']} / {report['max_attempts']}"
        )
    if report.get('total_tissue_px') is not None:
        lines.append(
            f"  Total tissue px   : {report['total_tissue_px']:,}"
        )
    if report.get('covered_tissue_px') is not None:
        lines.append(
            f"  Covered tissue px : {report['covered_tissue_px']:,}"
        )
    if report.get('warning'):
        lines.append(f"WARNING         : {report['warning']}")

    lines.append("-----------------------------\n")
    return '\n'.join(lines)


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
    mode: str = 'whole_image',

    tissue_mask: np.ndarray = np.array(np.nan),
    gfp: np.ndarray = np.array(np.nan),
    dapi: np.ndarray = np.array(np.nan)
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

    dense_polygon_max_area = 400 # um^2
    dense_polygon_max_length = 15 # um

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
            # "cv_hole_area": np.nan,
            "n_cells": 0,
            "cell_density": np.nan,
            "mean_cell_area": np.nan
        }])
    
    base_name = name.split('_bin')[0]
    if not os.path.exists(f'{image_folder}/{base_name}'):
        os.mkdir(f'{image_folder}/{base_name}')

    # Crop or read in the mask file
    if mode == 'windows':
        if np.isnan(tissue_mask).sum() > 0:
            tissue_mask = io.imread(f"tissue masks/{name.split('/')[-1].split('_sample')[0].replace('_bin', '_mask')}.tif")
        tissue_mask = tissue_mask[yrange[0]:yrange[1]].T[xrange[0]:xrange[1]].T.astype(bool) # This has to be in pixels
        fig, ax = plt.subplots()
        ax.imshow(~tissue_mask, cmap='binary')
        ax.set_title(f'{name} annotated tissue mask')
        ax.set_xlabel('X / um')
        ax.set_ylabel('Y / um')

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.0f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.0f}'))

        plt.savefig(f'{image_folder}/{base_name}/' + name + f'_tissue_mask_window_{xrange[1]}.png', bbox_inches='tight')
        plt.close(fig)
        if (~tissue_mask).sum()/np.prod(tissue_mask.shape) < min_tissue_area_frac:
            print(f'Skipped as region <{round(100*min_tissue_area_frac)}% occupied by tissue')
            return blank_output
    else:
        tissue_mask = io.imread(f"tissue masks/{name.split('/')[-1][:-4].replace('_bin', '_mask')}.tif").astype(bool)
    print('.', end='', flush=True)


    # Read in the original GFP/cxcl12 file
    if mode == 'windows':
        if np.isnan(gfp).sum() > 0:
            gfp = io.imread(f"originals/{name.split('/')[-1].split('_sample')[0].replace('_bin', '_cxcl12')}.tif")
        gfp = gfp[yrange[0]:yrange[1]].T[xrange[0]:xrange[1]].T # This has to be in pixels
        fig, ax = plt.subplots()
        ax.imshow(gfp, black_to_green)
        ax.set_title(f'{name} original GFP image')
        ax.set_xlabel('X / um')
        ax.set_ylabel('Y / um')

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.0f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.0f}'))

        plt.savefig(f'{image_folder}/{base_name}/' + name + f'_gfp_window_{xrange[1]}.png', bbox_inches='tight')
        plt.close(fig)
    else:
        gfp = io.imread(f"final tif files/{name.split('/')[-1][:-4].replace('_bin', '_mask')}.tif").astype(bool)
    print('.', end='', flush=True)
    del(gfp)


    # Read in the original dapi file
    try:
        if mode == 'windows':
            if np.isnan(dapi).sum() > 0:
                dapi = io.imread(f"originals/{name.split('/')[-1].split('_sample')[0].replace('_bin', '_dapi')}.tif")
            dapi = dapi[yrange[0]:yrange[1]].T[xrange[0]:xrange[1]].T # This has to be in pixels
            fig, ax = plt.subplots()
            ax.imshow(dapi, black_to_blue)
            ax.set_title(f'{name} original DAPI image')
            ax.set_xlabel('X / um')
            ax.set_ylabel('Y / um')

            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.0f}'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.0f}'))

            plt.savefig(f'{image_folder}/{base_name}/' + name + f'_dapi_window_{xrange[1]}.png', bbox_inches='tight')
            plt.close(fig)
        else:
            dapi = io.imread(f"originals/{name.split('/')[-1][:-4].replace('_bin', '_mask')}.tif").astype(bool)
        print('.', end='', flush=True)
    except:
        print('\nDAPI file not found, proceeding to next step.')
    # del(dapi) # Suppressing this for measuring non-parenchyma later

    # ========================================================
    # PIXEL SCALING
    # ========================================================

    # dx = (xrange[1] - xrange[0]) / a.shape[1]
    # dy = (yrange[1] - yrange[0]) / a.shape[0]

    # pixel_area = dx * dy

    # ========================================================
    # GFP-POSITIVE PIXELS ABOVE THRESHOLD
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
    ax.set_title(f'{name} gfp binary')
    ax.set_xlabel('X / um')
    ax.set_ylabel('Y / um')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.0f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.0f}'))

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

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.0f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.0f}'))

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

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.0f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.0f}'))

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

    tissue_area = np.sum(~tissue_mask) * pixel_area

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
    for i, max_dist in enumerate(np.array([50])*resolution):
        network_masks[i], avg_connections = connect_cells_to_neighbours(
            ~seed_mask,
            k=6, # In a close-packed 2D setting, this covers the nearest neighbours
            max_distance=max_dist*resolution, # Conversion from um to pixels
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

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.0f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.0f}'))

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
        n_high_dens = 0
        n_low_dens = 0
        n_discounted_regions = 0
        n_bg_regions = 0
        area_high_dens = 0
        area_low_dens = 0
        hole_area_pixels = 0
        area_discounted = 0
        bg_area = 0

        current_label = 1

        for i in range(1, n_components + 1):

            component = (labelled_holes == i)
            n_pixels = component.sum()
            if n_pixels < 10: # Discard tiny regions from counting
                colour_map[component] = 0
                continue

            touches_edge = (
                component[0, :].any() or
                component[-1, :].any() or
                component[:, 0].any() or
                component[:, -1].any()
            )

            locs = np.where(component)
            approx_l = np.sqrt((locs[0][-1]-locs[0][0])**2 + (locs[1][-1]-locs[1][0])**2) # Assuming a triangle with two similarly long sides. Will only matter for small shapes, which will be not too far from this

            if touches_edge:
                colour_map[component] = 0                      # grey - edge-touching
                n_bg_regions += 1
                bg_area += n_pixels
            elif n_pixels < dense_polygon_max_area * resolution**2:
                colour_map[component] = 1                      # blue - usual spacing in dense clusters
                n_high_dens += 1
                area_high_dens += n_pixels
            elif n_pixels < min_hole_area * resolution**2 and n_pixels > (dense_polygon_max_area * resolution**2 or approx_l > dense_polygon_max_length * resolution):
                colour_map[component] = 2                      # light blue - sub-dense regions
                n_low_dens += 1
                area_low_dens += n_pixels
            elif n_pixels > max_hole_area * resolution**2:
                colour_map[component] = 4                      # red - too large, probably a different tissue e.g. trabecular bone or blood vessel
                n_discounted_regions += 1
                area_discounted += n_pixels
            else:
                if dapi[component].mean() > 3000: # Parenchymal tissue but a CXCL12-GFP gap still
                    colour_map[component] = 3                      # green - a genuine gap in the target cells where they should be under equal distribution assumptions
                    retained_holes[component] = current_label
                    current_label += 1
                    holes += 1
                    hole_area_pixels += n_pixels
                else: # Non-parenchymal (maybe - see femur 3 for examples of large regions of low DAPI but high CXCL12-GFP)
                    colour_map[component] = 4
                    n_discounted_regions += 1
                    area_discounted += n_pixels


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

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{xrange[0] + x / resolution:.0f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{yrange[0] + y / resolution:.0f}'))

        plt.savefig(
            f'{image_folder}/{base_name}/' + name + f'_labelled_holes_{max_dist}_{min_hole_area}_{dense_polygon_max_area}.png',
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

        tissue_area = tissue_area - area_discounted * pixel_area # Remove sum of red region pixels

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

            # cxcl12 stromal cell statistics
            "filled_area": tissue_area,
            "n_cells": int(n_cells),
            "cell_density": cell_density,
            "mean_cell_area": mean_cell_area,

            # tissue statistics
            "high_density_polygons": n_high_dens,
            "low_density_polygons": n_low_dens,
            "high_density_area": area_high_dens / resolution**2,
            "low_density_area": area_low_dens / resolution**2,
            "ratio_high_low": area_high_dens/area_low_dens,
            "n_discounted_regions": n_discounted_regions,
            "area_discounted": area_discounted / resolution**2,
            "background_regions": n_bg_regions,
            "background_area": bg_area / resolution**2,

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
            cell_dilation_radius=2.0,
            network_dilation_radius=0.5,
            min_tissue_area_frac=0.4,
            min_hole_area=3E4, # Trying to remove normal spacing from counts of holes
            max_hole_area=3E5, # Try to remove any tissue tears or interrupting other tissues e.g. bone from the analysis
            min_cell_size=50,
            min_cell_radius=1.5,
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
    mask = io.imread('tissue masks/' + file_name.replace('_bin', '_mask'))
    gfp = io.imread('originals/' + file_name.replace('_bin', '_cxcl12'))
    try:
        dapi = io.imread('originals/' + file_name.replace('_bin', '_dapi'))
    except:
        dapi = np.array(np.nan)
    while len(results) < n_samples:
        if window_mode == 'tiled':
            windows, report = sample_windows_modes(
                a,
                window_size,
                n_samples,
                mode=window_mode,
                window_ids=copy.copy(window_ids),
                overlap_fraction=0.1
            )
        elif window_mode == 'greedy':
            windows, report = sample_windows_adaptive(
                a            = image,
                tissue_mask  = mask.astype(bool),
                base_size    = 1400,
                size_tolerance   = 0.3,
                overlap_fraction = 0.05,
                min_tissue_fraction = 0.3,
                search_stride    = 100,
                seed             = 42
            )
        if window_mode == 'tiled' or 'greedy':
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
                mode='windows',
                tissue_mask=mask,
                gfp=gfp,
                dapi=dapi
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
    target_directory = Folder('binaries')
    tile_size_lookup = {
        'M1femur_bin.tif': 1400,
        'M1hip_bin.tif': 750,
        'M2femur_bin.tif': 1600,
        'M2hip_bin.tif': 700,
        'M3femur_bin.tif': 1650,
        'M3hip_bin.tif': 850,
        'M4femur_bin.tif': 1550,
        'M4hip_bin.tif': 700,
        'M5femur_bin.tif': 1350,
        'M5hip_bin.tif': 800,
    }
    specific_files = tile_size_lookup.keys()
    hold_list = []
    for file in target_directory.files:
        if file.type == 'tif':
            if file.name.split('_')[-1] == 'bin.tif' and not file.name in hold_list and (file.name in specific_files if len(specific_files) > 0 else True):
                print(f'Opening {file.name}...')
                image = io.imread(file.path)
                try:
                    with Image.open(file.path) as img:
                        meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
                        resolution = float(img.tag_v2.get(282))  # 282 = XResolution tag, returns a clean float
                        if resolution != float(img.tag_v2.get(283)):
                            raise ValueError('Axes not set equal in Tif image.')
                    del(img)
                    if not 'micron' in meta_dict['ImageDescription'][0]:
                        warnings.warn("Warning - image length not in units of Micrometres.")
                except: # Get metadata from mask instead of binary if there is an issue with the binary
                    print('Taking metadata from roi instead of binary...')
                    with Image.open(file.path.replace('binaries', 'tissue masks').replace('_bin', '_mask')) as img:
                        meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
                        resolution = float(img.tag_v2.get(282))  # 282 = XResolution tag, returns a clean float
                        if resolution != float(img.tag_v2.get(283)):
                            raise ValueError('Axes not set equal in Tif image.')
                    del(img)
                    if not 'micron' in meta_dict['ImageDescription'][0]:
                        warnings.warn("Warning - image length not in units of Micrometres.")

                print('Analysing ' + file.name + ':')
                window_mode = 'tiled'
                df = analyse_image(
                    a=image,
                    xrange=(0, image.shape[1]),
                    yrange=(0, image.shape[0]),
                    window_mode=window_mode, # Process full image or not - full image is usually too big! put None for full image.
                    filename=file.name,
                    image_folder='troubleshoot_images',
                    resolution=resolution,
                    window_size=tile_size_lookup[file.name],
                    n_samples=50,
                )
                df.dropna().to_csv('data/' + file.name[:-4].replace('bin', 'results') + f'_{window_mode}.csv')