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

    if mode == 'windows':
        tissue_mask = io.imread(f"final tif files/{name.split('/')[-1].split('_sample')[0]}_roi.tif")
        tissue_mask = tissue_mask[yrange[0]:yrange[1]].T[xrange[0]:xrange[1]].T.astype(bool) # This has to be in pixels
        if (~tissue_mask).sum()/np.prod(tissue_mask.shape) < min_tissue_area_frac:
            return blank_output
        fig, ax = plt.subplots()
        ax.imshow(~tissue_mask, cmap='binary')
        ax.set_title(f'{name} annotated tissue mask')
        ax.set_xlabel('X / um')
        ax.set_ylabel('Y / um')

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / resolution:.2f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / resolution:.2f}'))

        plt.savefig(f'{image_folder}/' + name + f'_tissue_mask_segment.png', bbox_inches='tight')
        plt.close(fig)
    else:
        tissue_mask = io.imread(f"final tif files/{name.split('/')[-1][:-4]}_roi.tif").astype(bool)

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

    plt.savefig(f'{image_folder}/' + name + f'_seed_mask_noiseless_{round(min_cell_radius,1)}.png', bbox_inches='tight')
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
    # CELL STATISTICS AFTER EXPANDING CELLS
    # ========================================================

    labelled_cells, _ = label(~seed_mask)

    cell_sizes = np.bincount(
        labelled_cells.ravel()
    )[1:]

    valid_cells = cell_sizes >= min_cell_size

    n_cells = np.sum(valid_cells)
    if int(n_cells) < 50:
        return blank_output

    # ========================================================
    # Draw lines between cells and their 5 nearest neighbours
    # ========================================================

    seed_mask = connect_cells_to_neighbours(
        ~seed_mask,
        k=12, # In a close-packed setting, this covers the closest two layers of cells
        max_distance=150*resolution # Conversion from 100um to pixels
    )

    # Remove connections which extend across holes in the tissue marked on the roi file
    seed_mask = np.logical_and(seed_mask, ~tissue_mask)
     

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
    # RECONSTRUCT EFFECTIVE TISSUE
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
    # HOLE DETECTION
    # ========================================================

    labelled_holes, n_components = label(seed_mask)

    # ========================================================
    # COLOUR-CODED HOLE VISUALISATION
    # ========================================================

    # --- Colour categories ---
    # 0 : background / cells      → black
    # 1 : valid hole              → green
    # 2 : rejected, too small     → blue
    # 3 : rejected, too large     → red
    # 4 : rejected, touches edge  → grey

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
            colour_map[component] = 4                      # grey  — edge-touching
        elif n_pixels < min_hole_area * resolution**2:
            colour_map[component] = 2                      # blue  — too small
        elif n_pixels > max_hole_area * resolution**2:
            colour_map[component] = 3                      # red   — too large
        else:
            colour_map[component] = 1                      # green — valid
            retained_holes[component] = current_label
            current_label += 1
            holes += 1
            hole_area_pixels += n_pixels

    # --- Colourmap: black, green, blue, red, grey ---
    cmap = colors.ListedColormap(
        ['black', 'limegreen', 'steelblue', 'tomato', 'lightgrey']
    )
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    im = ax.imshow(colour_map, cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels([
        'Background',
        f'Valid',
        f'Too small (<{min_hole_area} um^2)',
        f'Too large (>{max_hole_area} um^2)',
        'Touches edge'
    ])

    ax.set_title(f'{name} hole classification')
    ax.set_xlabel('X / um')
    ax.set_ylabel('Y / um')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / resolution:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / resolution:.2f}'))

    plt.savefig(
        f'{image_folder}/' + name + f'_labelled_holes_{min_hole_area}.png',
        bbox_inches='tight'
    )
    plt.close(fig)

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

def analyse_image(
    a: np.ndarray,
    xrange: Tuple[float, float],
    yrange: Tuple[float, float],
    split_into_windows: bool = True,
    window_size: int = 1000,
    n_samples: int = 30,
    allow_overlap: bool = True,
    filename: str = 'image',
    image_folder: str = 'troubleshoot_images',
    resolution: float = 1   # Pixels per micrometre
):
    if split_into_windows:
        return analyse_resampled_windows(
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
            min_tissue_area_frac=0.4,
            min_hole_area=4000, # Trying to remove normal spacing from counts of holes
            max_hole_area=1E5,
            min_cell_size=50,
            min_cell_radius=2.0,
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
                min_tissue_area_frac=0.1, # Doesn't really matter, keep it kinda low for whole images
                min_hole_area=4000,
                max_hole_area=1E5,
                min_cell_size=10,
                min_cell_radius=2.0,
                image_folder=image_folder,
                resolution=resolution
            )
        ])

def analyse_resampled_windows(
        a: np.ndarray, # 2D binary occupancy mask
        xrange: Tuple[float, float], # width of whole image in um
        yrange: Tuple[float, float], # height of whole image in um
        window_size=100, # Entered in Micrometres, converted to pixels
        n_samples=50,
        allow_overlap=True,
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
    window_ids = [i for i in range(1, n_samples+1)]
    while len(results) < n_samples:
        windows = sample_windows(
            a,
            window_size,
            n_samples,
            allow_overlap=allow_overlap,
            window_ids=copy.copy(window_ids)
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
                min_tissue_area_frac=min_tissue_area_frac,
                min_hole_area=min_hole_area,
                max_hole_area=max_hole_area,
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
                window_ids.remove(sample_id)
                results.append(out)

    return pd.concat([pd.DataFrame(results)])


from PIL import Image
from PIL.TiffTags import TAGS
Image.MAX_IMAGE_PIXELS = 933120000
import warnings

if __name__ == '__main__':
    target_directory = Folder('mask tif files from Clara')
    hold_list = ['M1hip_bin.tif', 'M2hip_bin.tif', 'M1femur_bin.tif', 'M5hip_left_bin.tif', 'M5hip_right_bin.tif']
    for file in target_directory.files:
        if file.type == 'tif':
            if file.name.split('_')[-1] == 'bin.tif' and not file.name in hold_list:
                image = io.imread(file.path)

                with Image.open(file.path) as img:
                    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
                resolution = float(img.tag_v2.get(282))  # 282 = XResolution tag, returns a clean float
                if resolution != float(img.tag_v2.get(283)):
                    raise ValueError('Axes not set equal in Tif image.')
                if not 'micron' in meta_dict['ImageDescription'][0]:
                    warnings.warn("Warning - image length not in units of Micrometres.")
                print('Analysing ' + file.name)
                df = analyse_image(
                    a=image,
                    xrange=(0, image.shape[1]),
                    yrange=(0, image.shape[0]),
                    split_into_windows=True, # Process full image or not - full image is usually too big!
                    filename=file.name,
                    image_folder='troubleshoot_images',
                    resolution=resolution,
                    window_size=1600,
                    n_samples=8,
                    allow_overlap=True
                )
                df.dropna().to_csv(file.path[:-4] + '_resampled_analysed.csv')