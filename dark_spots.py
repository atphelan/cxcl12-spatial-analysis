import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; import matplotlib.cm as cm
import copy
from scipy.spatial import cKDTree
from scipy.ndimage import label
from skimage.segmentation import flood, flood_fill

from utils import Folder
from typing import Tuple

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
    dx, dy = xrange/a.shape[0], yrange/a.shape[1]  # Set scale based on input data in future

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


    

if __name__ == '__main__':
    target_directory = Folder('data')
    out_data = get_porosities(target_directory, dilation_radius=70.0, tissue_tol = 60) # Units of micrometres
    out_data.to_csv('hole_densities.csv')