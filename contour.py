import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from shapely.geometry import Point
from shapely.ops import unary_union

from utils import Folder


def contour_density_from_points(
	csv_path,
	tol=5.0,
	mask_radius=5.0,
	grid_res=200,
	out_plot="contour.png",
	out_csv="points_with_density.csv"
):
	# -------------------------
	# Load data
	# -------------------------
	df = pd.read_csv(csv_path)
	points = df[['X', 'Y']].values

	# -------------------------
	# Compute local density per point
	# -------------------------
	tree = cKDTree(points)

	densities = []
	for p in points:
		neighbors = tree.query_ball_point(p, tol)
		densities.append(len(neighbors) - 1)  # exclude self

	heatlabel = f"Local count within {tol} um"
	df[heatlabel] = densities

	# save updated CSV
	df.to_csv(out_csv, index=False)

	# -------------------------
	# Build spatial mask (shape of data)
	# -------------------------
	# union of small buffers gives a smooth shape
	regions = [Point(p).buffer(mask_radius) for p in points]
	region = unary_union(regions)

	# -------------------------
	# Create grid for contouring
	# -------------------------
	xmin, ymin = points.min(axis=0)
	xmax, ymax = points.max(axis=0)

	xi = np.linspace(xmin, xmax, grid_res)
	yi = np.linspace(ymin, ymax, grid_res)
	xi, yi = np.meshgrid(xi, yi)

	# -------------------------
	# Interpolate irregular point data to grid
	# -------------------------
	zi = griddata(
		points,
		df[heatlabel].values,
		(xi, yi),
		method="linear"
	)

	# -------------------------
	# Apply mask (set outside region to NaN)
	# -------------------------
	mask = np.zeros_like(zi, dtype=bool)

	for i in range(xi.shape[0]):
		for j in range(xi.shape[1]):
			if region.contains(Point(xi[i, j], yi[i, j])):
				mask[i, j] = True

	zi_masked = np.where(mask, zi, np.nan)

	# -------------------------
	# Plot contour
	# -------------------------
	plt.figure()

	# filled contours
	contour = plt.contourf(
		xi, yi, zi_masked,
		levels=20,
		cmap="turbo",#"RdBu_r"
	)
	plt.gca().invert_yaxis()

	plt.colorbar(contour, label=heatlabel)	

	plt.title("Density contour map")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.gca().set_aspect("equal")

	# white background for NaNs
	plt.gca().set_facecolor("white")

	plt.savefig(out_plot, dpi=300, bbox_inches="tight")
	plt.close()

def scatter(csv_path, out_plot):
	df = pd.read_csv(csv_path)
	points = df[['X', 'Y']].values
	plt.figure()
	plt.scatter(points[:, 0], points[:, 1], s=3)
	plt.gca().set_aspect("equal")
	plt.gca().invert_yaxis()
	plt.savefig(out_plot, dpi=300, bbox_inches="tight")
	plt.close()

if __name__ == '__main__':
	target_directory = Folder('data')
	for file in Folder('data').files:
		if ('analysed' in file.name and not 'density' in file.name) and ('Slide' in file.name and 'duplicates' in file.name):
			df = file.read()
			contour_density_from_points(
				file.path,
				tol=30,
				mask_radius=70,
				grid_res=200,
				out_plot='images for album'+file.path[4:-4]+'_contour.png',
				out_csv=file.path
			)
			scatter(
				csv_path=file.path,
				out_plot='images'+file.path[4:-4]+'_scatter.png'
			)