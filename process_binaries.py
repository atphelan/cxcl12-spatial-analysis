import numpy as np
import pandas as pd
from skimage import io, measure, morphology
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
import os


def process_image(image_path, min_size=50):
    # Load image
    image = io.imread(image_path)

    # If multi-channel, assume GFP is one channel (take first if unsure)
    if image.ndim == 3:
        image = image[..., 0]

    # Threshold to create binary mask
    thresh = threshold_otsu(image)
    binary = image > thresh

    # Remove small artifacts
    binary = morphology.remove_small_objects(binary, min_size=min_size)

    # Remove objects touching border (optional but often useful)
    binary = clear_border(binary)

    # Label connected components
    labeled = measure.label(binary)

    # Measure properties
    props = measure.regionprops(labeled)

    results = []

    for prop in props:
        # Skip degenerate objects
        if prop.major_axis_length == 0:
            continue

        aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else np.nan

        results.append({
            "image": os.path.basename(image_path),
            "cell_id": prop.label,
            "area_pixels": prop.area,
            "major_axis_length": prop.major_axis_length,
            "minor_axis_length": prop.minor_axis_length,
            "aspect_ratio": aspect_ratio
        })

    return results


def process_folder(folder_path, output_csv="cell_measurements.csv"):
    all_results = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(".tif") or file.lower().endswith(".tiff"):
            image_path = os.path.join(folder_path, file)
            print(f"Processing {file}...")
            results = process_image(image_path)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

    return df


def plot_distributions(df):
    import matplotlib.pyplot as plt

    # Aspect ratio distribution
    plt.figure()
    df["aspect_ratio"].dropna().hist(bins=30)
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Frequency")
    plt.title("Aspect Ratio Distribution")
    plt.show()

    # Area distribution
    plt.figure()
    df["area_pixels"].hist(bins=30)
    plt.xlabel("Cell Area (pixels)")
    plt.ylabel("Frequency")
    plt.title("Cell Size Distribution")
    plt.show()


if __name__ == "__main__":
    df = process_folder('mask tif files from Clara', output_csv='mask tif files from Clara/cell_measurements.csv')
    # Optional plotting
    plot_distributions(df)
