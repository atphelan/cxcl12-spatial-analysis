from utils import Folder
import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

def make_summary_table_10images(target_directory):
    data_dict_list = []
    anova_data = {}
    anova_variables = ['hole_coverage', 'cell_density']
    for variable in anova_variables:
        for file in target_directory.files:
            if (file.type == 'csv' and 'analysed' in file.path) and not 'dilate' in file.name:
                df = pd.read_csv(file.path)
                anova_data.update({'name': file.name.split('_bin')[0], 'hole_coverage': df['hole_coverage'].mean(), 'cell_density': df['cell_density'].mean()})
                data_dict_list.append({'name': file.name.split('_bin')[0]})

        # Build paired arrays, ensuring animal order matches
        paired = [
            (name, anova_data[name], anova_data[name.replace('hip', 'femur')])
            for name in anova_data.keys()
            if 'hip' in name and name.replace('hip', 'femur') in anova_data
        ]

        hip_means   = [np.mean(h) for _, h, _ in paired]
        femur_means = [np.mean(_, f) for _, _, f in paired]  

        # Check normality of differences first (Shapiro-Wilk)
        differences = np.array(hip_means) - np.array(femur_means)
        stat, p_norm = shapiro(differences)
        print(f'Shapiro-Wilk on differences: W={stat:.3f}, p={p_norm:.3f}')

        if p_norm > 0.05:
            # Differences are plausibly normal — paired t-test
            result = ttest_rel(hip_means, femur_means)
            print(f'Paired t-test: t={result.statistic:.3f}, p={result.pvalue:.3f}')
        else:
            # Non-normal — Wilcoxon signed-rank
            result = wilcoxon(hip_means, femur_means)
            print(f'Wilcoxon signed-rank: W={result.statistic:.3f}, p={result.pvalue:.3f}')


def run_paired_statistics(target_directory, metrics=None, min_windows=3):
    """
    Runs paired statistical tests (Wilcoxon signed-rank + paired t-test)
    comparing hip vs femur for each metric, using per-image area-weighted
    means (weighted by filled_area per tile) to avoid bias from unequal
    tile counts across images.

    Parameters
    ----------
    target_directory : Folder
        Directory containing analysed CSV files.

    metrics : list of str or None
        Metrics to test. Defaults to all numeric outcome columns.

    min_windows : int
        Minimum number of valid windows required to include an image.
        Images with fewer windows are excluded with a warning.

    Returns
    -------
    results_df : pd.DataFrame
        Table of test statistics and p-values for each metric.
    """

    if metrics is None:
        metrics = [
            'cell_density', 'cell_density_whole_mask', 'frac_dense_area',
            'frac_sparse_area', 'frac_hole_area', 'ratio_high_low',
            'prop_removed_area', 're-added_area_from_edge', 'sparse_number_density'
        ]

    tile_size_lookup = {
        'M1femur_bin.tif.zip': 1200,
        'M1hip_bin.tif.zip': 600,
        'M1thv_bin.tif.zip': 700,
        'M2femur_bin.tif.zip': 1400,
        'M2hip_bin.tif.zip': 900,
        'M3femur_bin.tif.zip': 1400,
        'M3hip_bin.tif.zip': 900,
        'M4femur_bin.tif.zip': 1258,
        'M4hip_bin.tif.zip': 1000,
        'M5femur_bin.tif.zip': 1300,
        'M5hip_bin.tif.zip': 900,
    }

    # --------------------------------------------------------
    # Load and compute area-weighted means for all valid files
    # --------------------------------------------------------

    image_means = {}  # { image_base_name : { metric : weighted_mean_value } }

    for file in target_directory.files:
        if file.type != 'csv' or not ('save states' in file.name) or '.tif' in file.name:
            continue

        df = pd.read_csv(file.path)
        recalculate = True
        ignore_yellows = [
            'M2femur_results_save states.csv',
            'M5hip_results_save states.csv',
            'M3hip_results_save states.csv'
        ]

        if not ('ratio_reconstructed_mask' in df.columns) or recalculate:
            df['mean_hole_area'] = df['hole_area'] / df['n_holes']
            df = df.drop(columns=[c for c in df.columns if 'Unnamed:' in c])

            if file.name not in ignore_yellows:
                df['total_hd_area'] = df['high_density_area'] + df['high_density_area_anom']
                df['total_ld_area'] = df['low_density_area']  + df['low_density_area_anom']
            else:
                df['total_hd_area'] = df['high_density_area']
                df['total_ld_area'] = df['low_density_area']

            df['reconstructed_tissue_area'] = (
                df['hole_area'] + df['total_hd_area'] + df['total_ld_area']
            )
            df['cell_density_whole_mask'] = (
                df['cell_density'] * df['reconstructed_tissue_area']
                / (df['reconstructed_tissue_area'] + df['area_discounted'] - df['background_area'])
            )
            df['frac_dense_area']      = df['total_hd_area'] / df['reconstructed_tissue_area']
            df['frac_sparse_area']     = df['total_ld_area']  / df['reconstructed_tissue_area']
            df['sparse_number_density']= df['low_density_polygons'] / df['reconstructed_tissue_area']
            df['frac_hole_area']       = df['hole_area'] / df['reconstructed_tissue_area']
            df['prop_removed_area']    = df['area_discounted'] / (df['reconstructed_tissue_area'] + df['area_discounted'])
            df['ratio_reconstructed_mask'] = df['reconstructed_tissue_area'] / df['filled_area']

            df['tile_area'] = tile_size_lookup[file.name.split('_')[0] + '_bin.tif.zip']**2

            df['frac_density_change'] = (df['cell_density']-df['cell_density_whole_mask'])/df['cell_density_whole_mask']

            if file.name in (
                'M5femur_results_save states.csv',
                # 'M2hip_results_save states.csv'
            ):
                # Dark spots in GFP image not well accounted for in whole
                # mask density — use reconstructed-area density directly
                df['cell_density_whole_mask'] = df['cell_density']

            df.reset_index(drop=True).to_csv(file.path)

        df = df[df['ratio_reconstructed_mask'] > 0.18]

        if len(df) < min_windows:
            print(
                f"  [WARN] {file.name}: only {len(df)} windows "
                f"— skipping (min={min_windows})"
            )
            continue

        base_name = file.name.split('_bin')[0]  # e.g. 'M3hip' or 'M3femur'

        # Area-weighted mean across tiles: each tile contributes in
        # proportion to its filled_area, so images with more/larger
        # tiles do not dominate the per-image summary statistic.
        # This matches the weighting used in the mixed-effects model.
        weights = df['filled_area'].values

        # See everything
        ordinal_type = df['name'].dtype
        metrics = [c for c in df.columns if df[c].dtype != ordinal_type]

        image_means[base_name] = {}
        for metric in metrics:
            if metric not in df.columns:
                continue
            vals = df[metric].values
            # Guard against all-NaN or zero-weight edge cases
            valid = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
            if valid.sum() == 0:
                image_means[base_name][metric] = np.nan
            else:
                image_means[base_name][metric] = (
                    np.average(vals[valid], weights=weights[valid])
                )

    if not image_means:
        raise ValueError(
            "No valid files found. Check target directory and file naming."
        )

    # --------------------------------------------------------
    # Build matched hip / femur pairs
    # --------------------------------------------------------

    pairs = []

    for name in image_means:
        if 'hip' not in name:
            continue
        partner = name.replace('hip', 'femur')
        if partner not in image_means:
            print(f"  [WARN] No matching femur found for {name} — skipping.")
            continue
        pairs.append((name, partner))

    if not pairs:
        raise ValueError(
            "No matched hip/femur pairs found. Check file naming."
        )

    # Sort pairs so mouse ordering is consistent everywhere
    pairs = sorted(pairs, key=lambda x: x[0])

    n_pairs = len(pairs)
    print(f"\nUsing {n_pairs} matched hip/femur pairs.")
    for hip, femur in pairs:
        print(f"  {hip}  <-->  {femur}  ")

    # --------------------------------------------------------
    # Run tests for each metric
    # --------------------------------------------------------

    results = []

    for metric in metrics:

        # Extract values in consistent pair order
        hip_means   = []
        femur_means = []
        for hip, femur in pairs:
            h = image_means[hip].get(metric,   np.nan)
            f = image_means[femur].get(metric, np.nan)
            if np.isfinite(h) and np.isfinite(f):
                hip_means.append(h)
                femur_means.append(f)

        if len(hip_means) != len(femur_means) or len(hip_means) < 2:
            print(f"  [WARN] {metric}: insufficient paired data — skipping.")
            continue

        hip_means   = np.array(hip_means)
        femur_means = np.array(femur_means)
        differences = hip_means - femur_means

        # --- Shapiro-Wilk on differences ---
        if len(differences) >= 3:
            sw_stat, sw_p = shapiro(differences)
        else:
            sw_stat, sw_p = np.nan, np.nan

        # --- Paired t-test ---
        tt = ttest_rel(hip_means, femur_means)

        # --- Wilcoxon signed-rank ---
        try:
            wx = wilcoxon(hip_means, femur_means)
            wx_stat, wx_p = wx.statistic, wx.pvalue
        except ValueError as e:
            wx_stat, wx_p = np.nan, np.nan
            print(f"  [WARN] Wilcoxon failed for {metric}: {e}")

        results.append({
            'metric':             metric,
            'n_pairs':            n_pairs,
            'hip_mean':           np.mean(hip_means),
            'hip_std':            np.std(hip_means,   ddof=1),
            'femur_mean':         np.mean(femur_means),
            'femur_std':          np.std(femur_means, ddof=1),
            'shapiro_W':          round(sw_stat, 4) if np.isfinite(sw_stat) else np.nan,
            'shapiro_p':          round(sw_p,    4) if np.isfinite(sw_p)    else np.nan,
            'differences_normal': 'yes' if (np.isfinite(sw_p) and sw_p > 0.05) else 'no',
            't_stat':             round(tt.statistic, 4),
            't_p':                round(tt.pvalue,    4),
            'wilcoxon_W':         round(wx_stat, 4) if np.isfinite(wx_stat) else np.nan,
            'wilcoxon_p':         round(wx_p,    4) if np.isfinite(wx_p)    else np.nan,
            'recommended_test':   'paired_t' if (np.isfinite(sw_p) and sw_p > 0.05) else 'wilcoxon',
            'recommended_p':      round(tt.pvalue, 4) if (np.isfinite(sw_p) and sw_p > 0.05) else round(wx_p, 4),
        })

    results_df = pd.DataFrame(results).set_index('metric')

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(f"\n{'='*70}")
    print(f"  PAIRED STATISTICAL TESTS — hip vs femur  (n={n_pairs} pairs)")
    print(f"{'='*70}")
    print(
        f"{'Metric':<22} {'Hip mean':>12} {'Femur mean':>12} {'Normal?':>8} "
        f"{'Test':>10} {'p':>8} {'Sig':>5}"
    )
    print(f"{'-'*70}")

    for _, row in results_df.iterrows():
        sig = (
            '***' if row['recommended_p'] < 0.001 else
            '**'  if row['recommended_p'] < 0.01  else
            '*'   if row['recommended_p'] < 0.05  else
            'ns'
        )
        print(
            f"|{row.name:<22} "
            f"|{row['hip_mean']:>12.4g} "
            f"|{row['femur_mean']:>12.4g} "
            f"|{row['differences_normal']:>8} "
            f"|{row['recommended_test']:>10} "
            f"|{row['recommended_p']:>8.4f} "
            f"|{sig:>5} |  "
        )

    print(f"{'='*70}")
    print(
        "Significance: * p<0.05   ** p<0.01   *** p<0.001   ns = not significant\n"
        "Note: Shapiro-Wilk has low power at small n — consider Wilcoxon as default.\n"
        "Note: Means are area-weighted across tiles (by filled_area) to avoid\n"
        "      bias from unequal tile counts between images.\n"
    )

    return results_df


if __name__ == '__main__':
    target_directory = Folder('data')
    # analyse_files(target_directory)
    # make_summary_table_10images(target_directory)
    run_paired_statistics(target_directory)
