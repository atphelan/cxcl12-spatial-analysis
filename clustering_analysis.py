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
    comparing hip vs femur for each metric, using per-image means.

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
            'cell_density','frac_dense_area', 'frac_sparse_area', 'frac_hole_area', 'ratio_high_low', 'prop_removed_area'
        ]

    # --------------------------------------------------------
    # Load and average all valid files
    # --------------------------------------------------------

    image_means = {}  # { image_base_name : { metric : mean_value } }

    for file in target_directory.files:
        if file.type != 'csv' or not ('save states' in file.name) or '.tif' in file.name:
            continue

        df = pd.read_csv(file.path)
        recalculate = True
        if not ('ratio_reconstructed_mask' in df.columns) or recalculate:
            df['mean_hole_area'] = df['hole_area']/df['n_holes']
            df = df.drop(columns=[c for c in df.columns if 'Unnamed:' in c])
            if not file.name == 'M2femur_results_save states.csv' or file.name == 'M5hip_results_save states.csv':
                df['total_hd_area'] = df['high_density_area'] + df['high_density_area_anom']
                df['total_ld_area'] = df['low_density_area'] + df['low_density_area_anom']
            else:
                df['total_hd_area'] = df['high_density_area']
                df['total_ld_area'] = df['low_density_area']
            df['reconstructed_tissue_area'] = (df['hole_area'] + df['total_hd_area'] + df['total_ld_area'])
            df['frac_dense_area'] = df['total_hd_area']/df['reconstructed_tissue_area']
            df['frac_sparse_area'] = df['total_ld_area']/df['reconstructed_tissue_area']
            df['frac_hole_area'] = df['hole_area']/df['reconstructed_tissue_area']
            df['prop_removed_area'] = df['area_discounted']/df['reconstructed_tissue_area']
            # df['adjusted_fraction_high_density_tissue'] = df['high_density_area']/(df['filled_area'])
            # df['adjusted_fraction_low_density_tissue'] = df['low_density_area']/(df['filled_area'])
            
            # df['adjusted_cell_density'] = df['n_cells']/(df['tissue_area']-df['hole_area'])

            df['ratio_reconstructed_mask'] = df['reconstructed_tissue_area']/df['filled_area']
            
            df.reset_index(drop=True).to_csv(file.path)
        df = df[df['ratio_reconstructed_mask']>0.18]


        if len(df) < min_windows:
            print(f"  [WARN] {file.name}: only {len(df)} windows — skipping (min={min_windows})")
            continue

        base_name = file.name.split('_bin')[0]  # e.g. 'M3hip' or 'M3femur'

        image_means[base_name] = {
            metric: df[metric].mean()
            for metric in metrics
            if metric in df.columns
        }

    if not image_means:
        raise ValueError("No valid files found. Check target directory and file naming.")

    # --------------------------------------------------------
    # Build matched hip / femur pairs
    # --------------------------------------------------------

    pairs = []

    for name, values in image_means.items():
        if 'hip' not in name:
            continue

        partner = name.replace('hip', 'femur')

        if partner not in image_means:
            print(f"  [WARN] No matching femur found for {name} — skipping.")
            continue

        pairs.append((name, partner))

    if not pairs:
        raise ValueError("No matched hip/femur pairs found. Check file naming.")

    n_pairs = len(pairs)
    print(f"\nFound {n_pairs} matched hip/femur pairs:")
    for hip, femur in pairs:
        print(f"  {hip}  <-->  {femur}")

    # --------------------------------------------------------
    # Run tests for each metric
    # --------------------------------------------------------

    results = []

    for metric in metrics:

        hip_means   = [image_means[hip][metric]   for hip, _      in pairs if metric in image_means[hip]]
        femur_means = [image_means[femur][metric]  for _,   femur  in pairs if metric in image_means[femur]]

        if len(hip_means) != len(femur_means) or len(hip_means) < 2:
            print(f"  [WARN] {metric}: insufficient paired data — skipping.")
            continue

        hip_means   = np.array(hip_means)
        femur_means = np.array(femur_means)
        differences = hip_means - femur_means

        # --- Shapiro-Wilk on differences ---
        if len(differences) >= 3:  # Shapiro requires n >= 3
            sw_stat, sw_p = shapiro(differences)
        else:
            sw_stat, sw_p = np.nan, np.nan

        # --- Paired t-test ---
        tt  = ttest_rel(hip_means, femur_means)

        # --- Wilcoxon signed-rank ---
        # Requires n >= 2; catches edge case where all differences are zero
        try:
            wx = wilcoxon(hip_means, femur_means)
            wx_stat, wx_p = wx.statistic, wx.pvalue
        except ValueError as e:
            wx_stat, wx_p = np.nan, np.nan
            print(f"  [WARN] Wilcoxon failed for {metric}: {e}")

        results.append({
            'metric':               metric,
            'n_pairs':              n_pairs,

            # Per-group descriptives
            'hip_mean':             np.mean(hip_means),
            'hip_std':              np.std(hip_means,   ddof=1),
            'femur_mean':           np.mean(femur_means),
            'femur_std':            np.std(femur_means, ddof=1),

            # Normality of differences
            'shapiro_W':            round(sw_stat, 4) if not np.isnan(sw_stat) else np.nan,
            'shapiro_p':            round(sw_p,    4) if not np.isnan(sw_p)    else np.nan,
            'differences_normal':   'yes' if sw_p > 0.05 else 'no',

            # Paired t-test
            't_stat':               round(tt.statistic, 4),
            't_p':                  round(tt.pvalue,    4),

            # Wilcoxon signed-rank
            'wilcoxon_W':           round(wx_stat, 4) if not np.isnan(wx_stat) else np.nan,
            'wilcoxon_p':           round(wx_p,    4) if not np.isnan(wx_p)    else np.nan,

            # Recommended test given normality check
            'recommended_test':     'paired_t' if sw_p > 0.05 else 'wilcoxon',
            'recommended_p':        round(tt.pvalue, 4) if sw_p > 0.05 else round(wx_p, 4),
        })

    results_df = pd.DataFrame(results).set_index('metric')

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(f"\n{'='*70}")
    print(f"  PAIRED STATISTICAL TESTS — hip vs femur  (n={n_pairs} pairs)")
    print(f"{'='*70}")
    print(f"{'Metric':<22} {'Hip mean':>12} {'Femur mean':>12} {'Normal?':>8} "
        f"{'Test':>10} {'p':>8} {'Sig':>5}")
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
    print("Significance: * p<0.05   ** p<0.01   *** p<0.001   ns = not significant under this test")
    print("Note: Shapiro-Wilk has low power at small n — consider Wilcoxon as default.\n")

    return results_df

if __name__ == '__main__':
    target_directory = Folder('data')
    # analyse_files(target_directory)
    # make_summary_table_10images(target_directory)
    run_paired_statistics(target_directory)
