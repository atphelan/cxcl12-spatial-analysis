import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro


# ── Colours ───────────────────────────────────────────────────────────────────
FEMUR_COLOUR = '#2166ac'   # blue
HIP_COLOUR   = '#d6604d'   # red
ALPHA_BOX    = 0.35
ALPHA_MEAN   = 0.85
fontsize_base = 16

def plot_per_mouse_boxplots(
    target_directory,
    dependent         = 'cell_density_whole_mask',
    adjustment_factor = 1.0,
    y_label           = None,
    title             = None,
    min_windows       = 3,
    ignore_yellows    = None,
    figsize           = (11, 5),
    save_path         = None,
):
    """
    Single-axis paired boxplots — one femur/hip pair per mouse, evenly
    spaced along x with a small gap between bones within each pair and a
    larger gap between mice.  Overall femur and hip means are drawn as
    dashed horizontal lines clipped to the plot area.
    """

    if ignore_yellows is None:
        ignore_yellows = [
            'M2femur_results_save states.csv',
            'M5hip_results_save states.csv',
            'M3hip_results_save states.csv',
        ]

    if y_label is None:
        y_label = dependent
    if title is None:
        title = f'{dependent}  —  femur vs hip per mouse'

    FILE_ENDING = '_results_save states.csv'

    # ── 1. Load data ──────────────────────────────────────────────────────────

    image_data  = {}
    image_means = {}

    for file in target_directory.files:
        if file.type != 'csv':
            continue
        if 'save states' not in file.name:
            continue
        if '.tif' in file.name:
            continue

        df = pd.read_csv(file.path)
        df = df.drop(columns=[c for c in df.columns if 'Unnamed:' in c],
                     errors='ignore')

        if file.name not in ignore_yellows:
            df['total_hd_area'] = df['high_density_area'] + df['high_density_area_anom']
            df['total_ld_area'] = df['low_density_area']  + df['low_density_area_anom']
        else:
            df['total_hd_area'] = df['high_density_area']
            df['total_ld_area'] = df['low_density_area']

        if file.name in (
            'M5femur_results_save states.csv',
            'M2hip_results_save states.csv'
        ):
            # Dark spots in GFP image not well accounted for in whole
            # mask density — use reconstructed-area density directly
            df['cell_density_whole_mask'] = df['cell_density']

        if len(df) < min_windows:
            print(f'  [WARN] {file.name}: only {len(df)} windows — skipping.')
            continue

        if dependent not in df.columns:
            print(f'  [WARN] {file.name}: column "{dependent}" not found — skipping.')
            continue

        base    = file.name          # keep full filename as key
        vals    = df[dependent].values * adjustment_factor
        weights = df['filled_area'].values
        valid   = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)

        image_data[base]  = vals[valid]
        image_means[base] = np.average(vals[valid], weights=weights[valid])

    # ── 2. Mice and overall means ─────────────────────────────────────────────

    mice = sorted({k[:2] for k in image_data})   # ['M1', 'M2', ...]

    femur_mouse_means, hip_mouse_means = [], []
    for m in mice:
        fk = m + 'femur' + FILE_ENDING
        hk = m + 'hip'   + FILE_ENDING
        if fk in image_means:
            femur_mouse_means.append(image_means[fk])
        if hk in image_means:
            hip_mouse_means.append(image_means[hk])

    overall_femur = np.mean(femur_mouse_means) if femur_mouse_means else None
    overall_hip   = np.mean(hip_mouse_means)   if hip_mouse_means   else None

    # ── 3. Significance ───────────────────────────────────────────────────────

    paired_f = np.array(femur_mouse_means)
    paired_h = np.array(hip_mouse_means)
    diffs    = paired_h - paired_f

    sw_p = shapiro(diffs).pvalue if len(diffs) >= 3 else np.nan

    try:
        if np.isfinite(sw_p) and sw_p > 0.05:
            p_val      = ttest_rel(paired_f, paired_h).pvalue
            test_label = 'paired t-test'
        else:
            p_val      = wilcoxon(paired_f, paired_h).pvalue
            test_label = 'Wilcoxon'
    except Exception:
        p_val, test_label = np.nan, ''

    if np.isfinite(p_val):
        stars    = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        p_str    = 'p < 0.001' if p_val < 0.001 else f'p = {p_val:.4f}'
        sig_str  = f'{p_str}  {stars}  ({test_label})'
    else:
        sig_str = ''

    # ── 4. X-position layout ─────────────────────────────────────────────────
    #
    #   Within each mouse pair the femur box sits at  centre - INNER_GAP/2
    #   and the hip box at  centre + INNER_GAP/2.
    #   Mouse centres are spaced MOUSE_STEP apart.

    INNER_GAP  = 0.30   # gap between femur and hip box centres within a pair
    MOUSE_STEP = 1.00   # distance between successive mouse centres
    BOX_WIDTH  = 0.25   # width of each individual box

    mouse_centres = {m: i * MOUSE_STEP for i, m in enumerate(mice)}

    femur_positions = {m: mouse_centres[m] - INNER_GAP / 2 for m in mice}
    hip_positions   = {m: mouse_centres[m] + INNER_GAP / 2 for m in mice}

    x_min = min(femur_positions.values()) - BOX_WIDTH * 2
    x_max = max(hip_positions.values())   + BOX_WIDTH * 2

    # ── 5. Plot ───────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=figsize)

    bp_base = dict(
        patch_artist = True,
        widths       = BOX_WIDTH,
        medianprops  = dict(linewidth=2.0, color='black'),
        whiskerprops = dict(linewidth=1.2),
        capprops     = dict(linewidth=1.2),
        flierprops   = dict(marker='o', markersize=3, linestyle='none',
                            markeredgewidth=0.8),
        manage_ticks = False,   # we set ticks manually
    )

    for mouse in mice:
        fk = mouse + 'femur' + FILE_ENDING
        hk = mouse + 'hip'   + FILE_ENDING

        f_vals = image_data.get(fk, np.array([]))
        h_vals = image_data.get(hk, np.array([]))

        fp = femur_positions[mouse]
        hp = hip_positions[mouse]

        # ── femur box ────────────────────────────────────────────────────────
        if len(f_vals):
            bp = ax.boxplot(
                [f_vals], positions=[fp],
                **bp_base,
            )
            bp['boxes'][0].set_facecolor(FEMUR_COLOUR)
            bp['boxes'][0].set_alpha(ALPHA_BOX)
            bp['boxes'][0].set_edgecolor(FEMUR_COLOUR)
            bp['fliers'][0].set(markerfacecolor=FEMUR_COLOUR,
                                markeredgecolor=FEMUR_COLOUR)

        # ── hip box ──────────────────────────────────────────────────────────
        if len(h_vals):
            bp = ax.boxplot(
                [h_vals], positions=[hp],
                **bp_base,
            )
            bp['boxes'][0].set_facecolor(HIP_COLOUR)
            bp['boxes'][0].set_alpha(ALPHA_BOX)
            bp['boxes'][0].set_edgecolor(HIP_COLOUR)
            bp['fliers'][0].set(markerfacecolor=HIP_COLOUR,
                                markeredgecolor=HIP_COLOUR)

        # ── per-mouse weighted mean diamonds ─────────────────────────────────
        if fk in image_means:
            ax.plot(fp, image_means[fk], marker='D', markersize=7,
                    color=FEMUR_COLOUR, zorder=5)
        if hk in image_means:
            ax.plot(hp, image_means[hk], marker='D', markersize=7,
                    color=HIP_COLOUR, zorder=5)

        # ── mouse label centred between the pair ─────────────────────────────
        ax.text(
            mouse_centres[mouse], -0.055,
            mouse,
            transform      = ax.get_xaxis_transform(),
            ha             = 'center',
            va             = 'top',
            fontsize       = fontsize_base,
            fontweight     = 'bold',
        )

    # ── Overall mean lines, clipped to axes data area ─────────────────────────
    #    Using axhline with xmin/xmax in axes-fraction coordinates derived
    #    from the data x-limits so the lines stay inside the plot box.

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1100)

    x_range   = x_max - x_min
    xmin_frac = (min(femur_positions.values()) - BOX_WIDTH * 1.2 - x_min) / x_range
    xmax_frac = (max(hip_positions.values())   + BOX_WIDTH * 1.2 - x_min) / x_range

    if overall_femur is not None:
        ax.axhline(
            overall_femur,
            xmin      = xmin_frac,
            xmax      = xmax_frac,
            color     = FEMUR_COLOUR,
            linewidth = 1.6,
            linestyle = '--',
            alpha     = ALPHA_MEAN,
            zorder    = 3,
        )

    if overall_hip is not None:
        ax.axhline(
            overall_hip,
            xmin      = xmin_frac,
            xmax      = xmax_frac,
            color     = HIP_COLOUR,
            linewidth = 1.6,
            linestyle = '--',
            alpha     = ALPHA_MEAN,
            zorder    = 3,
        )

    # ── Axes formatting ───────────────────────────────────────────────────────

    ax.set_xticks([])                      # mouse labels drawn manually above
    ax.set_ylabel(y_label, fontsize=fontsize_base)
    ax.tick_params(axis='y', labelsize=fontsize_base)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)  # no x-axis line needed

    # ── Legend ────────────────────────────────────────────────────────────────

    mean_f_str = f'{overall_femur:.3g}' if overall_femur is not None else '—'
    mean_h_str = f'{overall_hip:.3g}'   if overall_hip   is not None else '—'

    femur_patch = mpatches.Patch(
        facecolor=FEMUR_COLOUR, alpha=ALPHA_BOX, edgecolor=FEMUR_COLOUR,
        label=f'Femur  (overall mean {mean_f_str})',
    )
    hip_patch = mpatches.Patch(
        facecolor=HIP_COLOUR, alpha=ALPHA_BOX, edgecolor=HIP_COLOUR,
        label=f'Hip  (overall mean {mean_h_str})',
    )
    mean_line = plt.Line2D(
        [0], [0], color='grey', linestyle='--', linewidth=1.6,
        label='Overall femur/hip mean',
    )
    diamond = plt.Line2D(
        [0], [0], marker='D', markersize=7, linestyle='none', color='grey',
        label='Per-mouse area-weighted mean',
    )

    ax.legend(
        handles       = [femur_patch, hip_patch,
                         mean_line, diamond],
        loc           = 'upper left',
        fontsize      = fontsize_base*0.9,
        frameon       = False,
        ncol          = 2
    )

    # ── Title ─────────────────────────────────────────────────────────────────

    ax.set_title(f'{title}\n{sig_str}', fontsize=fontsize_base*1.2, pad=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
    else:
        plt.show()

    return fig, ax


def plot_two_column_scatter(
    target_directory,
    dependent         = 'cell_density_whole_mask',
    adjustment_factor = 1.0,
    normalise_to_femur= False,
    y_label           = None,
    title             = None,
    min_windows       = 3,
    ignore_yellows    = None,
    figsize           = (4, 5),
    font_size         = 13,
    save_path         = None,
):
    """
    Two-column scatter plot (femur left, hip right) showing all tile-level
    values across all mice, colour-coded by mouse.  Per-mouse weighted means
    are shown as horizontal lines spanning each column.  Optionally normalises
    all values to the per-mouse femur weighted mean.

    Parameters
    ----------
    target_directory : Folder
    dependent : str
        Column name to plot on the y-axis.
    adjustment_factor : float
        Multiplicative scaling for y values (e.g. 1e6 for µm⁻² → mm⁻²).
    normalise_to_femur : bool
        If True, divide all values (both bones) for each mouse by that
        mouse's area-weighted femur mean, so femur points scatter around 1.
    y_label : str or None
    title : str or None
    min_windows : int
    ignore_yellows : list of str or None
    figsize : tuple
    font_size : int
        Base font size — axis labels, tick labels, and annotations scale
        from this value.
    save_path : str or None
    """

    from scipy.stats import ttest_rel, wilcoxon, shapiro

    if ignore_yellows is None:
        ignore_yellows = [
            'M2femur_results_save states.csv',
            'M5hip_results_save states.csv',
            'M3hip_results_save states.csv',
        ]

    if y_label is None:
        y_label = dependent
        if normalise_to_femur:
            y_label += '  (normalised to femur mean)'
    if title is None:
        title = dependent

    FILE_ENDING = '_results_save states.csv'

    # Distinct colours per mouse — colourblind-friendly palette
    MOUSE_COLOURS = {
        'M1': '#4477AA',
        'M2': '#EE6677',
        'M3': '#228833',
        'M4': '#CCBB44',
        'M5': '#AA3377',
    }

    # ── 1. Load tile-level data ───────────────────────────────────────────────

    image_tiles = {}    # { 'M1femur...' : array of scaled tile values }
    image_means = {}    # { 'M1femur...' : area-weighted mean }

    for file in target_directory.files:
        if file.type != 'csv':
            continue
        if 'save states' not in file.name:
            continue
        if '.tif' in file.name:
            continue

        df = pd.read_csv(file.path)
        df = df.drop(columns=[c for c in df.columns if 'Unnamed:' in c],
                     errors='ignore')

        if file.name in (
            'M5femur_results_save states.csv',
            'M2hip_results_save states.csv',
        ):
            df['cell_density_whole_mask'] = df['cell_density']

        df = df[df['ratio_reconstructed_mask'] > 0.18]

        if len(df) < min_windows:
            print(f'  [WARN] {file.name}: only {len(df)} windows — skipping.')
            continue

        if dependent not in df.columns:
            print(f'  [WARN] {file.name}: column "{dependent}" not found — skipping.')
            continue

        vals    = df[dependent].values * adjustment_factor
        weights = df['filled_area'].values
        valid   = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)

        image_tiles[file.name] = vals[valid]
        image_means[file.name] = np.average(vals[valid], weights=weights[valid])

    # ── 2. Mice and normalisation denominators ────────────────────────────────

    mice = sorted({k[:2] for k in image_means})

    # Per-mouse femur mean used for optional normalisation
    femur_norm = {}
    for m in mice:
        fk = m + 'femur' + FILE_ENDING
        if fk in image_means:
            femur_norm[m] = image_means[fk]

    def normalise(mouse, val):
        if not normalise_to_femur:
            return val
        denom = femur_norm.get(mouse, np.nan)
        return val / denom if denom and np.isfinite(denom) else np.nan

    # ── 3. Significance test on per-mouse weighted means ─────────────────────

    paired_f, paired_h = [], []
    for m in mice:
        fk = m + 'femur' + FILE_ENDING
        hk = m + 'hip'   + FILE_ENDING
        if fk in image_means and hk in image_means:
            paired_f.append(normalise(m, image_means[fk]))
            paired_h.append(normalise(m, image_means[hk]))

    paired_f = np.array(paired_f)
    paired_h = np.array(paired_h)
    diffs    = paired_h - paired_f

    sw_p = shapiro(diffs).pvalue if len(diffs) >= 3 else np.nan

    try:
        if np.isfinite(sw_p) and sw_p > 0.05:
            p_val      = ttest_rel(paired_f, paired_h).pvalue
            test_label = 'paired t'
        else:
            p_val      = wilcoxon(paired_f, paired_h).pvalue
            test_label = 'Wilcoxon'
    except Exception:
        p_val, test_label = np.nan, ''

    if np.isfinite(p_val):
        stars   = '***' if p_val < 0.001 else '**' if p_val < 0.01 \
                  else '*' if p_val < 0.05 else 'ns'
        p_str   = 'p < 0.001' if p_val < 0.001 else f'p = {p_val:.4f}'
        sig_str = f'{p_str}  {stars}\n({test_label}, n = {len(paired_f)})'
    else:
        sig_str = ''

    # ── 4. Plot ───────────────────────────────────────────────────────────────

    JITTER      = 0.07    # horizontal jitter width
    ALPHA_TILE  = 0.55    # tile point transparency
    MARKER_SIZE = 40      # tile point size
    MEAN_WIDTH  = 0.30    # half-width of the per-mouse mean tick

    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(seed=42)

    # Track all plotted y values to set ylim sensibly
    all_y = []

    for mouse in mice:
        colour = MOUSE_COLOURS.get(mouse, 'grey')

        for col_x, bone in ((0, 'femur'), (1, 'hip')):
            key  = mouse + bone + FILE_ENDING
            if key not in image_tiles:
                continue

            raw_tiles = image_tiles[key]
            raw_mean  = image_means[key]

            tiles = np.array([normalise(mouse, v) for v in raw_tiles])
            mean  = normalise(mouse, raw_mean)

            # Jittered x positions
            jx = col_x + rng.uniform(-JITTER, JITTER, size=len(tiles))

            ax.scatter(
                jx, tiles,
                color      = colour,
                s          = MARKER_SIZE,
                alpha      = ALPHA_TILE,
                linewidths = 0.3,
                edgecolors = 'white',
                zorder     = 3,
            )

            # Per-mouse mean as a bold horizontal tick
            ax.plot(
                [col_x - MEAN_WIDTH, col_x + MEAN_WIDTH],
                [mean, mean],
                color     = colour,
                linewidth = 2.5,
                solid_capstyle = 'round',
                zorder    = 5,
            )

            all_y.extend(tiles[np.isfinite(tiles)].tolist())

    # ── 5. Overall mean lines ─────────────────────────────────────────────────

    OVERALL_WIDTH = 0.42

    if len(paired_f):
        ax.plot(
            [-OVERALL_WIDTH, OVERALL_WIDTH],
            [paired_f.mean(), paired_f.mean()],
            color     = 'black',
            linewidth = 2.0,
            linestyle = '--',
            zorder    = 6,
        )
    if len(paired_h):
        ax.plot(
            [1 - OVERALL_WIDTH, 1 + OVERALL_WIDTH],
            [paired_h.mean(), paired_h.mean()],
            color     = 'black',
            linewidth = 2.0,
            linestyle = '--',
            zorder    = 6,
        )

    # ── 6. Significance bracket ───────────────────────────────────────────────

    y_max     = max(all_y) if all_y else 1
    y_range   = y_max - min(all_y) if all_y else 1
    bracket_y = y_max + y_range * 0.10
    desc_len  = y_range * 0.03

    ax.plot([0, 0, 1, 1],
            [bracket_y - desc_len, bracket_y,
             bracket_y, bracket_y - desc_len],
            color='dimgrey', linewidth=1.2)

    ax.text(
        0.5, bracket_y + y_range * 0.01,
        sig_str,
        ha        = 'center',
        va        = 'bottom',
        fontsize  = font_size - 2,
        color     = 'dimgrey',
        linespacing = 1.4,
    )

    # ── 7. Reference line at 1.0 when normalised ─────────────────────────────

    if normalise_to_femur:
        ax.axhline(1.0, color='dimgrey', linewidth=0.8,
                   linestyle=':', alpha=0.6, zorder=2)

    # ── 8. Legend — mice only ─────────────────────────────────────────────────

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=MOUSE_COLOURS.get(m, 'grey'),
                   markersize=font_size * 0.7, label=m)
        for m in mice if m in MOUSE_COLOURS
    ]
    # legend_handles.append(
    #     plt.Line2D([0], [0], color='black', linewidth=2.0,
    #                linestyle='--', label='Overall mean')
    # )
    # legend_handles.append(
    #     plt.Line2D([0], [0], color='grey', linewidth=2.5,
    #                label='Per-mouse mean')
    # )

    ax.legend(
        handles   = legend_handles,
        fontsize  = font_size - 3,
        frameon   = False,
        loc       = 'upper right',
        handlelength = 1.2,
    )

    # ── 9. Axes formatting ────────────────────────────────────────────────────

    y_pad = y_range * 0.05
    ax.set_ylim(min(all_y) - y_pad,
                bracket_y + y_range * 0.18 + len(sig_str.splitlines()) * y_range * 0.06)
    ax.set_xlim(-0.55, 1.55)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Femur', 'Hip'], fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size - 1)
    ax.tick_params(axis='y', labelsize=font_size - 2)

    ax.set_title(title, fontsize=font_size - 1, pad=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
    else:
        plt.show()

    return fig, ax



def plot_2d_scatter(
    target_directory,
    x                 = '',
    y                 = '',
    adjustment_factor = 1.0,
    y_label           = None,
    x_label           = 'Proportion of image area removed (non-parenchymal)',
    title             = None,
    min_windows       = 3,
    ignore_yellows    = None,
    figsize           = (7, 5),
    save_path         = None,
):
    """
    Scatter plot of prop_removed_area (x) vs a density metric (y),
    with one point per tile, coloured by bone type (femur=blue, hip=red),
    and a linear regression line per bone type with 95% confidence band.

    Parameters
    ----------
    target_directory : Folder
    density_metric : str
        Column to use as the y-axis (default: cell_density_whole_mask).
    adjustment_factor : float
        Multiplicative scaling for the y-axis values.
    y_label, x_label, title : str or None
    min_windows : int
    ignore_yellows : list of str or None
    figsize : tuple
    save_path : str or None
    """

    from scipy.stats import pearsonr, t as t_dist

    if ignore_yellows is None:
        ignore_yellows = [
            'M2femur_results_save states.csv',
            'M5hip_results_save states.csv',
            'M3hip_results_save states.csv',
        ]
    if x_label is None:
        x_label = x
    if y_label is None:
        y_label = y
    if title is None:
        title = f'{x} vs {y}'

    FILE_ENDING = '_results_save states.csv'

    # ── 1. Load tile-level data ───────────────────────────────────────────────

    records = []   # list of dicts, one per valid tile

    for file in target_directory.files:
        if file.type != 'csv':
            continue
        if 'save states' not in file.name:
            continue
        if '.tif' in file.name:
            continue

        df = pd.read_csv(file.path)
        df = df.drop(columns=[c for c in df.columns if 'Unnamed:' in c],
                     errors='ignore')

        df = df[df['ratio_reconstructed_mask'] > 0.18]

        if len(df) < min_windows:
            print(f'  [WARN] {file.name}: only {len(df)} windows — skipping.')
            continue

        if y not in df.columns or 'prop_removed_area' not in df.columns:
            print(f'  [WARN] {file.name}: required columns missing — skipping.')
            continue

        mouse = file.name[:2]                                    # 'M1' etc.
        bone  = 'femur' if 'femur' in file.name else 'hip'

        for _, row in df.iterrows():
            x0 = row['prop_removed_area']
            y0 = row[y] * adjustment_factor
            if np.isfinite(x0) and np.isfinite(y0):
                records.append({'mouse': mouse, 'bone': bone,
                                'x': x0, 'y': y0})

    if not records:
        raise ValueError('No valid tile data found.')

    data = pd.DataFrame(records)

    # ── 2. Regression helper ──────────────────────────────────────────────────

    def regression_with_ci(x, y, n_pts=200, alpha=0.05):
        """
        Returns x_line, y_line, y_lower, y_upper for a simple OLS fit
        with pointwise 95 % confidence band.
        """
        x, y   = np.asarray(x), np.asarray(y)
        n      = len(x)
        x_mean = x.mean()
        Sxx    = ((x - x_mean) ** 2).sum()
        # OLS coefficients
        b1     = np.dot(x - x_mean, y) / Sxx
        b0     = y.mean() - b1 * x_mean
        y_hat  = b0 + b1 * x
        # Residual standard error
        sse    = ((y - y_hat) ** 2).sum()
        s      = np.sqrt(sse / (n - 2))
        # Prediction grid
        x_line = np.linspace(x.min(), x.max(), n_pts)
        y_line = b0 + b1 * x_line
        # Standard error of fitted mean at each x_line point
        se     = s * np.sqrt(1/n + (x_line - x_mean)**2 / Sxx)
        t_crit = t_dist.ppf(1 - alpha / 2, df=n - 2)
        # Pearson r and p
        r, p   = pearsonr(x, y)
        return x_line, y_line, y_line - t_crit*se, y_line + t_crit*se, r, p, b1

    # ── 3. Plot ───────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=figsize)

    MARKER_ALPHA = 0.55
    MARKER_SIZE  = 28

    bone_cfg = {
        'femur': {'colour': FEMUR_COLOUR, 'label': 'Femur', 'marker': 'o'},
        'hip':   {'colour': HIP_COLOUR,   'label': 'Hip',   'marker': 's'},
    }

    # Use a distinct point marker per mouse so individual mice are traceable
    mouse_markers = {m: mk for m, mk in
                     zip(sorted(data['mouse'].unique()),
                         ['o', 's', '^', 'D', 'v', 'P', 'X'])}

    legend_handles = []

    for bone, cfg in bone_cfg.items():
        sub = data[data['bone'] == bone]
        if sub.empty:
            continue

        colour = cfg['colour']

        # ── scatter, one marker style per mouse ───────────────────────────────
        for mouse, grp in sub.groupby('mouse'):
            sc = ax.scatter(
                grp['x'], grp['y'],
                color     = colour,
                marker    = mouse_markers[mouse],
                s         = MARKER_SIZE,
                alpha     = MARKER_ALPHA,
                linewidths= 0.4,
                edgecolors= 'white',
                zorder    = 4,
            )

        # ── regression line + CI band ─────────────────────────────────────────
        x_line, y_line, y_lo, y_hi, r, p, slope = regression_with_ci(
            sub['x'].values, sub['y'].values
        )

        ax.plot(x_line, y_line, color=colour, linewidth=2.0, zorder=5)
        ax.fill_between(x_line, y_lo, y_hi,
                        color=colour, alpha=0.12, zorder=3)

        # ── annotation: r and p ───────────────────────────────────────────────
        p_str  = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
        stars  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        annot  = f'{cfg["label"]}:  r = {r:.2f},  {p_str}  {stars}'

        legend_handles.append(
            mpatches.Patch(facecolor=colour, alpha=0.5,
                           edgecolor=colour, label=annot)
        )

    # ── mouse marker legend entries ───────────────────────────────────────────
    for mouse, mk in mouse_markers.items():
        if mouse in data['mouse'].values:
            legend_handles.append(
                plt.Line2D([0], [0], marker=mk, color='grey',
                           markersize=6, linestyle='none',
                           label=mouse)
            )

    # ── axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel(x_label, fontsize=fontsize_base)
    ax.set_ylabel(y_label, fontsize=fontsize_base)
    ax.set_title(title,    fontsize=fontsize_base*1.2, pad=8)
    ax.tick_params(labelsize=fontsize_base)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        handles       = legend_handles,
        fontsize      = fontsize_base,
        frameon       = False,
        loc           = 'upper right',
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
    else:
        plt.show()

    return fig, ax


def plot_density_adjustment_bars(
    target_directory,
    adjustment_factor = 1.0,
    y_label           = 'CXCL12 cell density  (cells / mm²)',
    title             = 'Effect of non-parenchymal area removal on reported density',
    min_windows       = 3,
    ignore_yellows    = None,
    figsize           = (11, 5),
    save_path         = None,
):
    """
    Stacked bar chart showing, for each mouse and bone type, the
    baseline density (cell_density_whole_mask, darker) and the
    additional density gained by restricting to reconstructed
    parenchymal area only (cell_density - cell_density_whole_mask,
    lighter).  Bars are arranged in the same paired layout as
    plot_per_mouse_boxplots.  A significance bracket compares the
    fractional adjustment between bone types.
    """

    from scipy.stats import ttest_rel, wilcoxon, shapiro

    if ignore_yellows is None:
        ignore_yellows = [
            'M2femur_results_save states.csv',
            'M5hip_results_save states.csv',
            'M3hip_results_save states.csv',
        ]

    FILE_ENDING = '_results_save states.csv'

    # ── 1. Load and aggregate ─────────────────────────────────────────────────

    image_means = {}   # { 'M1femur_results_save states.csv' :
                       #   { 'base': float, 'adjusted': float } }

    for file in target_directory.files:
        if file.type != 'csv':
            continue
        if 'save states' not in file.name:
            continue
        if '.tif' in file.name:
            continue

        df = pd.read_csv(file.path)
        df = df.drop(columns=[c for c in df.columns if 'Unnamed:' in c],
                     errors='ignore')

        if len(df) < min_windows:
            print(f'  [WARN] {file.name}: only {len(df)} windows — skipping.')
            continue

        weights = df['filled_area'].values
        valid   = np.isfinite(weights) & (weights > 0)

        def wmean(col):
            v = df[col].values
            ok = valid & np.isfinite(v)
            return np.average(v[ok], weights=weights[ok]) if ok.sum() else np.nan

        base     = wmean('cell_density_whole_mask') * adjustment_factor
        adjusted = wmean('cell_density')            * adjustment_factor

        image_means[file.name] = {
            'base':     base,
            'adjusted': adjusted,
            # fractional uplift: how much extra density the adjustment adds
            # relative to the baseline
            'frac_uplift': (adjusted - base) / base if base > 0 else np.nan,
        }

    # ── 2. Organise into mice / bones ─────────────────────────────────────────

    mice = sorted({k[:2] for k in image_means})

    # ── 3. Significance of fractional uplift difference ───────────────────────

    femur_uplift, hip_uplift = [], []
    for m in mice:
        fk = m + 'femur' + FILE_ENDING
        hk = m + 'hip'   + FILE_ENDING
        if fk in image_means and hk in image_means:
            femur_uplift.append(image_means[fk]['frac_uplift'])
            hip_uplift.append(image_means[hk]['frac_uplift'])

    paired_f = np.array(femur_uplift)
    paired_h = np.array(hip_uplift)
    diffs    = paired_h - paired_f

    sw_p = shapiro(diffs).pvalue if len(diffs) >= 3 else np.nan

    try:
        if np.isfinite(sw_p) and sw_p > 0.05:
            p_val      = ttest_rel(paired_f, paired_h).pvalue
            test_label = 'paired t-test'
        else:
            p_val      = wilcoxon(paired_f, paired_h).pvalue
            test_label = 'Wilcoxon'
    except Exception:
        p_val, test_label = np.nan, ''

    if np.isfinite(p_val):
        stars   = '***' if p_val < 0.001 else '**' if p_val < 0.01 \
                  else '*' if p_val < 0.05 else 'ns'
        p_str   = 'p < 0.001' if p_val < 0.001 else f'p = {p_val:.4f}'
        sig_str = f'Fractional adjustment hip vs femur:  {p_str}  {stars}  ({test_label})'
    else:
        sig_str = ''

    # ── 4. X layout (mirrors plot_per_mouse_boxplots) ─────────────────────────

    INNER_GAP  = 0.30
    MOUSE_STEP = 1.00
    BAR_WIDTH  = 0.25

    mouse_centres   = {m: i * MOUSE_STEP for i, m in enumerate(mice)}
    femur_positions = {m: mouse_centres[m] - INNER_GAP / 2 for m in mice}
    hip_positions   = {m: mouse_centres[m] + INNER_GAP / 2 for m in mice}

    x_min = min(femur_positions.values()) - BAR_WIDTH * 2.5
    x_max = max(hip_positions.values())   + BAR_WIDTH * 2.5

    # ── 5. Normalise bars to femur baseline = 1.0 per mouse ───────────────────
    #    This makes the fractional uplift visually comparable across mice
    #    regardless of absolute density differences.

    fig, ax = plt.subplots(figsize=figsize)

    # Track y-tops for the significance bracket
    bar_tops = {'femur': [], 'hip': []}

    for mouse in mice:
        fk = mouse + 'femur' + FILE_ENDING
        hk = mouse + 'hip'   + FILE_ENDING

        if fk not in image_means or hk not in image_means:
            continue

        # Normalise both bars to the femur baseline of this mouse
        norm = image_means[fk]['base']

        for key, pos, colour, bone in (
            (fk, femur_positions[mouse], FEMUR_COLOUR, 'femur'),
            (hk, hip_positions[mouse],   HIP_COLOUR,   'hip'),
        ):
            base_h     = image_means[key]['base']     / norm
            adjusted_h = image_means[key]['adjusted'] / norm
            extra_h    = adjusted_h - base_h          # stacked top segment

            # Base segment (darker — whole mask density)
            ax.bar(
                pos, base_h,
                width     = BAR_WIDTH,
                color     = colour,
                alpha     = 0.80,
                zorder    = 3,
            )
            # Adjustment segment (lighter — extra from parenchyma-only area)
            ax.bar(
                pos, extra_h,
                bottom    = base_h,
                width     = BAR_WIDTH,
                color     = colour,
                alpha     = 0.30,
                zorder    = 3,
                hatch     = '////',
                edgecolor = colour,
                linewidth = 0.5,
            )

            bar_tops[bone].append(adjusted_h)

            # Fractional uplift label above bar
            uplift_pct = image_means[key]['frac_uplift'] * 100
            ax.text(
                pos, adjusted_h + 0.015,
                f'+{uplift_pct:.0f}%',
                ha        = 'center',
                va        = 'bottom',
                fontsize  = fontsize_base*0.8,
                color     = colour,
                fontweight= 'bold',
            )

        # Mouse label centred between the pair
        ax.text(
            mouse_centres[mouse], -0.055,
            mouse,
            transform  = ax.get_xaxis_transform(),
            ha         = 'center',
            va         = 'top',
            fontsize   = fontsize_base,
            fontweight = 'bold',
        )

    # ── 6. Significance bracket spanning all femur vs all hip bars ────────────

    bracket_y  = max(max(bar_tops['femur']), max(bar_tops['hip'])) + 0.12
    # bracket_pad= 0.03

    # all_femur_x = [femur_positions[m] for m in mice]
    # all_hip_x   = [hip_positions[m]   for m in mice]

    # f_left, f_right = min(all_femur_x) - BAR_WIDTH/2, max(all_femur_x) + BAR_WIDTH/2
    # h_left, h_right = min(all_hip_x)   - BAR_WIDTH/2, max(all_hip_x)   + BAR_WIDTH/2

    # # Draw bracket: two horizontal arms + two short descenders + central label
    # arm_y    = bracket_y
    # desc_len = 0.04

    # for left, right, colour in (
    #     (f_left, f_right, FEMUR_COLOUR),
    #     (h_left, h_right, HIP_COLOUR),
    # ):
    #     ax.plot([left, right], [arm_y, arm_y],
    #             color=colour, linewidth=1.4, clip_on=False)
    #     ax.plot([left,  left],  [arm_y - desc_len, arm_y],
    #             color=colour, linewidth=1.4, clip_on=False)
    #     ax.plot([right, right], [arm_y - desc_len, arm_y],
    #             color=colour, linewidth=1.4, clip_on=False)

    # # Connecting line between the two arms
    # mid_f = (f_left + f_right) / 2
    # mid_h = (h_left + h_right) / 2
    # ax.plot([mid_f, mid_h], [arm_y, arm_y],
    #         color='dimgrey', linewidth=1.2,
    #         linestyle='--', clip_on=False)

    # ax.text(
    #     (mid_f + mid_h) / 2, arm_y + bracket_pad,
    #     stars,
    #     ha        = 'center',
    #     va        = 'bottom',
    #     fontsize  = fontsize_base,
    #     color     = 'dimgrey',
    # )

    # ── 7. Reference line at y = 1 (femur baseline) ───────────────────────────

    ax.axhline(1.0, color=FEMUR_COLOUR, linewidth=1.0,
               linestyle=':', alpha=0.6, zorder=2, xmax=0.87)
    ax.text(x_max + 0.05, 1.0 - 0.02, 'Full femur',
            va='center', fontsize=fontsize_base*0.7, color=FEMUR_COLOUR, alpha=0.8)

    ax.axhline(402.3/543.5, color=HIP_COLOUR, linewidth=1.0,
               linestyle=':', alpha=0.6, zorder=2, xmax=0.87)
    ax.text(x_max + 0.05, 402.3/543.5, 'Full hip',
            va='center', fontsize=fontsize_base*0.7, color=HIP_COLOUR, alpha=0.8)

    # Adding lines showing new femur and hip means except not because normalisation is per mouse

    ax.axhline(604.0/543.5, color=FEMUR_COLOUR, linewidth=1.0,
               linestyle=':', alpha=0.6, zorder=2, xmax=0.87)
    ax.text(x_max + 0.05, 604.0/543.5 + 0.02, 'Parenchyma-only\nfemur & hip',
            va='center', fontsize=fontsize_base*0.7, color='k', alpha=0.8)

    ax.axhline(597.0/543.5, color=HIP_COLOUR, linewidth=1.0,
               linestyle=':', alpha=0.6, zorder=2, xmax=0.87)


    # ── 8. Legend ─────────────────────────────────────────────────────────────

    femur_base_patch = mpatches.Patch(
        facecolor=FEMUR_COLOUR, alpha=0.80, edgecolor='none',
        label='Femur — whole mask density',
    )
    femur_adj_patch = mpatches.Patch(
        facecolor=FEMUR_COLOUR, alpha=0.30, edgecolor=FEMUR_COLOUR,
        hatch='////', label='Femur — parenchyma-only uplift',
    )
    hip_base_patch = mpatches.Patch(
        facecolor=HIP_COLOUR, alpha=0.80, edgecolor='none',
        label='Hip — whole mask density',
    )
    hip_adj_patch = mpatches.Patch(
        facecolor=HIP_COLOUR, alpha=0.30, edgecolor=HIP_COLOUR,
        hatch='////', label='Hip — parenchyma-only uplift',
    )

    ax.legend(
        handles  = [femur_base_patch, femur_adj_patch,
                    hip_base_patch,   hip_adj_patch],
        fontsize = fontsize_base,
        frameon  = False,
        loc      = 'upper left',
        ncol     = 2,
    )

    # ── 9. Axes formatting ────────────────────────────────────────────────────

    ax.set_xlim(x_min, x_max*1.15)
    ax.set_ylim(0, bracket_y*1.4)
    ax.set_xticks([])
    ax.set_ylabel(f'{y_label}\n(normalised to femur baseline per mouse)',
                  fontsize=fontsize_base*0.9)
    ax.tick_params(axis='y', labelsize=fontsize_base)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_title(f'{title}\n{sig_str}', fontsize=fontsize_base*1.2, pad=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
    else:
        plt.show()

    return fig, ax


if __name__ == '__main__':

    from utils import Folder 

    folder = Folder('data')
    dep = 'cell_density_whole_mask'

    # plot_per_mouse_boxplots(
    #     target_directory  = folder,
    #     dependent         = dep,
    #     adjustment_factor = 1E6,          # µm^-2 → mm^-2
    #     y_label           = 'CXCL12 cell density (cells / mm²)', #dep.replace('_', ' '), 
    #     title             = f'Total CXCL12 cell density within endosteal boundary — femur vs hip',
    #     save_path         = f'figures/fig_{dep}.png',
    # )

    # plot_2d_scatter(
    #     target_directory  = folder,
    #     x                 = 'cell_density_whole_mask',
    #     y                 = 'frac_density_change',
    #     adjustment_factor = 1.0,
    #     x_label           = None,
    #     y_label           = None,
    #     title             = None,
    #     min_windows       = 3,
    #     ignore_yellows    = None,
    #     figsize           = (11, 5),
    #     save_path         = None,
    # )

    # plot_density_adjustment_bars(
    #     target_directory  = folder,
    #     adjustment_factor = 1.0,
    #     y_label           = 'CXCL12 cell density  (cells / mm²)',
    #     title             = 'Effect of non-parenchymal area removal on reported density',
    #     min_windows       = 3,
    #     ignore_yellows    = None,
    #     figsize           = (11, 5),
    #     save_path         = 'figures/fig_density_increase.png',
    # )

    plot_two_column_scatter(
        target_directory  = folder,
        dependent         = dep,
        adjustment_factor = 1E6,
        normalise_to_femur= False,
        y_label           = dep.replace('_', ' '),
        title             = None,
        min_windows       = 3,
        ignore_yellows    = None,
        figsize           = (6, 5),
        font_size         = 16,
        save_path         = f'figures/small_fig_{dep}.png',
    )