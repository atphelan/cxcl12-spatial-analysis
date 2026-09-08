import numpy as np 
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.formula.api import ols
from scipy.stats import f_oneway

from utils import Folder

def aggregate_and_analyse_resampling_files_normalised(target_directory, dependent, print_tile_count=False, adjustment_factor=1.0, rounding=1):
    print("-"*80)
    print("")
    print("Beginning statistical analysis for", f"{dependent}.  ")
    target_directory = Folder(target_directory)
    df_list = []
    for file in target_directory.files:
        if file.type == 'csv':
            df = file.read()
            df['mouse'] = df['name'].apply(lambda name: name[0:2])
            df['bone'] = df['name'].apply(lambda name: name.split('_')[0][2:])
            if print_tile_count:
                print(f'Number of tiles from {file.name}: ', len(df[df['ratio_reconstructed_mask'] > 0.18]), '  ')
            if dependent == 'hole_coverage':
                df_list.append(df[df['ratio_reconstructed_mask'] > 0.18].copy())
            else:
                df_list.append(df.copy())
    print('')
    final_df = pd.concat(df_list)

    # Aggregate to one value per mouse per bone (area-weighted mean across tiles)
    aggregated = final_df.groupby(['mouse', 'bone']).apply(
        lambda g: pd.Series({
            dependent: (g[dependent] * g['filled_area']).sum() / g['filled_area'].sum(),
            'n_windows': len(g)
        }), include_groups=False
    ).reset_index()

    # Mixed-effects model on aggregated per-mouse values
    # With only one observation per mouse per bone, mouse is the correct
    # grouping variable — this is now equivalent in structure to the
    # paired t-test but expressed as a linear mixed model
    lmm = smf.mixedlm(
        f'{dependent} ~ C(bone)',
        data=aggregated,
        groups=aggregated['mouse'],
    )
    result = lmm.fit(reml=True)

    print('')
    print('Results: fitted parameters  ')
    print(result.params)
    print('Relative effect:', round(100*result.params['C(bone)[T.hip]']/result.params['Intercept'], 1), '%')
    print('')
    print('Results: confidence intervals  ')
    print(result.conf_int(alpha=0.05))
    print('')

    # Direction of hip vs femur per mouse
    print('Percentage differences per mouse:  ')
    for mouse in ['M1','M2','M3','M4','M5']:
        sub = aggregated[aggregated['mouse'] == mouse]
        hip   = sub[sub['bone'] == 'hip'][dependent].values[0]
        femur = sub[sub['bone'] == 'femur'][dependent].values[0]
        direction = 'hip > femur' if hip > femur else 'femur > hip'
        stats = f"({round(femur*adjustment_factor, rounding)} (f), {round(hip*adjustment_factor, rounding)} (h))"
        print(f"{mouse}: {direction} {stats} ({abs(hip-femur)/femur*100:.1f}% difference)  ")
    print('')

def aggregate_and_analyse_resampling_files(target_directory, dependent, print_tile_count=False, adjustment_factor=1.0, rounding=1):
    print("-"*80)
    print("")
    print("Beginning statistical analysis for", f"{dependent}.  ")
    target_directory = Folder(target_directory)
    df_list = []
    for file in target_directory.files:
        if file.type == 'csv':
            df = file.read()
            df['mouse'] = df['name'].apply(lambda name: name[0:2])
            df['bone'] = df['name'].apply(lambda name: name.split('_')[0][2:])
            if print_tile_count:
                print(f'Number of tiles from {file.name}: ', len(df[df['ratio_reconstructed_mask'] > 0.18]), '  ')
            if dependent == 'hole_coverage':
                df_list.append(df[df['ratio_reconstructed_mask'] > 0.18].copy())
            else:
                df_list.append(df.copy())
            # df_list.append(pd.DataFrame([{'mouse': df['mouse'].iloc[0], 'bone': df['bone'].iloc[0], dependent: (df[dependent]*df['filled_area']).sum()/df['filled_area'].sum()}]))
    print('')
    final_df = pd.concat(df_list)
    aggregated = final_df.groupby(['mouse', 'bone']).apply(
        lambda g: pd.Series({
            dependent: (g[dependent] * g['filled_area']).sum() / g['filled_area'].sum(),
            'n_windows': len(g)
        }), include_groups=False
    ).reset_index()

    final_df['image_id'] = final_df['mouse'] + '_' + final_df['bone']



    lmm = smf.mixedlm(
        f'{dependent} ~ C(bone)',
        data=final_df,
        groups=final_df['image_id'],
    )
    result = lmm.fit()

    print('')

    print('Results: fitted parameters  ')
    print(result.params)
    print('Relative effect:', round(100*result.params['C(bone)[T.hip]']/result.params['Intercept'], 1), '%')
    print('')

    print('Results: confidence intervals  ')
    print(result.conf_int(alpha=0.05))
    print('')

    # Direction of hip vs femur per mouse
    print('Percentage differences per mouse:  ')
    for mouse in ['M1','M2','M3','M4','M5']:
        sub = aggregated[aggregated['mouse'] == mouse]
        hip   = sub[sub['bone'] == 'hip'][dependent].values[0]
        femur = sub[sub['bone'] == 'femur'][dependent].values[0]
        direction = 'hip > femur' if hip > femur else 'femur > hip'
        stats = f"({round(femur*adjustment_factor, rounding)} (f), {round(hip*adjustment_factor, rounding)} (h))"
        print(f"{mouse}: {direction} {stats} ({abs(hip-femur)/femur*100:.1f}% difference)  ")
    print('')

# aggregate_and_analyse_resampling_files_normalised('data', 'cell_density', print_tile_count=False, adjustment_factor=1E6, rounding=1)
aggregate_and_analyse_resampling_files_normalised('data', 'cell_density_whole_mask', adjustment_factor=1E6, rounding=1)
# aggregate_and_analyse_resampling_files_normalised('data', 'frac_dense_area', rounding=3)
# aggregate_and_analyse_resampling_files_normalised('data', 'frac_sparse_area', rounding=3)
# aggregate_and_analyse_resampling_files_normalised('data', 'frac_hole_area', rounding=3)
# aggregate_and_analyse_resampling_files_normalised('data', 'frac_density_change', rounding=3)

