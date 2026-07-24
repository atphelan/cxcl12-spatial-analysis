import numpy as np 
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.formula.api import ols
from scipy.stats import f_oneway

from utils import Folder

# target_file = 'mask tif files from Clara/cell_measurements.csv'
# df = pd.read_csv(target_file)
# # df['mouse_number'] = df['image'].apply(lambda s: s[0:2])
# # df['bone_type'] = df['image'].apply(lambda s: s.split('_')[0][2:])
# # df.to_csv(target_file)
# df['dilation'] = df['image'].apply(lambda s: int(s.split('_')[2]))
# df = df.loc[df['dilation']==0]

# dep = "aspect_ratio"
# indep1, indep2 = "bone_type", "mouse_number"

# ols_fun = f"{dep} ~ C({indep1}) + C({indep2}) + C({indep1}):C({indep2})"
# model = ols(ols_fun, data=df).fit()
# anova_result = sm.stats.anova_lm(model, type=2)

# ss_total = anova_result['sum_sq'].sum()

# anova_result['eta_sq'] = (
#     anova_result['sum_sq'] / ss_total
# )

# print("Cell aspect ratio ANOVA results")
# print(anova_result) # If eta_sq << 1, not much difference explained. Go for 0.1+.



# target_file = 'mask particle csvs from Clara/hole_densities_55.0_60.0.csv'
# df = pd.read_csv(target_file)
# df['mouse_number'] = df['name'].apply(lambda s: s[0:2])
# df['bone_type'] = df['name'].apply(lambda s: s.split('_')[0][2:])
# df.to_csv(target_file)
# df['dilation'] = df['name'].apply(lambda s: int(s.split('_')[2]))
# df1 = df.loc[df['dilation']==0]
# df2 = df.loc[df['dilation']==10].dropna()

# dep = "hole_coverage"
# indep1, indep2 = "bone_type", "mouse_number"

# ols_fun = f"{dep} ~ C({indep1}) + C({indep2}) + C({indep1}):C({indep2})"
# model1 = ols(ols_fun, data=df1).fit()
# model2 = ols(ols_fun, data=df2).fit()
# anova_result1 = sm.stats.anova_lm(model1, type=2)
# anova_result2 = sm.stats.anova_lm(model2, type=2)

# ss_total1 = anova_result1['sum_sq'].sum()
# ss_total2 = anova_result2['sum_sq'].sum()

# anova_result1['eta_sq'] = (
#     anova_result1['sum_sq'] / ss_total1
# )
# anova_result2['eta_sq'] = (
#     anova_result2['sum_sq'] / ss_total2
# )

# print("Hole coverage ANOVA results undilated")
# print(anova_result1) # If eta_sq << 1, not much difference explained. Go for 0.1+.
# print("Hole coverage ANOVA results dilated 10x")
# print(anova_result2) # If eta_sq << 1, not much difference explained. Go for 0.1+.

def aggregate_and_analyse_resampling_files(target_directory, dependent):
    target_directory = Folder(target_directory)
    df_list = []
    for file in target_directory.files:
        if file.type == 'csv':
            df = file.read()
            df['mouse'] = df['name'].apply(lambda name: name[0:2])
            df['bone'] = df['name'].apply(lambda name: name.split('_')[0][2:])
            df_list.append(df.copy())
            # df_list.append(pd.DataFrame([{'mouse': df['mouse'].iloc[0], 'bone': df['bone'].iloc[0], dependent: (df[dependent]*df['filled_area']).sum()/df['filled_area'].sum()}]))
    final_df = pd.concat(df_list)
    aggregated = final_df.groupby(['mouse', 'bone']).apply(
        lambda g: pd.Series({
            dependent: (g[dependent] * g['filled_area']).sum() / g['filled_area'].sum(),
            'n_windows': len(g)
        })
    ).reset_index()


    final_df['image_id'] = final_df['mouse'] + '_' + final_df['bone']



    lmm = smf.mixedlm(
        f'{dependent} ~ C(bone)',
        data=final_df,
        groups=final_df['image_id'],
    )
    result = lmm.fit()
    print('Results: fitted parameters')
    print(result.params)
    print('')

    print('Results: confidence intervals')
    print(result.conf_int())
    print('')

    # Direction of hip vs femur per mouse
    print('Percentage differences per mouse:')
    for mouse in ['M1','M2','M3','M4','M5']:
        sub = aggregated[aggregated['mouse'] == mouse]
        hip   = sub[sub['bone'] == 'hip'][dependent].values[0]
        femur = sub[sub['bone'] == 'femur'][dependent].values[0]
        direction = 'hip > femur' if hip > femur else 'femur > hip'
        print(f"{mouse}: {direction} ({abs(hip-femur)/femur*100:.1f}% difference)")

aggregate_and_analyse_resampling_files('data', 'cell_density')