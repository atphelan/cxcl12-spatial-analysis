import numpy as np 
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

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



target_file = 'mask particle csvs from Clara/hole_densities_55.0_60.0.csv'
df = pd.read_csv(target_file)
df['mouse_number'] = df['name'].apply(lambda s: s[0:2])
df['bone_type'] = df['name'].apply(lambda s: s.split('_')[0][2:])
df.to_csv(target_file)
df['dilation'] = df['name'].apply(lambda s: int(s.split('_')[2]))
df1 = df.loc[df['dilation']==0]
df2 = df.loc[df['dilation']==10].dropna()

dep = "hole_coverage"
indep1, indep2 = "bone_type", "mouse_number"

ols_fun = f"{dep} ~ C({indep1}) + C({indep2}) + C({indep1}):C({indep2})"
model1 = ols(ols_fun, data=df1).fit()
model2 = ols(ols_fun, data=df2).fit()
anova_result1 = sm.stats.anova_lm(model1, type=2)
anova_result2 = sm.stats.anova_lm(model2, type=2)

ss_total1 = anova_result1['sum_sq'].sum()
ss_total2 = anova_result2['sum_sq'].sum()

anova_result1['eta_sq'] = (
    anova_result1['sum_sq'] / ss_total1
)
anova_result2['eta_sq'] = (
    anova_result2['sum_sq'] / ss_total2
)

print("Hole coverage ANOVA results undilated")
print(anova_result1) # If eta_sq << 1, not much difference explained. Go for 0.1+.
print("Hole coverage ANOVA results dilated 10x")
print(anova_result2) # If eta_sq << 1, not much difference explained. Go for 0.1+.

def aggregate_and_analyse_resampling_files(target_directory):
    target_directory = Folder(target_directory)
    df_list = []
    for file in target_directory.files:
        if file.type == 'csv' and 'resampled_analysed' in file.name:
            df = file.read()
            df['source'] = file.name
            df['bone_type'] = df['source'].apply('Not a function yet')
            df_list.append(df.copy())
    final_df = pd.concat(df_list)
    final_df.to_csv(target_directory + '2way_anova.csv')