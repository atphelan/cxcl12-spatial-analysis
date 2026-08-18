# cxcl12-spatial-analysis
*Code tools to analyse spatial distribution of CXCL12 cells in bone marrow images.*

**Results of latest run:**

Using 5 matched hip/femur pairs.

  M1hip_results_tiled.csv  <-->  M1femur_results_tiled.csv  
  M2hip_results_tiled.csv  <-->  M2femur_results_tiled.csv  
  M3hip_results_tiled.csv  <-->  M3femur_results_tiled.csv  
  M4hip_results_tiled.csv  <-->  M4femur_results_tiled.csv  
  M5hip_results_tiled.csv  <-->  M5femur_results_tiled.csv  
  
  
#PAIRED STATISTICAL TESTS — hip vs femur  (n=5 pairs)

| Metric | Hip mean |Femur mean |Normal? | Test   |  p |  Sig |  
|---|---|---|---|---|---|---|  
|cell_density           |   0.0006014 |   0.0009236 |     yes |  paired_t |  0.0491 |    * |  
|frac_dense_area        |      0.1283 |      0.1305 |     yes |  paired_t |  0.8906 |   ns |  
|frac_sparse_area       |      0.7649 |      0.7139 |     yes |  paired_t |  0.3324 |   ns |  
|frac_hole_area         |      0.1069 |      0.1556 |     yes |  paired_t |  0.2937 |   ns |  
|ratio_high_low         |      0.2207 |      0.1673 |     yes |  paired_t |  0.1572 |   ns | 

--------------------------------------------------------------------------------

Number of tiles from M1hip_results_save states.csv:  5   
Number of tiles from M1femur_results_save states.csv:  7   
Number of tiles from M2hip_results_save states.csv:  22   
Number of tiles from M2femur_results_save states.csv:  9   
Number of tiles from M3hip_results_save states.csv:  16    
Number of tiles from M3femur_results_save states.csv:  7   
Number of tiles from M4hip_results_save states.csv:  9   
Number of tiles from M4femur_results_save states.csv:  13   
Number of tiles from M5hip_results_save states.csv:  16   
Number of tiles from M5femur_results_save states.csv:  8   
 
--------------------------------------------------------------------------------

Beginning statistical analysis for cell_density

Results: fitted parameters  
Intercept         0.000922  
C(bone)[T.hip]   -0.000323  
Group Var         0.610787  
dtype: float64  
Relative effect: -35.0 %

Results: confidence intervals  
                       0         1
Intercept       0.000739  0.001105  
C(bone)[T.hip] -0.000579 -0.000066  
Group Var      -0.121235  1.342809  

Percentage differences per mouse per mm^2:  
M1: femur > hip (650.2 (f), 587.9 (h)) (9.6% difference)  
M2: femur > hip (900.0 (f), 386.1 (h)) (57.1% difference)  
M3: femur > hip (1045.9 (f), 624.7 (h)) (40.3% difference)  
M4: femur > hip (777.9 (f), 673.1 (h)) (13.5% difference)  
M5: femur > hip (1227.3 (f), 724.2 (h)) (41.0% difference)  

--------------------------------------------------------------------------------

Beginning statistical analysis for frac_dense_area.  

Results: fitted parameters  
Intercept         0.131731  
C(bone)[T.hip]   -0.005527  
Group Var         0.096766  
dtype: float64  
Relative effect: -4.2 %

Results: confidence intervals  
                       0         1  
Intercept       0.110035  0.153428  
C(bone)[T.hip] -0.035112  0.024059  
Group Var      -0.089153  0.282685  

Percentage differences per mouse:  
M1: hip > femur (0.096 (f), 0.138 (h)) (43.2% difference)  
M2: femur > hip (0.114 (f), 0.096 (h)) (16.2% difference)  
M3: hip > femur (0.132 (f), 0.142 (h)) (7.8% difference)  
M4: femur > hip (0.149 (f), 0.116 (h)) (22.3% difference)  
M5: femur > hip (0.151 (f), 0.119 (h)) (21.5% difference)  

--------------------------------------------------------------------------------

Beginning statistical analysis for frac_sparse_area.  

Results: fitted parameters  
Intercept         0.716956
C(bone)[T.hip]    0.054290
Group Var         0.382039
dtype: float64
Relative effect: 7.6 %

Results: confidence intervals  
                       0         1
Intercept       0.648225  0.785686
C(bone)[T.hip] -0.041472  0.150052
Group Var      -0.135508  0.899586

Percentage differences per mouse:  
M1: femur > hip (0.71 (f), 0.646 (h)) (9.0% difference)  
M2: hip > femur (0.76 (f), 0.849 (h)) (11.7% difference)  
M3: hip > femur (0.612 (f), 0.772 (h)) (26.1% difference)  
M4: hip > femur (0.811 (f), 0.815 (h)) (0.5% difference)  
M5: hip > femur (0.691 (f), 0.81 (h)) (17.1% difference)  

--------------------------------------------------------------------------------

Beginning statistical analysis for frac_hole_area.  

Results: fitted parameters  
Intercept         0.151705  
C(bone)[T.hip]   -0.050822  
Group Var         0.340663  
dtype: float64  
Relative effect: -33.5 %

Results: confidence intervals  
                       0         1  
Intercept       0.086879  0.216532  
C(bone)[T.hip] -0.140892  0.039249  
Group Var      -0.135439  0.816764  

Percentage differences per mouse:  
M1: hip > femur (0.194 (f), 0.217 (h)) (11.6% difference)  
M2: femur > hip (0.125 (f), 0.055 (h)) (56.4% difference)  
M3: femur > hip (0.256 (f), 0.086 (h)) (66.4% difference)  
M4: hip > femur (0.039 (f), 0.069 (h)) (74.1% difference)  
M5: femur > hip (0.157 (f), 0.072 (h)) (54.4% difference)  