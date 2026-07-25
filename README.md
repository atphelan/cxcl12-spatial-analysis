# cxcl12-spatial-analysis
*Code tools to analyse spatial distribution of CXCL12 cells in bone marrow images.*

**Results of latest run:**

Using 5 matched hip/femur pairs.

  M2hip_results_tiled.csv  <-->  M2femur_results_tiled.csv  
  M3hip_results_tiled.csv  <-->  M3femur_results_tiled.csv  
  M5hip_results_tiled.csv  <-->  M5femur_results_tiled.csv  
  M4hip_results_tiled.csv  <-->  M4femur_results_tiled.csv  
  M1hip_results_tiled.csv  <-->  M1femur_results_tiled.csv  


#PAIRED STATISTICAL TESTS — hip vs femur  (n=5 pairs)

| Metric | Hip mean |Femur mean |Normal? | Test   |  p |  Sig |  
|---|---|---|---|---|---|---|
|mean_hole_area         |        5686    |     7678   |   yes |  paired_t |  0.5127  |  ns |  
|std_hole_area           |       3570     |    4219   |   yes  | paired_t |  0.6084  |  ns |  
|cell_density             | 0.0005437  |  0.0009209   |   yes  | paired_t  | 0.0120  |   * |  
|frac_high_density_tissue  |     0.2818   |    0.2586  |    yes |  paired_t |  0.7168  |  ns |  
|frac_low_density_tissue    |   0.5665   |    0.5137    |  yes |  paired_t |  0.3509  |  ns |  
|frac_gap_tissue            |  0.1517    |   0.2277    |  yes |  paired_t |  0.0576  |  ns |  
|ratio_high_low             |  0.6272    |   0.5326    |   no  | wilcoxon |  0.8125  |  ns |  


Significance: * p<0.05   ** p<0.01   *** p<0.001   ns = not significant under this test
Note: Shapiro-Wilk has low power at small n — consider Wilcoxon as default.


Number of tiles from M2hip_results_tiled.csv:  27  
Number of tiles from M3hip_results_tiled.csv:  13  
Number of tiles from M2femur_results_tiled.csv:  9  
Number of tiles from M5femur_results_tiled.csv:  8  
Number of tiles from M5hip_results_tiled.csv:  20  
Number of tiles from M1femur_results_tiled.csv:  9  
Number of tiles from M4hip_results_tiled.csv:  5  
Number of tiles from M4femur_results_tiled.csv:  7  
Number of tiles from M1hip_results_tiled.csv:  6  
Number of tiles from M3femur_results_tiled.csv:  7  


Results: fitted parameters  
Intercept         0.000920  
C(bone)[T.hip]   -0.000374  
Relative slope    -0.407  
average reduction in CXCL12 cell density of 40.7% in a hip compared to the same mouse's femur  
Group Var         1.012031  

Results: 95% confidence intervals 
                Lower     Upper  
Intercept       0.000724  0.001116  
C(bone)[T.hip] -0.000650 -0.000099  
Group Var      -0.125197  2.149259  

Percentage differences per mouse:  
M1: femur > hip (24.3% difference) (low evidence on hip, only 6 samples)  
M2: femur > hip (58.8% difference)  
M3: femur > hip (35.0% difference)  
M4: femur > hip (47.1% difference) (low evidence on hip, only 5 samples)  
M5: femur > hip (41.8% difference)  