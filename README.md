# cxcl12-spatial-analysis
*Code tools and analysis statistics of spatial distribution of CXCL12 cells in bone marrow images.*

**Results described as of latest analysis 26-08-2026:**

# CXCL12 Spatial Analysis — Results Summary

## Overview

Image analysis of fluorescently labelled bone marrow sections from 5 matched  
hip/femur pairs in mice expressing CXCL12-GFP. Sections were additionally  
stained with DAPI to reveal nucleated cells. The analysis pipeline  
identifies and counts CXCL12+ stromal cells based on identifiable bodies of  
a sufficient size (3um < diameter < 20um) in a user-defined binary image file,  
computes their 2D density relative to a user-defined tissue mask, and  
characterises the spatial organisation of the marrow microenvironment using  
a polygon classification approach based on a nearest-neighbour cell network.  

Cell counts were carefully validated to exclude inflated counts arising from  
reticular filaments of well-exposed CAR cells registering as multiple objects  
per real cell body, or conversely diminished counts due to excessive inclusion  
of non-parenchymal tissue area. For images M2hip and M5femur, tissue area was  
instead calculated from a polygon reconstruction of the tissue area due to a  
number of dim regions in the CXCL12-GFP signal resulting in falsely low  
reported density due to imaging artefact.  

---

## Files

clustering_analysis.py - statistical summary of the analysed files
2way_anova.py - specific testing of mouse number and bone type on one variable at a time
main.py - core analysis pipeline
summary_plotting.py - final plots
utils.py, __init__.py - io utils
data/* - outputs from analysis pipeline
figures/* - analysis output visualisations per sample

## Key Finding

**Femur bone marrow contains a significantly higher density of CXCL12+
stromal cells than hip bone marrow per cross sectional area of bone** (paired t-test, p = 0.0028, **).

A mixed-effects linear regression estimated a mean reduction of **19.1%** in
CXCL12 cell density in hip relative to femur, consistent in direction across
all five animals. Mean densities were:

| Bone   | Mean cell density (cells / mm²) |
|--------|----------------------------------|
| Femur  | 524.9                            |
| Hip    | 424.5                            |

### Per-Mouse Breakdown

| Mouse (sex, age) | Femur (cells/mm²) | Hip (cells/mm²) | Difference |
|------------------|-------------------|-----------------|------------|
| M1 (f, 38w)      | 544.6             | 414.9           | −23.8%     |
| M2 (m, 32w)      | 558.6             | 457.5           | −18.1%     |
| M3 (m, 33w)      | 525.1             | 441.8           | −15.9%     |
| M4 (m, 41w)      | 598.0             | 445.3           | −25.5%     |
| M5 (m, 30w)      | 491.0             | 439.8           | −10.4%     |

The effect is consistent in direction across all mice, with individual
differences ranging from 10.4% (M5) to 25.5% (M4).

### Mixed-Effects Model Parameters

Intercept (femur mean): 0.000533  
Slope (femur -> hip): -0.000107  
Group Var: 0.000901  


95% Confidence intervals:  

|                    | Lower     | Upper     |
|--------------------|-----------|-----------|
| Intercept          | 0.000477  | 0.000589  |
| C(bone)[T.hip]     | −0.000182 | −0.000033 |

The confidence interval on the hip effect does not cross zero, confirming
the significance of the bone-type effect on cell density.

---

## Spatial Organisation of Marrow Tissue

Polygons formed between connected CXCL12+ cell network nodes were classified
into four categories based on GFP and DAPI signal within each polygon:

- **Dense**: small GFP+ polygons (area < 500 um^2, max length < 30 um),
  indicating tightly packed CXCL12+ neighbourhoods
- **Sparse**: larger GFP+ polygons, indicating wider CXCL12+ cell spacing
  within functional parenchyma
- **Holes**: DAPI+ but GFP− polygons, representing dark patches in the
  CXCL12-rich parenchyma (e.g. sinusoids, adipocytes, or other
  non-CXCL12-producing tissue)
- **Removed**: double-negative polygons or non-tissue regions excluded from
  area calculations

### Full Statistical Summary (n = 5 pairs)

| Metric                  | Hip mean | Femur mean | Test      | p      | Sig |
|-------------------------|----------|------------|-----------|--------|-----|
| cell_density_whole_mask | 0.000425 | 0.000525   | paired t  | 0.0042 | **  |
| cell_density            | 0.000588 | 0.000580   | paired_t  | 0.8756 | ns  |
| frac_density_change     | 0.5538   | 0.2066     | paired_t  | 0.0382 | *   |  
| frac_hole_area          | 0.1676   | 0.1499     | paired_t  | 0.5637 | ns  |  
| frac_dense_area         | 0.1239   | 0.1131     | paired_t  | 0.4983 | ns  |  
| frac_sparse_area        | 0.7085   | 0.737      | paired_t  | 0.3125 | ns  |  
| ratio_high_low          | 0.1877   | 0.1487     | paired t  | 0.0864 | ns  |

All fractions are relative to reconstructed tissue area. Normality was
assessed prior to test selection; prop_removed_area was non-normal and
tested with a Wilcoxon signed-rank test.

### Interpretation of Spatial Results

Despite the clear difference in overall cell density, **no significant
differences were found in the fractional areas of sparse, hole, or removed
tissue** between bone types. This suggests that the lower CXCL12 cell density
in hip is not explained by a greater physical displacement of parenchyma by
non-CXCL12-producing structures (sinusoids, adipocytes, tears), nor by a
larger proportion of tissue being devoid of marrow altogether.

However, **ratio_high_low was close to significance, with the ratio higher in hip than femur**
(p = 0.0864, *), indicating that hip marrow could have a moderately higher
proportion of its GFP+ tissue area in dense CXCL12+ neighbourhoods relative
to sparse neighbourhoods, compared to femur. This is a counterintuitive
finding given the lower overall density in hip, and warrants further
investigation with a larger sample. Hip marrows are thinner than femurs, 
with a greater proportion of the space close to bone tissue - it is not 
expected to find as many CAR cells in the endosteum compared to the interior
perisinusoidal region except in aged organisms - further work could look at 
younger adult mice to see if the differences by bone type are more pronounced
than in aged mice.  

---

*Analysis performed using custom Python pipeline, all files in this repository. Last updated: 26th August, 2026.*

