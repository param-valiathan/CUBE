# CUBE DLC-output comparison — summary report

Stage 1 (DLC tracking output only). CUBE-blue series indicate CUBE's own DLC run; all other series are published/reference tracking data.

## elevated_plus_maze
- Only one source available for this paradigm — distribution plots generated, no cross-source statistics yet.

## forced_swim_test
- Only one source available for this paradigm — distribution plots generated, no cross-source statistics yet.

## novel_object
### mean_likelihood
- Omnibus (mann_whitney_u): stat=387.000, p=0.8339
- Levene's test (variance homogeneity): stat=0.806, p=0.3732 (variances similar)
- Coefficient of variation per source: patterns2025_batch1=0.003, patterns2025_batch2=0.002
- No pairwise differences survived FDR correction.

### interp_rate
- Omnibus (mann_whitney_u): stat=439.500, p=0.2267
- Levene's test (variance homogeneity): stat=4.486, p=0.03879 (variances differ)
- Coefficient of variation per source: patterns2025_batch1=2.560, patterns2025_batch2=3.750
- No pairwise differences survived FDR correction.

### mean_velocity_px_s
- Omnibus (mann_whitney_u): stat=311.000, p=0.2944
- Levene's test (variance homogeneity): stat=1.722, p=0.195 (variances similar)
- Coefficient of variation per source: patterns2025_batch1=0.251, patterns2025_batch2=0.293
- No pairwise differences survived FDR correction.

### mean_jitter
- Omnibus (mann_whitney_u): stat=252.000, p=0.04151
- Levene's test (variance homogeneity): stat=0.405, p=0.5271 (variances similar)
- Coefficient of variation per source: patterns2025_batch1=0.191, patterns2025_batch2=0.208
- Significant pairwise differences (FDR<0.05), largest effect first:
  - patterns2025_batch1 vs patterns2025_batch2: Mann-Whitney p_fdr=0.04151, Cliff's delta=-0.326, Cohen's d=-0.490

### bbox_area
- Omnibus (mann_whitney_u): stat=508.000, p=0.02511
- Levene's test (variance homogeneity): stat=1.921, p=0.1715 (variances similar)
- Coefficient of variation per source: patterns2025_batch1=0.057, patterns2025_batch2=0.037
- Significant pairwise differences (FDR<0.05), largest effect first:
  - patterns2025_batch1 vs patterns2025_batch2: Mann-Whitney p_fdr=0.02511, Cliff's delta=0.358, Cohen's d=0.488


## open_field
- Cross-source statistics skipped: too few sessions (n<3 required per group) in: bsoid_demo (n=2). Distribution plots for this paradigm were still generated.
