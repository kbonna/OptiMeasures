#### OptiMeasures
The choice of parameters (connectivity measure, parcellation, connection threshold...) in generating functional connectivity matrices is an important issue. These parameters are often chosen arbitralily or based on their popularity in papers.
The rough goal of this project is to find the parameters for generating functional connectivity matrices that give the best stability of global graph measures (modularity, global efficiency, transitivity etc.) across time and allows the best subject identifiability.

#### Usage
1. Preprocess your fMRI-data using fMRIprep (use fMRIprep with ICA-AROMA option).
2. Generate functional connectivity for various sets of parameters using generate_fc_matrices script.
3. Calculate global measures for each connectivity matrix.

#### Required packages to generate FC matrices:
nilearn, nibabel, scipy, numpy, os, pandas
