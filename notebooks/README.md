# Notebooks

Run in order; each opens with the paper section it reproduces.

| Notebook | Paper | Runtime (CPU) |
|---|---|---|
| `01_state_generation.ipynb` | Sec. II A, Table I | seconds |
| `02_entanglement_measures.ipynb` | Sec. II B, Table II; Sec. V B, Table VIII | seconds |
| `03_feature_reduction_anova.ipynb` | Sec. III C, IV A, Table III | seconds |
| `04_slocc_ann.ipynb` | Sec. III A, IV B, Table IV, Fig. 3 | ~1 min (demo) + reads `results/` |
| `05_gme_ann.ipynb` | Sec. IV C, Table V | ~1 min (demo) + reads `results/` |
| `06_svm_knn_benchmark.ipynb` | Sec. V A, Fig. 4, Tables VI–VII | seconds |
| `07_experimental_states.ipynb` | Sec. III D, Fig. 2 | seconds |

## `00_original/`: the research notebooks as they were (outputs kept)

Prefixed A–H in chronological and logical order; the original file name follows `__`. Local Windows paths were
replaced with `<LOCAL_DATA_DIR>`, so these notebooks do not run as-is. They are kept as the record of the work.

| File | Original name | Content |
|---|---|---|
| `A_simulated_state_analysis__…` | `ENT-SIM-DATA-ANALYSIS.ipynb` (Colab) | `*_ASD` state generators, rank checks, tangles, witness classifier → `states.py`, `checks.py`, `measures.py` |
| `B_experimental_state_analysis__…` | `ENT-EXP-DATA-ANALYSIS.ipynb` (Colab) | the experimental and theoretical matrices (→ `data/raw/`), negativity, entropies, witnesses |
| `C_pca_knn_experimental__…` | `ENT-PCA.ipynb` (Colab) | PCA and KNN on the experimental states (exploratory) |
| `D_training_data_generation__…` | `ANN-ENT-Data-Gen.ipynb` | random-state training-data generation |
| `E_2022-11-12_ann_exploration__…` | `new_data_ann_12_11_2022.ipynb` | first Keras ANN experiments |
| `F_2022-11-20_training_data_assembly__…` | `ann_training_data_20_11_2022.ipynb` | merging the per-class training CSVs |
| `G_2022-11-20_entropy_3tangle__…` | `new_data_tangle_20_11_2022.ipynb` | reduced-state ranks and 3-tangle for the experimental states |
| `H_2022-12-19_ann_slocc__…` | `new_data_ann_19_12_2022.ipynb` | SLOCC ANN trained on simulated data, tested on the experimental states |
