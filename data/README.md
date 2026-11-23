# Data

All density matrices are 8×8, basis order |q1 q2 q3⟩ = |000⟩, |001⟩, …, |111⟩ (index 0–7), stored as
`Re_rho_ab` / `Im_rho_ab` columns (row `a`, column `b`).

## `raw/` — used by the code

| File | Rows | What | Provenance |
|---|---|---|---|
| `experimental_states_3q.csv` | 34 | Experimentally reconstructed density matrices with `state_id`, `class`, `class_code` (0 SEP, 1 BS1, 2 BS2, 3 BS3, 4 W, 5 GHZ), `fidelity`, `used_in_paper` | Extracted verbatim from the matrices hard-coded in the original analysis notebook (`notebooks/00_original/B_experimental_state_analysis__…ipynb`, cell 2). The matrices were reconstructed by full state tomography (`mathematica/tomography/`) on three ¹⁹F spins of trifluoroiodoethylene at the NMR Research Facility, IISER Mohali. 30 rows have `used_in_paper = True` (5 per class; fidelities match Table VIII). The 4 rows with `used_in_paper = False` are lower-fidelity repeats (F = 0.7646, 0.8913, 0.9612, 0.9246) removed by the original `bad_fid` filter. |
| `theoretical_targets_3q.csv` | 30 | Ideal target state for each of the 30 experimental states, same order | Same notebook, cell 3. |
| `source/Final_exp_states_three_qubit.docx` | — | Bar-plot images of the 30 experimental density matrices (as in Fig. 2 d–e) | Shared by V. Gulati. |

Two tomography outputs present in the original notebook, `pps_output` (pseudo-pure state) and `Random_bs1_1_output`,
were never assigned a class there and are not included.

**Labelling note.** The state `random_bs2_1_output` is labelled **BS1** (its theoretical target is a BS1 state,
correlation-tensor ranks (1,3,3)); it is BS1₄ (F = 0.9731) in Table VIII. The `bs2` in the file name is the
tomography run name, not its class.

## `legacy/` — intermediate files from the original pipeline (Nov 2022 – Jan 2023)

Kept for provenance; the code does not read them.

| File | What |
|---|---|
| `new_data_entropy_3tangle_20_11_2022.csv` | 35 experimental states (incl. the low-fidelity repeats and the pseudo-pure state) with one-hot class, `CODE`, `FIDELITY`, reduced-state determinants and ranks, separability checks and the 3-tangle (`00_original/G_…`) |
| `ANN_PCA_DATASET_new_data.csv` | The same 35 states flattened to the 128-feature layout used by the ANN notebooks, with one-hot class, `CODE`, `FIDELITY` |
| `ann_pred_04_01_2023.csv` | ANN class probabilities for the 30 states (early model, Jan 2023) |
| `final_pred_04_01_2023.csv` | Per-state summary: class code, fidelity, witness expectation values `exp_O…O3`, ANN class probabilities, correctness |

## Not included (regenerable)

| Original file | Size | Regenerate with |
|---|---|---|
| `TRAINING DATA/data_0.csv … data_5.csv` | ~1.1 GB | `ann_entanglement.dataset.generate_training_set(n_per_class, full=True)` |
| `ANN_PCA_DATASET/ANN/data_all.csv` | ~1.2 GB | same |

The paper's training set is 12×10³ states (2×10³ per class), generated in a few seconds by
`generate_training_set(2000)`.

## Permission

The experimental data were acquired by V. Gulati in Prof. K. Dorai's group and are shared here with the co-authors'
permission. Please cite the paper (see `CITATION.cff`) if you use them.
