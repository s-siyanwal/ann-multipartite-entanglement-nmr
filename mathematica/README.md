# Mathematica notebooks (author: Vaishali Gulati)

Shared by V. Gulati with the co-authors (June 2022 and April 2023). They need Wolfram Mathematica 12+ to run.

| Path | Paper | What |
|---|---|---|
| `tomography/*.nb` | Sec. III D, Fig. 2 (d–e) | Full three-qubit state tomography of the prepared NMR states. Each notebook reconstructs ρ from the measured spectra (constrained convex optimisation, CCO) and computes the fidelity with its target. |
| `correlation_tensor_ranks_3q.nb` | Sec. II B, Sec. V B, Table VIII | Correlation tensors t_ijk and the ranks R(T123), R(T213), R(T312) for the theoretical and experimental states (original file name `matrix_rank_3_qubit_acin.nb`). |

## Tomography file → state

| Notebook | Class | Notes |
|---|---|---|
| `3F_sep.nb`, `random_sep1–3.nb`, `3F_random_sep1–2.nb` | SEP | |
| `15jan21_bs1.nb`, `random_1bs1–5.nb` | BS1 | |
| `3F_2bs.nb`, `random_2bs1,3–5.nb` | BS2 | |
| `3F_3bs.nb` | BS3 | the random BS3 tomography notebooks were not in the mail or Drive |
| `3F_w.nb` | W | the random W tomography notebooks were not in the mail or Drive |
| `3F_ghz.nb`, `3F_random_ghz1–4,6.nb` | GHZ | |
| `3F_pps.nb` | pseudo-pure state | reference state, not part of the dataset |

The reconstructed matrices used by the Python code are in `data/raw/experimental_states_3q.csv`.
