# Paper ↔ code map

Gulati*, Siyanwal*, Arvind & Dorai, *ANN-enhanced detection of multipartite entanglement in a three-qubit NMR quantum
processor*, Quantum Inf. Process. **24**, 74 (2025), doi:10.1007/s11128-025-04696-8 (received 2 Oct 2024,
accepted 13 Feb 2025, published 1 Mar 2025). Section, equation, table and figure numbers follow the arXiv version
(arXiv:2409.19739v1, 29 Sep 2024; the only arXiv version), which is the version submitted to QIP on 2 Oct 2024. The
journal typesetting may renumber equations.

Legend: ✅ reproduced from the original code · 🔁 re-implemented from the paper's Methods · 📦 original notebook or
file only · ⚠️ known discrepancy

## Sections

| Paper | Content | Code / data | Status |
|---|---|---|---|
| I | Introduction | — | |
| **II A**, Eq. (1), Table I | Canonical form λ₀\|000⟩+λ₁e^{iφ}\|100⟩+λ₂\|101⟩+λ₃\|110⟩+λ₄\|111⟩; class supports; λᵢ ~ U(0,1] | `states.py` (`random_canonical_state`, `_SUPPORT`, `_SEP_SUPPORTS`) · `notebooks/01_state_generation.ipynb` · original `00_original/A_…` (`*_ASD`) and `D_…` | ✅ |
| **II B**, Table II | Correlation tensor t_ijk, ranks {R(T123), R(T213), R(T312)} → class; 3-tangle τ = 4ρ₀₀ρ₇₇ | `measures.py` (`correlation_tensor`, `correlation_tensor_ranks`, `ranks_to_class`, `three_tangle_canonical`, `three_tangle_ckw`) · `notebooks/02_entanglement_measures.ipynb` · `mathematica/correlation_tensor_ranks_3q.nb` | ✅ |
| II B | Reduced-state rank checks used for labelling | `checks.py` (`single_qubit_ranks`, `rank_class`, `check_rho`) | ✅ (trace bug fixed) |
| **III A**, Fig. 1 | ANN protocol; SLOCC N×6×6 (softmax), GME N×(N+N mod 2)/2×1 (sigmoid, p₀ ≥ 0.5) | `models.py` (`build_slocc_ann`, `build_gme_ann`) | 🔁 |
| **III B**, Eqs. (3)–(5) | 128-element vector [Re ρ, Im ρ]; one-hot labels H(ρ) | `features.py` (`flatten_rho`, `FULL_128`) · `dataset.py` | ✅ |
| **III C** | Manual reduction 128 → 18 (14 real + 4 imaginary) | `features.py` (`CANONICAL_18`, `canonical_18`) · `notebooks/03_feature_reduction_anova.ipynb` | ✅ |
| **III D**, Fig. 2, Eqs. (7)–(12) | 30 NMR states (³ ¹⁹F spins, trifluoroiodoethylene), GRAPE preparation, CCO tomography; 18 : 12 validation : test, 100 of (⁵C₃)⁶ = 10⁶ splits; projected accuracy µ(A) = 0.6µ(A_v) + 0.4µ(A_t) | `data/raw/experimental_states_3q.csv`, `theoretical_targets_3q.csv` · `dataset.py` (`load_experimental`, `validation_test_splits`) · `experiments.overall_accuracy` · `mathematica/tomography/` · `matlab/grape_target_states.m` · `notebooks/07_experimental_states.ipynb` | ✅ data · 🔁 protocol |
| III E | Metrics (accuracy, loss) | Keras metrics in `experiments.py` | 🔁 |
| **IV A**, Table III | ANOVA F-values (`f_classif`, `SelectKBest`) | `selection.py` · `scripts/reproduce_tables.py` → `results/table3_anova.csv` | 🔁 |
| **IV B**, Table IV, Fig. 3 | SLOCC ANN, N = 1…18; optimum 6×6×6, µ(A_t) = 0.854 | `experiments.ann_sweep('slocc', …)` · `notebooks/04_slocc_ann.ipynb` · `results/table4_slocc_ann_*.csv` · early version `00_original/H_…` | 🔁 |
| **IV C**, Table V | GME ANN, N = 1…18; optimum 4×2×1 on {Re ρ₀₇, Re ρ₅₆, Re ρ₀₅, Re ρ₀₆}, µ(A_t) = 0.948 | `experiments.ann_sweep('gme', …)` · `notebooks/05_gme_ann.ipynb` · `results/table5_gme_ann_*.csv` | 🔁 |
| **V A**, Fig. 4, Table VI | SVM (linear, C = 0.1) and KNN (GME: k = 18, uniform, p = 1; SLOCC: k = 20, distance, p = 2) | `models.svm_classifier`, `models.knn_classifier` · `experiments.baselines` · `notebooks/06_svm_knn_benchmark.ipynb` | 🔁 |
| **V B**, Tables VII–VIII | 3-tangle and correlation-tensor ranks of the 30 states (tolerances 0.25 experimental, 10⁻¹⁰ theoretical) | `measures.py` · `notebooks/02_entanglement_measures.ipynb` · `mathematica/correlation_tensor_ranks_3q.nb` · `data/legacy/new_data_entropy_3tangle_20_11_2022.csv` | ✅ theory · ⚠️ experimental ranks (see below) |
| VI | Conclusions | — | |

## Hyper-parameters taken from the paper text

| Setting | Value | Paper |
|---|---|---|
| Training set | 12 × 10³ states, 2 × 10³ per SLOCC class (GME: 8 × 10³ non-GME, 4 × 10³ GME) | IV B, IV C |
| Optimiser | Adam, initial learning rate 10⁻³ | IV B |
| Activations | hidden "linear"; output softmax (SLOCC) / sigmoid (GME) | IV B, III A |
| Loss | categorical cross-entropy (SLOCC), binary cross-entropy (GME) | IV B, Eq. (14) |
| Early stopping | min_delta 0.01, patience 20, restore best weights | IV B |
| Epochs, batch | 100 epochs; batch 1000 (SLOCC), 500 (GME) | IV B, IV C |
| Splits | 100 random validation (18) : test (12) splits, 3 + 2 per class | III D |

## Where the code differs from, or could not confirm, the paper

1. **The final Tables III–VII script was not recovered.** Neither the academic Drive nor the co-author e-mail threads
   contain it; the most recent ANN notebook is from Dec 2022 (`00_original/H_…`, ReLU hidden layer, 100-neuron
   variants). `experiments.py` is a re-implementation from Sec. III D–V A, and its numbers come from new random training
   states, so they match the paper only statistically (see *Re-run results*).
2. **Table III absolute F-values.** The paper's F-values (e.g. 315.85 for Re ρ₀₇, SLOCC) are about 10 × smaller than
   those from 12 × 10³ states; F scales with sample size, so the paper's ranking was probably computed on a subset.
   The **GME ranking reproduces**: top four = {Re ρ₀₇, Re ρ₅₆, Re ρ₀₆, Re ρ₀₅}, the paper's 4×2×1 features.
   For SLOCC the paper's optimal six {Re ρ₀₇, Re ρ₀₅, Re ρ₀₆, Re ρ₅₆, Re ρ₅₅, Re ρ₀₀} are not the top six of its
   own Table III (Re ρ₅₇ 98.17 and Re ρ₆₇ 95.44 outrank Re ρ₅₅ 80.63 and Re ρ₀₀ 70.43).
3. **Table VIII experimental ranks** came from Mathematica (`mathematica/correlation_tensor_ranks_3q.nb`). The Python
   ranks reproduce the theoretical column for 30/30 states but the experimental column for 22/30 at the stated
   tolerance 0.25 (absolute singular-value cut-off; 23/30 at 0.30). The experimental ranks are sensitive to the
   tolerance convention and to small differences between the tomography output and the matrices in the analysis
   notebook.
4. **State `random_bs2_1_output` is labelled BS1.** Its theoretical target is a BS1 state and it is BS1₄ (F = 0.9731)
   in Table VIII; only the tomography file name says `bs2`.
5. **Spectrometer field.** Sec. III D says Bruker AVANCE-III 400 MHz; the acknowledgements say 600 MHz.
6. **Parameter counts.** Tables IV–V's P2/P3 columns equal L1 + L2 and L2 + L3 (e.g. 6 and 3 for 4×2×1; 12 and 12
   for 6×6×6), i.e. neuron counts, not the Keras weight + bias counts (10 and 3 for 4×2×1). The code follows the
   architectures.
7. **Hidden activation.** The paper says "linear"; the Dec 2022 notebook used ReLU. Default here: linear
   (`hidden_activation="relu"` reproduces the notebook).

## Re-run results

### Table IV (SLOCC): paper vs this repository (10 splits, `table4_slocc_ann_splits10.csv`)

| N | network | µ(A_t) paper | µ(A_t) here | A_T paper | A_T here | SVM A1 paper | SVM A1 here | KNN A2 paper | KNN A2 here |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1x6x6 | 0.330 | 0.300 ± 0.085 | 0.391 | 0.320 | 0.333 | 0.333 | 0.233 | 0.233 |
| 2 | 2x6x6 | 0.560 | 0.358 ± 0.135 | 0.601 | 0.373 | 0.600 | 0.333 | 0.400 | 0.400 |
| 3 | 3x6x6 | 0.782 | 0.358 ± 0.129 | 0.798 | 0.360 | 0.767 | 0.333 | 0.633 | 0.367 |
| 4 | 4x6x6 | 0.792 | 0.592 ± 0.108 | 0.794 | 0.570 | 0.767 | 0.567 | 0.667 | 0.400 |
| 5 | 5x6x6 | 0.776 | 0.800 ± 0.055 | 0.787 | 0.813 | 0.767 | 0.767 | 0.667 | 0.700 |
| 6 | 6x6x6 | 0.854 | 0.808 ± 0.065 | 0.867 | 0.817 | 0.833 | 0.833 | 0.733 | 0.633 |
| 7 | 7x6x6 | 0.783 | 0.817 ± 0.050 | 0.792 | 0.837 | 0.800 | 0.833 | 0.833 | 0.700 |
| 8 | 8x6x6 | 0.793 | 0.800 ± 0.055 | 0.806 | 0.820 | 0.800 | 0.833 | 0.800 | 0.767 |
| 9 | 9x6x6 | 0.802 | 0.800 ± 0.055 | 0.816 | 0.820 | 0.833 | 0.833 | 0.833 | 0.800 |
| 10 | 10x6x6 | 0.792 | 0.783 ± 0.055 | 0.804 | 0.807 | 0.833 | 0.800 | 0.767 | 0.733 |
| 11 | 11x6x6 | 0.789 | 0.783 ± 0.055 | 0.806 | 0.810 | 0.833 | 0.800 | 0.767 | 0.767 |
| 12 | 12x6x6 | 0.788 | 0.783 ± 0.055 | 0.807 | 0.813 | 0.833 | 0.800 | 0.767 | 0.733 |
| 13 | 13x6x6 | 0.787 | 0.783 ± 0.055 | 0.806 | 0.813 | 0.833 | 0.800 | 0.767 | 0.733 |
| 14 | 14x6x6 | 0.788 | 0.783 ± 0.055 | 0.803 | 0.813 | 0.833 | 0.800 | 0.767 | 0.733 |
| 15 | 15x6x6 | 0.792 | 0.783 ± 0.055 | 0.803 | 0.813 | 0.833 | 0.800 | 0.767 | 0.700 |
| 16 | 16x6x6 | 0.786 | 0.783 ± 0.055 | 0.802 | 0.813 | 0.833 | 0.800 | 0.767 | 0.733 |
| 17 | 17x6x6 | 0.788 | 0.783 ± 0.055 | 0.803 | 0.807 | 0.833 | 0.800 | 0.800 | 0.733 |
| 18 | 18x6x6 | 0.789 | 0.783 ± 0.055 | 0.805 | 0.807 | 0.833 | 0.800 | 0.800 | 0.800 |

Best µ(A_t) here: N = 7 (7x6x6), 0.817; features: Re_rho_07, Re_rho_67, Re_rho_57, Re_rho_05, Re_rho_06, Re_rho_56, Re_rho_00.

### Table V (GME): paper vs this repository (10 splits, `table5_gme_ann_splits10.csv`)

| N | network | µ(A_t) paper | µ(A_t) here | A_T paper | A_T here | SVM A1 paper | SVM A1 here | KNN A2 paper | KNN A2 here |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1x1x1 | 0.817 | 0.817 ± 0.050 | 0.823 | 0.817 | 0.833 | 0.833 | 0.400 | 0.400 |
| 2 | 2x1x1 | 0.862 | 0.883 ± 0.067 | 0.872 | 0.873 | 0.900 | 0.900 | 0.567 | 0.533 |
| 3 | 3x2x1 | 0.847 | 0.900 ± 0.050 | 0.866 | 0.880 | 0.933 | 0.933 | 0.833 | 0.767 |
| 4 | 4x2x1 | 0.948 | 0.967 ± 0.055 | 0.954 | 0.960 | 0.967 | 0.967 | 0.933 | 0.900 |
| 5 | 5x3x1 | 0.950 | 0.983 ± 0.050 | 0.958 | 0.957 | 0.933 | 0.900 | 0.933 | 0.967 |
| 6 | 6x3x1 | 0.865 | 0.942 ± 0.084 | 0.874 | 0.933 | 0.733 | 0.900 | 1.000 | 0.967 |
| 7 | 7x4x1 | 0.866 | 0.950 ± 0.067 | 0.869 | 0.943 | 0.733 | 0.900 | 1.000 | 0.967 |
| 8 | 8x4x1 | 0.865 | 0.983 ± 0.033 | 0.872 | 0.960 | 0.733 | 0.933 | 1.000 | 0.967 |
| 9 | 9x5x1 | 0.879 | 0.992 ± 0.025 | 0.887 | 0.970 | 0.733 | 0.967 | 0.967 | 0.967 |
| 10 | 10x5x1 | 0.892 | 0.992 ± 0.025 | 0.891 | 0.967 | 0.733 | 0.933 | 0.967 | 0.967 |
| 11 | 11x6x1 | 0.899 | 1.000 ± 0.000 | 0.909 | 0.973 | 0.733 | 0.933 | 1.000 | 0.967 |
| 12 | 12x6x1 | 0.889 | 1.000 ± 0.000 | 0.907 | 0.973 | 0.733 | 0.933 | 1.000 | 0.967 |
| 13 | 13x7x1 | 0.902 | 0.992 ± 0.025 | 0.915 | 0.970 | 0.733 | 0.933 | 0.967 | 1.000 |
| 14 | 14x7x1 | 0.906 | 0.992 ± 0.025 | 0.923 | 0.970 | 0.733 | 0.933 | 0.967 | 1.000 |
| 15 | 15x8x1 | 0.913 | 0.992 ± 0.025 | 0.930 | 0.967 | 0.733 | 0.933 | 0.967 | 1.000 |
| 16 | 16x8x1 | 0.955 | 0.992 ± 0.025 | 0.965 | 0.967 | 0.800 | 0.933 | 0.967 | 1.000 |
| 17 | 17x9x1 | 0.953 | 0.992 ± 0.025 | 0.965 | 0.970 | 0.800 | 0.933 | 0.967 | 1.000 |
| 18 | 18x9x1 | 0.952 | 0.992 ± 0.025 | 0.964 | 0.970 | 0.800 | 0.933 | 0.967 | 1.000 |

Best µ(A_t) here: N = 11 (11x6x1), 1.000; features: Re_rho_07, Re_rho_56, Re_rho_06, Re_rho_05, Re_rho_44, Re_rho_57, Re_rho_67, Re_rho_66, Re_rho_55, Re_rho_00, Im_rho_04.

### Table III ranking (this repository, 12 × 10³ states)

SLOCC order: Re_rho_07, Re_rho_67, Re_rho_57, Re_rho_05, Re_rho_06, Re_rho_56, Re_rho_00, Re_rho_55 …  
GME order: Re_rho_07, Re_rho_56, Re_rho_06, Re_rho_05, Re_rho_44, Re_rho_57, Re_rho_67, Re_rho_66 …

**Reading the comparison.** GME (Table V) reproduces: the 4×2×1 model on the paper's four features reaches
µ(A_t) = 0.967 (paper 0.948), and SVM A1 matches the paper exactly for N = 1–4. SLOCC (Table IV) matches for N ≥ 5
(µ(A_t) ≈ 0.78–0.82 vs the paper's 0.78–0.85; the 6×6×6 optimum gives 0.808 vs 0.854). For N = 2–4 it is lower,
because the re-computed ANOVA order puts Re ρ₆₇ and Re ρ₅₇ ahead of Re ρ₀₅ and Re ρ₀₆ (see item 2 above), so the
small models see different features from the paper's. These runs use 10 splits; the paper used 100.

Re-run with `python scripts/reproduce_tables.py --splits 100` for the full paper protocol.

## Published-version check

What the mail and Drive show about the path from preprint to publication:
- Submitted to QIP by K. Dorai on 2 Oct 2024; a revised Overleaf project ("ann_revised") was shared the same day.
- No referee reports, revision letters or proofs were in either of Shivanshu's mailboxes (they went to the
  corresponding author), so changes made during review **could not be checked**. This map follows arXiv v1. If the
  published version renumbered tables or changed numbers, update this file against the PDF of
  doi:10.1007/s11128-025-04696-8.
