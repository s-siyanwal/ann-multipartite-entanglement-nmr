# Changelog

## 1.0.0 — repository assembled (2026-11)

The code was reorganised from the original research notebooks (2022–2023, `notebooks/00_original/`) into the
`ann_entanglement` package. Behaviour changes relative to the originals:

### Bugs fixed
- **`check_rho` returned the purity as the trace.** `ENT-SIM-DATA-ANALYSIS` computed `tr = np.trace(M @ M)` and
  reported it as the trace. `checks.check_rho` now reports both `trace` = tr(ρ) and `purity` = tr(ρ²).
- **Witness classifier `NameError`.** `wit_o_classifier` referenced the undefined `exp_02` instead of `exp_O2`.
  The witness classifier is not part of the paper's results and is not ported; the bug is noted here for anyone
  re-running the original notebook.
- **`class_return_3q` called `FS_ASD(0)`**, but `FS_ASD` takes no arguments (`TypeError`). Replaced by
  `states.random_canonical_state(cls, rng)`.
- **Bi-separable labels swapped in docstrings.** The original comments called BS2 "AB-C" and BS3 "AC-B". The
  generators and rank checks (and the paper's Table I) give **BS2 = B|AC** and **BS3 = C|AB**; the package uses the
  correct partitions (`states.CLASS_PARTITION`).
- **Hard-coded Windows paths** (`C:\Users\…`) in the original notebooks were replaced by `<LOCAL_DATA_DIR>`; the new
  code reads data relative to the repository.

### Changes
- Coefficients are sampled from U(0,1] (`1 - rng.random()`), as the paper states, rather than U[0,1); this avoids
  a zero coefficient silently changing the state's class.
- Seeded `numpy.random.Generator` everywhere, so generated datasets are reproducible.
- **Re-implementation:** `experiments.py` and `scripts/reproduce_tables.py` re-create the protocol behind
  Tables III–VII from the paper's Methods (Sec. III D, IV, V A): ANOVA ranking, 18 models per task, 100 random
  18:12 validation:test splits, early stopping (min_delta 0.01, patience 20), 100 epochs, batch 1000/500,
  projected overall accuracy via Eqs. (11)–(12), and linear-SVM / KNN baselines. The original 2023 script was not
  among the recovered files.
- The ANN hidden layer uses the `linear` activation stated in the paper (Sec. IV B); the Dec-2022 exploratory
  notebook used `relu`. Both are available through `hidden_activation`.
