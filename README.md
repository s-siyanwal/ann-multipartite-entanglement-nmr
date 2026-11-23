# ANN-enhanced detection of multipartite entanglement in a three-qubit NMR quantum processor

Code and data for

> **Vaishali Gulati\*, Shivanshu Siyanwal\*, Arvind, Kavita Dorai**
> *ANN-enhanced detection of multipartite entanglement in a three-qubit NMR quantum processor*,
> Quantum Information Processing **24**, 74 (2025).
> [doi:10.1007/s11128-025-04696-8](https://doi.org/10.1007/s11128-025-04696-8) ·
> [arXiv:2409.19739](https://arxiv.org/abs/2409.19739)
>
> \* These authors contributed equally to this work.
> Department of Physical Sciences, IISER Mohali, India. Also presented at QTML 2024.

Artificial neural networks are trained on simulated three-qubit pure states in the canonical form and then used to
(i) classify **six SLOCC-inequivalent classes** (SEP, three bi-separable classes, W, GHZ) and
(ii) detect **genuine multipartite entanglement (GME)** in 30 states prepared on an NMR quantum processor, using
only a handful of density-matrix elements chosen by ANOVA F-scores. Results are benchmarked against SVM, KNN,
correlation-tensor ranks and the 3-tangle.

**Read the code next to the paper:** [`PAPER_MAP.md`](PAPER_MAP.md) maps every section, equation, table and figure to a
module, notebook or data file.

## Repository layout

```
├── src/ann_entanglement/   Python package, one module per paper section (see PAPER_MAP.md)
│   ├── states.py           Sec. II A, Table I  canonical-form state generators
│   ├── measures.py         Sec. II B, Table II correlation-tensor ranks, 3-tangle, entropies
│   ├── checks.py           density-matrix validity / reduced-state rank checks
│   ├── features.py         Sec. III B-C        128 -> 18 feature reduction
│   ├── dataset.py          Sec. III B, III D   training set, experimental set, 18:12 splits
│   ├── selection.py        Sec. IV A           ANOVA F-value ranking (Table III)
│   ├── models.py           Sec. III A, IV, V A ANN / SVM / KNN with the paper's hyper-parameters
│   └── experiments.py      Sec. IV B-C, V A    protocol behind Tables IV-VII (re-implementation)
├── notebooks/              01-07: clean notebooks in paper order; 00_original/: the original research notebooks
├── scripts/reproduce_tables.py   re-run Tables III-VII and Fig. 4, write CSVs to results/
├── results/                outputs of scripts/reproduce_tables.py
├── data/                   raw experimental data + provenance (data/README.md)
├── mathematica/            NMR state tomography and correlation-tensor rank notebooks (V. Gulati)
├── matlab/                 GRAPE target density matrices for the prepared states
└── tests/                  pytest: physics checks and a smoke test
```

## Quick start

```bash
conda env create -f environment.yml && conda activate ann-entanglement   # or: pip install -r requirements.txt
pip install -e .
pytest                                   # ~10 s: physics checks + tiny ANN forward pass
jupyter lab notebooks/                   # 01 -> 07 follow the paper
python scripts/reproduce_tables.py --splits 10    # quick re-run of Tables III-VII (~1 h CPU)
python scripts/reproduce_tables.py               # full protocol, 100 splits (many hours on CPU)
```

## What is and is not here

* **Reproduced from the original code:** state generation, validity / rank checks, correlation tensor, 3-tangle,
  experimental and theoretical density matrices (extracted from the original analysis notebooks).
* **Re-implemented from the paper's Methods:** the final ANOVA + 18-model ANN sweep and the SVM/KNN benchmark
  (`experiments.py`). The 2023 script that produced Tables IV-VII was not among the recovered files. Numbers differ in
  the last digits because random states and TensorFlow versions differ; see PAPER_MAP.md for a side-by-side.
* **Not included:** the large simulated training CSVs (~1 GB; regenerate with `dataset.generate_training_set`),
  trained checkpoints, and third-party code. The correlation-tensor method follows Gulati, Arvind & Dorai,
  *Eur. Phys. J. D* **76**, 194 (2022) (paper ref. [49]). The unsupervised entanglement-detection code of
  Chen, Pan, Zhang & Cheng, *Detecting quantum entanglement with unsupervised learning*,
  [arXiv:2103.04804](https://arxiv.org/abs/2103.04804), was used during exploratory work; it is third-party and is
  referenced, not redistributed.

## Citation

See [`CITATION.cff`](CITATION.cff), or:

```bibtex
@article{gulati2025ann,
  title   = {{ANN}-enhanced detection of multipartite entanglement in a three-qubit {NMR} quantum processor},
  author  = {Gulati, Vaishali and Siyanwal, Shivanshu and Arvind and Dorai, Kavita},
  journal = {Quantum Information Processing},
  volume  = {24},
  pages   = {74},
  year    = {2025},
  doi     = {10.1007/s11128-025-04696-8},
  eprint  = {2409.19739},
  archivePrefix = {arXiv}
}
```

## License

Code: MIT (see `LICENSE`). Experimental NMR data were acquired at the NMR Research Facility, IISER Mohali, and are
shared with the co-authors' permission; please cite the paper if you use them.
