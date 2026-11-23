"""Re-run the Table IV / V / VI-VII / Fig. 4 protocol and write CSVs to results/.

    python scripts/reproduce_tables.py                 # full paper protocol: 100 splits (hours on CPU)
    python scripts/reproduce_tables.py --splits 10     # quick version used for results/ in this repo
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from ann_entanglement import experiments, selection  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--splits", type=int, default=100)
p.add_argument("--per-class", type=int, default=2000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tasks", default="gme,slocc", help="comma-separated subset of gme,slocc")
p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
a = p.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

train, exp_x = experiments.prepare(a.per_class, seed=a.seed)
t3 = selection.anova_ranking(train, "class_code").rename(columns={"F": "F_SLOCC", "p": "p_SLOCC"}).merge(
    selection.anova_ranking(train, "gme").rename(columns={"F": "F_GME", "p": "p_GME"}), on="feature")
t3.to_csv(out / "table3_anova.csv", index=False)
for task, name in (("gme", "table5_gme_ann"), ("slocc", "table4_slocc_ann")):
    if task not in a.tasks.split(","):
        continue
    sweep = experiments.ann_sweep(task, train, exp_x, n_splits=a.splits, seed=a.seed)
    base = experiments.baselines(task, train, exp_x)
    sweep.merge(base, on="N").to_csv(out / f"{name}_splits{a.splits}.csv", index=False)
    print(name, "done", flush=True)
