"""Run / animal nesting audit (review task 2).

Images identify their imaging run with accuracy 1.00 (notebook 08). If every animal sits in exactly one run, then
cross-validation grouped by animal does not break the run confound for any model trained on pooled runs: test animals'
runs are always also in the training folds. Reports, from the annotated object:
  - how many animals appear in more than one run;
  - the animal x run contingency table (cells per animal per run);
  - disease stage x run and course (arm) x run (number of animals);
  - the same for runs 5/6 only (the data behind notebooks 01-09).

usage: python scripts/02_nesting_audit.py [--h5ad data/RRMAP2_all_runs.h5ad]
"""
import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(ROOT / "data" / "RRMAP2_all_runs.h5ad"))
    a = ap.parse_args()
    obs = ad.read_h5ad(a.h5ad, backed="r").obs[["sample_name", "run_id", "stage", "model"]].copy()
    for c in obs:
        obs[c] = obs[c].astype(str)
    obs["course"] = obs.model.str.replace("RELAPSE REMITTING", "RR").str.replace("CHRONIC", "chronic")
    pd.set_option("display.width", 200, "display.max_rows", 200, "display.max_columns", 20)

    runs_per_animal = obs.groupby("sample_name").run_id.nunique()
    print(f"animals: {len(runs_per_animal)}; runs: {sorted(obs.run_id.unique())}")
    print(f"animals in more than one run: {(runs_per_animal > 1).sum()}")
    if (runs_per_animal > 1).any():
        print("  ", sorted(runs_per_animal[runs_per_animal > 1].index))
    print("=> animals are", "NESTED within runs" if (runs_per_animal == 1).all() else "NOT nested within runs")

    print("\n## animal x run (cells per animal; one non-zero entry per row = nested)")
    ct = pd.crosstab(obs.sample_name, obs.run_id)
    print(ct.to_string())

    an = obs.drop_duplicates("sample_name")
    print("\n## disease stage x run (animals)")
    print(pd.crosstab(an.stage, an.run_id, margins=True).to_string())
    print("\n## course x run (animals)")
    print(pd.crosstab(an.course, an.run_id, margins=True).to_string())

    sub = an[an.run_id.isin(["run5", "run6"])]
    print("\n## runs 5/6 only (notebooks 01-09)")
    print(f"animals: {len(sub)}; in both run5 and run6: "
          f"{(obs[obs.run_id.isin(['run5', 'run6'])].groupby('sample_name').run_id.nunique() > 1).sum()}")
    print(pd.crosstab(sub.stage, sub.run_id, margins=True).to_string())
    print(pd.crosstab(sub.course, sub.run_id, margins=True).to_string())


if __name__ == "__main__":
    main()
