"""MILD30 sensitivity check (review task 3).

Claim under test: "chronic severity is set at the first attack" (notebooks 11-12): chronic SEVERE animals all peaked at
3.5 in their first attack and never recovered; MILD animals peaked lower and partly recovered. MILD30 is run-1-only
(batch-confounded), so the comparison is repeated with MILD30 excluded. Uses the per-animal clinical metrics
(`scripts/clinical_metrics.py` -> data/clinical/animal_course_metrics.csv).

Reported for chronic-late animals (MILD16, MILD30, SEVERE16, SEVERE30), with and without MILD30:
  - first-attack peak and lowest score after it, MILD vs SEVERE (Mann-Whitney, one-sided: SEVERE higher);
  - separation (does every SEVERE animal have a higher first peak than every MILD animal?);
  - Spearman of first-attack peak with the score at sacrifice.

usage: python scripts/03_mild30_sensitivity.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def report(d: pd.DataFrame, label: str):
    mild, sev = d[d.stage.str.startswith("MILD")], d[d.stage.str.startswith("SEVERE")]
    print(f"\n## {label}: {len(mild)} MILD ({', '.join(sorted(mild.stage.unique()))}) vs {len(sev)} SEVERE")
    for col in ["first_peak", "nadir_after_first"]:
        p = mannwhitneyu(sev[col], mild[col], alternative="greater").pvalue
        print(f"  {col}: MILD median {mild[col].median():.2f} (range {mild[col].min():.2f}-{mild[col].max():.2f}) vs "
              f"SEVERE {sev[col].median():.2f} ({sev[col].min():.2f}-{sev[col].max():.2f}); one-sided p = {p:.3g}")
    print(f"  every SEVERE first peak > every MILD first peak: {sev.first_peak.min() > mild.first_peak.max()}")
    r = spearmanr(d.first_peak, d.score)
    print(f"  rho(first-attack peak, score at sacrifice) = {r.statistic:+.2f} (p = {r.pvalue:.2g}, n = {len(d)})")


def main():
    m = pd.read_csv(ROOT / "data" / "clinical" / "animal_course_metrics.csv", index_col=0)
    cl = m[m.stage.isin(["MILD16", "MILD30", "SEVERE16", "SEVERE30"])]
    print(cl[["stage", "first_peak", "nadir_after_first", "score", "day"]].sort_values(["stage", "first_peak"]).to_string())
    report(cl, "all chronic-late animals")
    report(cl[cl.stage != "MILD30"], "MILD30 excluded")
    report(cl[cl.stage.isin(["MILD16", "SEVERE16"])], "same-run, same-day cohort only (MILD16 vs SEVERE16)")


if __name__ == "__main__":
    main()
