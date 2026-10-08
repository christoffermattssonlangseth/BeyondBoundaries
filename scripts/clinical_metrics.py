"""Per-animal disease-course metrics from the daily clinical scores -> data/clinical/animal_course_metrics.csv

Input: data/clinical/Fixed_RRMap2_FinalSamples_AllScore/AllWeight_curated_20260723_212152.xlsx (one row per tissue
piece, d0-d51). Metrics: onset day (first score > 0), first-attack peak (max within 10 days of onset) and its day,
lowest score after it, area under the score curve, max score, relapse seen (rise >= 1 after that nadir), max weight
loss (% of d0).

Name fixes so the table joins the annotation (sample_name):
- the MILD30 animals are C_M30_k in the clinical sheet and C_L_k in the annotation. k = 1, 4, 5 match uniquely on
  sacrifice day + sex; k = 2, 3 (both d45, male, score 1) are paired by number and could be swapped.
- RR_MP_3 has no sacrifice score in the sheet; the annotation's 0.75 is used.

usage: python scripts/clinical_metrics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLIN = ROOT / "data" / "clinical"
TAG = "curated_20260723_212152"
ALIAS = {f"C_M30_{k}": f"C_L_{k}" for k in range(1, 6)}
SCORE_FILL = {"RR_MP_3": 0.75}


def main():
    S = pd.read_excel(CLIN / f"Fixed_RRMap2_FinalSamples_AllScore_{TAG}.xlsx")
    W = pd.read_excel(CLIN / f"Fixed_RRMap2_FinalSamples_AllWeight_{TAG}.xlsx")
    days = [c for c in S.columns if c.startswith("d") and c[1:].isdigit()]
    s = S.drop_duplicates("sample_name").set_index("sample_name")
    w = W.drop_duplicates("sample_name").set_index("sample_name")
    rows = []
    for a, r in s.iterrows():
        y = pd.to_numeric(r[days], errors="coerce")
        d = np.array([int(c[1:]) for c in days])
        ok = y.notna().to_numpy()
        y, d = y.to_numpy()[ok], d[ok]
        on = d[np.argmax(y > 0)] if (y > 0).any() else np.nan
        fa = fa_day = nadir = relapse = np.nan
        if on == on:
            win = (d >= on) & (d <= on + 10)
            fa, fa_day = y[win].max(), d[win][np.argmax(y[win])]
            post = d > fa_day
            if post.any():
                nadir = y[post].min()
                after = d > d[post][np.argmin(y[post])]
                relapse = bool(after.any() and y[after].max() - nadir >= 1)
        else:
            fa = 0.0
        ww = pd.to_numeric(w.loc[a, days], errors="coerce").to_numpy()[ok]
        rows.append(dict(animal=ALIAS.get(a, a), model=r.model, stage=r.stage, day=r.day_of_sacrifice,
                         score=SCORE_FILL.get(a, r.score_sacrifice), onset_day=on, first_peak=fa, first_peak_day=fa_day,
                         max_score=y.max(), auc=np.trapezoid(y, d), nadir_after_first=nadir, relapse_seen=relapse,
                         max_weight_loss_pct=100 * (1 - np.nanmin(ww) / ww[0]) if len(ww) else np.nan,
                         last_day_scored=d.max()))
    m = pd.DataFrame(rows).set_index("animal")
    m.to_csv(CLIN / "animal_course_metrics.csv")
    print(f"{len(m)} animals -> {CLIN / 'animal_course_metrics.csv'}")


if __name__ == "__main__":
    main()
