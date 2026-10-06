"""Phase 1: per-cell image features for every in-scope section -> <features_dir>/<section_id>.parquet

usage: python scripts/01_extract_features.py [--config config.yaml] [--sections run5_C2_G1_Mid_0088858 ...]
                                             [--workers N] [--max-windows N --out-suffix _bench] [--overwrite]
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from beyondboundaries.extract import extract_section  # noqa: E402
from beyondboundaries.features import FeatureParams  # noqa: E402
from beyondboundaries.io import find_bundles  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "config.yaml")
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-windows", type=int)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out_dir = ROOT / cfg["features_dir"]
    tl = cfg["tiling"]
    bundles = {s: b for s, b in find_bundles(cfg).items() if not a.sections or s in a.sections}
    print(f"{len(bundles)} sections in scope", flush=True)
    t0 = time.time()
    for sid, b in bundles.items():
        out = out_dir / f"{sid}{a.out_suffix}.parquet"
        if out.exists() and not a.overwrite:
            print(f"skip {out.name} (exists)", flush=True)
            continue
        extract_section(b, out, FeatureParams(**cfg["features"]), tile=tl["tile_px"], halo=tl["halo_px"],
                        workers=a.workers or tl["workers"], max_windows=a.max_windows,
                        log=lambda m: print(m, flush=True))
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
