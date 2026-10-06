"""Phase 1: per-cell image features for every in-scope section -> <features_dir>/<section_id>.parquet

usage: python scripts/01_extract_features.py [--config config.yaml] [--sections run5_C2_G1_Mid_0088858 ...]
                                             [--workers N] [--max-windows N --out-suffix _bench] [--overwrite]
"""
import argparse
import fnmatch
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from beyondboundaries.extract import extract_section  # noqa: E402
from beyondboundaries.features import FeatureParams  # noqa: E402
from beyondboundaries.io import section_id  # noqa: E402


def in_scope_bundles(cfg):
    for run_dir in cfg["runs"].values():
        for b in sorted(Path(run_dir).glob("output-*")):
            region = b.name.split("__")[2]
            if not any(fnmatch.fnmatch(region, pat) for pat in cfg["exclude_regions"]):
                yield b


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
    bundles = [b for b in in_scope_bundles(cfg) if not a.sections or section_id(b) in a.sections]
    print(f"{len(bundles)} sections in scope", flush=True)
    t0 = time.time()
    for b in bundles:
        out = out_dir / f"{section_id(b)}{a.out_suffix}.parquet"
        if out.exists() and not a.overwrite:
            print(f"skip {out.name} (exists)", flush=True)
            continue
        extract_section(b, out, FeatureParams(**cfg["features"]), tile=tl["tile_px"], halo=tl["halo_px"],
                        workers=a.workers or tl["workers"], max_windows=a.max_windows,
                        log=lambda m: print(m, flush=True))
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
