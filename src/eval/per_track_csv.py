#!/usr/bin/env python3
"""Flatten an NTv3 finetune result JSON into a per-track CSV (track_id, assay, pearson) by joining
``per_track_pearson`` with the benchmark's ``benchmark_metadata.tsv`` (file_id -> assay). Paper-ready
view of all 34 human tracks. Usage:
  python -m src.eval.per_track_csv --result <result.json> --meta <benchmark_metadata.tsv> --out <csv>
"""

import os
import argparse
import csv
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument(
        "--meta",
        default=os.environ.get("OG_SCRATCH", "output")
        + "/ntv3_benchmark_data/benchmark_metadata.tsv",
    )
    ap.add_argument("--species", default="human")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = json.load(open(args.result))
    per_track = res["per_track_pearson"]  # file_id -> pearson
    assay = {}  # file_id -> assay (this species)
    with open(args.meta) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["species_common_name"] == args.species:
                assay[r["file_id"]] = r["assay"]

    rows = [(tid, assay.get(tid, "UNKNOWN"), pcc) for tid, pcc in per_track.items()]
    rows.sort(key=lambda x: (x[1], -x[2]))  # by assay, then PCC desc
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "assay", "test_pearson"])
        for tid, asy, pcc in rows:
            w.writerow([tid, asy, f"{pcc:.4f}"])

    # sanity: per-assay mean here must reproduce the JSON's test_pearson_by_assay
    from collections import defaultdict

    agg = defaultdict(list)
    for _, asy, pcc in rows:
        agg[asy].append(pcc)
    print(f"wrote {len(rows)} tracks -> {args.out}")
    print(f"{'assay':<20}{'n':>3}{'mean_PCC':>10}{'  (JSON by_assay)':>18}")
    by_assay = res.get("test_pearson_by_assay", {})
    for asy in sorted(agg):
        m = sum(agg[asy]) / len(agg[asy])
        print(f"{asy:<20}{len(agg[asy]):>3}{m:>10.4f}{by_assay.get(asy, float('nan')):>18.4f}")
    print(
        f"{'TOTAL':<20}{len(rows):>3}{sum(p for *_, p in rows) / len(rows):>10.4f}"
        f"{res.get('test_mean_pearson', float('nan')):>18.4f}"
    )


if __name__ == "__main__":
    main()
