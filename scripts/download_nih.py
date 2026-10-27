"""Download a patient-split subset of NIH ChestX-ray14 from Kaggle, without versioning it.

    uv run python scripts/download_nih.py --out DIR [--positives-per-class N|all]
                                          [--negatives N|all] [--from-archive [DIR]]

Writes <out>/{train,val,test_locked}/images/*.png and metadata.csv in the canonical schema.
`make seed` makes the same selection inside the Prefect `ingest` flow and then validates it,
versions it with DVC, and loads it into Postgres; use this script to inspect the data or to
pre-fetch images. Files are fetched one by one through kagglehub (public dataset, no token
needed) and cached under .cache/cxr/kaggle, so a later `make seed` downloads nothing again.
With --from-archive, the whole 45 GB archive is downloaded once instead (or an extracted copy
is used), the fast path for large subsets. No images are ever committed to the repo.
"""

import argparse
import logging
from pathlib import Path

from cxr_flows import nih
from cxr_flows.schema import SPLITS, write_metadata
from cxr_flows.settings import Paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positives-per-class", type=nih.parse_count, default=500, metavar="N|all")
    parser.add_argument("--negatives", type=nih.parse_count, default=1500, metavar="N|all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--from-archive", nargs="?", const="", metavar="DIR")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cache = Paths.discover().cache
    staging = cache / "kaggle"
    archive = None
    if args.from_archive:
        archive = nih.check_archive_root(Path(args.from_archive).expanduser().resolve())
    elif args.from_archive is not None:
        archive = nih.fetch_archive(cache / "kagglehub")
    entries = nih.read_data_entry(nih.fetch_data_entry(staging, archive))
    subset = nih.select_subset(entries, args.positives_per_class, args.negatives, args.seed)
    for split in SPLITS:
        rows = subset[subset["split"] == split]
        out = args.out / split
        nih.download_images(rows, out, staging, args.workers, archive, link=not args.from_archive)
        write_metadata(nih.canonical(nih.measure_sizes(rows, out)), out)
        print(f"{split}: {len(rows)} images, {rows['patient_id'].nunique()} patients → {out}")


if __name__ == "__main__":
    main()
