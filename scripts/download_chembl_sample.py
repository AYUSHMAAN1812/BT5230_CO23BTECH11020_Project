#!/usr/bin/env python
"""Download a deterministic, distributed ChEMBL molecule sample for synthetic OCSR data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_ROOT = "https://www.ebi.ac.uk/chembl/api/data"
USER_AGENT = "handdrawn-ocsr-student-project/0.1"


def get_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--pages", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_records < 1 or args.pages < 1:
        raise ValueError("max-records and pages must be positive")

    status = get_json(f"{API_ROOT}/status.json")
    query = {
        "molecule_structures__isnull": "false",
        "only": "molecule_chembl_id,molecule_structures",
    }
    initial = get_json(f"{API_ROOT}/molecule.json?{urlencode({**query, 'limit': 1})}")
    total_count = int(initial["page_meta"]["total_count"])
    page_count = min(args.pages, args.max_records)
    page_size = (args.max_records + page_count - 1) // page_count
    max_offset = max(total_count - page_size, 0)
    offsets = [round(i * max_offset / max(page_count - 1, 1)) for i in range(page_count)]

    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for offset in offsets:
        params = {**query, "limit": page_size, "offset": offset}
        payload = get_json(f"{API_ROOT}/molecule.json?{urlencode(params)}")
        for molecule in payload.get("molecules", []):
            structures = molecule.get("molecule_structures") or {}
            chembl_id = molecule.get("molecule_chembl_id")
            smiles = structures.get("canonical_smiles")
            if chembl_id and smiles and chembl_id not in seen_ids:
                seen_ids.add(chembl_id)
                records.append({"compound_id": chembl_id, "smiles": smiles})
            if len(records) >= args.max_records:
                break
        if len(records) >= args.max_records:
            break

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["compound_id", "smiles"])
        writer.writeheader()
        writer.writerows(records)

    checksum = hashlib.sha256(args.output_csv.read_bytes()).hexdigest()
    metadata = {
        "source": "ChEMBL Data Web Services",
        "api_root": API_ROOT,
        "chembl_db_version": status.get("chembl_db_version"),
        "chembl_release_date": status.get("chembl_release_date"),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "requested_records": args.max_records,
        "downloaded_records": len(records),
        "source_total_count": total_count,
        "distributed_offsets": offsets,
        "sha256": checksum,
    }
    metadata_path = args.output_csv.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

