"""Materialize connector-provided temporary download references (never commit URLs)."""

import concurrent.futures
import json
import urllib.request
from pathlib import Path


def download(record: dict) -> str:
    destination = Path("data/raw/drive_manuals") / record["filename"]
    if destination.exists() and destination.stat().st_size == record["size_bytes"]:
        return f"SKIP {destination.name}"
    partial = destination.with_suffix(".part")
    try:
        with (
            urllib.request.urlopen(record["download_url"], timeout=90) as response,
            partial.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        with partial.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("Response is not a PDF")
        if partial.stat().st_size != record["size_bytes"]:
            raise ValueError("File size differs from Drive metadata")
        partial.replace(destination)
        return f"OK {destination.name}"
    except (OSError, ValueError) as error:
        partial.unlink(missing_ok=True)
        return f"ERROR {destination.name}: {type(error).__name__}: {error}"


if __name__ == "__main__":
    records = json.loads(Path(".tmp/download_refs.json").read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for result in executor.map(download, records):
            print(result, flush=True)
