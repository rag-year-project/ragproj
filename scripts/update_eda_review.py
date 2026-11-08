"""Reapply human source-quality decisions to a completed EDA without re-extracting PDFs."""

import json
from pathlib import Path

import pandas as pd

from yearproj.eda import (
    attach_document_review,
    enrich_text_quality,
    make_charts,
    report,
)

root = Path(__file__).resolve().parents[1]
output = root / "reports/eda"
summary_path = output / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
documents = pd.read_csv(output / "documents.csv")
pages = pd.read_csv(output / "pages.csv", keep_default_na=False)
enrich_text_quality(summary, pages, output)
make_charts(pages, documents, output)
attach_document_review(summary, documents, root)
report(summary, output)
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Attached verified source-quality decisions; raw corpus counts preserved.")
