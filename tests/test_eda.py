"""Numerical and data-validity checks; no network, models or original PDFs needed."""

import json
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pymupdf

from yearproj.eda import (
    Config,
    enrich_text_quality,
    extract_pdf,
    question_analysis,
    rectangle_union_area,
)


@contextmanager
def temporary_directory():
    # tempfile uses mode 0700, which conflicts with the Windows restricted token.
    directory = Path.cwd() / ".tmp" / "eda-tests" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()


class EdaTests(unittest.TestCase):
    def test_padding_does_not_inflate_nonspace_text_volume(self):
        with temporary_directory() as directory:
            records = [
                {"page_id": "p1", "text": "a" + " " * 99 + "b"},
                {"page_id": "p2", "text": ""},
            ]
            (directory / "pages_text.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records), encoding="utf-8"
            )
            pages = pd.DataFrame({"page_id": ["p1", "p2"], "char_count": [101, 0]})
            summary = {}
            enrich_text_quality(summary, pages, directory)
            self.assertEqual(list(pages.nonspace_char_count), [2, 0])
            self.assertAlmostEqual(pages.iloc[0].whitespace_share, 99 / 101)
            self.assertEqual(pages.iloc[1].whitespace_share, 0)
            self.assertEqual(summary["text_quality"]["whitespace_90pct_pages"], 1)

    def test_overlapping_image_placements(self):
        self.assertAlmostEqual(
            rectangle_union_area([(0, 0, 10, 10), (5, 5, 15, 15)]), 175
        )
        self.assertAlmostEqual(rectangle_union_area([(0, 0, 10, 10)] * 2), 100)
        self.assertEqual(rectangle_union_area([]), 0)

    def test_blank_page_is_not_automatically_a_scan(self):
        with temporary_directory() as directory:
            path = Path(directory) / "example.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), "Readable technical manual page. " * 4)
                image = page.get_pixmap().tobytes("png")
                document.new_page()
                raster = document.new_page()
                raster.insert_image(raster.rect, stream=image)
                punctuation = document.new_page()
                punctuation.insert_text((72, 72), "?!")
                document.save(path)
            pages, doc = extract_pdf(path, Config())
            self.assertEqual(doc["page_count"], 4)
            self.assertFalse(pages[0]["no_text_layer"])
            self.assertTrue(pages[1]["no_text_layer"])
            self.assertFalse(pages[1]["scan_candidate"])
            self.assertTrue(pages[2]["scan_candidate"])
            self.assertAlmostEqual(pages[2]["image_coverage"], 1.0, places=6)
            self.assertFalse(pages[3]["no_text_layer"])
            self.assertEqual([p["page_num"] for p in pages], [1, 2, 3, 4])

    def test_unknown_answers_do_not_become_negative_labels(self):
        pages = pd.DataFrame([{"doc_id": "manual", "page_num": 1}])
        rows = [
            {
                "q_id": str(i),
                "question": "Where?",
                "doc_id": "manual",
                "target_page": 1,
                "question_type": "schematic",
                "answer": "Evidence",
                "requires_visual": label,
                "evidence_notes": "Compare the arrows in PDF with the extracted text.",
            }
            for i, label in enumerate(["yes", "no", "unknown"])
        ]
        with temporary_directory() as directory:
            path = Path(directory) / "qa.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            _, result = question_analysis(path, pages)
            self.assertEqual(result["reviewed"], 2)
            self.assertEqual(result["visual_share_among_reviewed"], 0.5)
            self.assertEqual(result["full_sample_bounds"], [1 / 3, 2 / 3])
            low, high = result["wilson_95_conditional"]
            self.assertTrue(0 < low < 0.5 < high < 1)
            rows[0]["target_page"] = 1.5
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Invalid QA target"):
                question_analysis(path, pages)

    def test_missing_questions_have_no_metric(self):
        _, result = question_analysis(None, pd.DataFrame())
        self.assertEqual(result, {"status": "no_annotations", "total": 0})


if __name__ == "__main__":
    unittest.main()
