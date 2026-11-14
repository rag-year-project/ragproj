"""Reproducible, local-only EDA of PDF manuals; automatic labels are candidates."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import combinations, pairwise
from pathlib import Path

import pandas as pd
import pymupdf
from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 42


@dataclass(frozen=True)
class Config:
    seed: int = 42
    sparse_chars: int = 100
    scan_image_coverage: float = 0.8
    meaningful_image_coverage: float = 0.01
    vector_items: int = 20
    max_table_paths: int = 1500
    language_min_letters: int = 100
    language_probability: float = 0.8
    near_duplicate_jaccard: float = 0.65
    duplicate_min_tokens: int = 100
    shingle_size: int = 5


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def rectangle_union_area(rectangles: list[tuple]) -> float:
    """Exact union area, without double-counting overlapping PDF image placements."""
    xs = sorted({x for r in rectangles for x in (r[0], r[2])})
    area = 0.0
    for left, right in pairwise(xs):
        intervals = sorted(
            (r[1], r[3]) for r in rectangles if r[0] < right and r[2] > left
        )
        end = -math.inf
        height = 0.0
        for bottom, top in intervals:
            height += max(0.0, top - max(bottom, end))
            end = max(end, top)
        area += (right - left) * height
    return area


def language_features(text: str, config: Config) -> dict:
    letters = [c for c in text if c.isalpha()]
    cyrillic = sum("\u0400" <= c <= "\u052f" for c in letters)
    latin = sum("a" <= c.lower() <= "z" for c in letters)
    language, probability = "unknown", 0.0
    if len(letters) >= config.language_min_letters:
        try:
            # Equally spaced parts reduce the bias towards the top of the page.
            sample = (
                text
                if len(text) <= 6000
                else "\n".join(
                    text[i : i + 2000]
                    for i in (0, len(text) // 2 - 1000, len(text) - 2000)
                )
            )
            candidate = detect_langs(sample)[0]
            probability = candidate.prob
            if probability >= config.language_probability:
                language = candidate.lang
        except LangDetectException:
            pass
    return {
        "language_candidate": language,
        "language_probability": probability,
        "cyrillic_share": cyrillic / len(letters) if letters else 0.0,
        "mixed_script_candidate": cyrillic >= 50 and latin >= 50,
        "russian_candidate": language == "ru",
    }


def extract_pdf(path: Path, config: Config) -> tuple[list[dict], dict]:
    records = []
    text_parts = []
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError("Encrypted PDF requires a password")
        for page_index, page in enumerate(document):
            text = page.get_text("text", sort=True).strip()
            norm = normalized_text(text)
            tokens = norm.split()
            text_parts.append(norm)
            width, height = page.rect.width, page.rect.height
            area = width * height
            image_info = page.get_image_info(hashes=False, xrefs=False)
            rectangles = []
            for image in image_info:
                clipped = pymupdf.Rect(image["bbox"]) & page.rect
                if not clipped.is_empty:
                    rectangles.append(tuple(clipped))
            coverage = rectangle_union_area(rectangles) / area if area else 0.0
            paths = page.get_drawings()
            vector_count = sum(len(p.get("items", [])) for p in paths)
            table_count = 0
            table_error = ""
            table_status = "no_vector_paths"
            if paths and len(paths) <= config.max_table_paths:
                try:
                    kwargs = {"strategy": "lines_strict", "paths": paths}
                    # PyMuPDF 2.x defaults to a layout model; this pipeline is geometric only.
                    if (
                        "use_layout"
                        in inspect.signature(pymupdf.table.find_tables).parameters
                    ):
                        kwargs["use_layout"] = False
                    tables = page.find_tables(**kwargs).tables
                    table_count = sum(
                        t.row_count >= 2 and t.col_count >= 2 for t in tables
                    )
                    table_status = "ok"
                except Exception as error:  # noqa: BLE001 - record failures without hiding them
                    table_status = "error"
                    table_error = f"{type(error).__name__}: {error}"
            elif len(paths) > config.max_table_paths:
                table_status = "skipped_complex_page"
            has_text = bool(text)
            meaningful_image = coverage >= config.meaningful_image_coverage
            scan = not has_text and coverage >= config.scan_image_coverage
            sparse = len(text) < config.sparse_chars
            graphics = meaningful_image or vector_count >= config.vector_items
            if scan:
                kind = "scan_candidate"
            elif not has_text:
                kind = "no_text_other"
            elif table_count:
                kind = "table_candidate"
            elif graphics:
                kind = "graphics_candidate"
            else:
                kind = "text_without_detected_graphics"
            record = {
                "page_id": f"{path.stem}_p{page_index + 1:03d}",
                "doc_id": path.stem,
                "filename": path.name,
                "page_num": page_index + 1,
                "char_count": len(text),
                "token_count": len(tokens),
                "width_pt": width,
                "height_pt": height,
                "image_count": len(image_info),
                "image_coverage": coverage,
                "vector_items": vector_count,
                "table_count_candidate": table_count,
                "table_detection_status": table_status,
                "table_error": table_error,
                "no_text_layer": not has_text,
                "sparse_text": sparse,
                "scan_candidate": scan,
                "graphics_candidate": graphics,
                "page_type_candidate": kind,
                "text_sha256": hashlib.sha256(norm.encode()).hexdigest()
                if norm
                else "",
                **language_features(text, config),
                "text": text,
            }
            records.append(record)
    full_text = "\n".join(text_parts)
    doc_record = {
        "doc_id": path.stem,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "page_count": len(records),
        "total_chars": sum(p["char_count"] for p in records),
        "total_tokens": sum(p["token_count"] for p in records),
        "document_text_sha256": hashlib.sha256(full_text.encode()).hexdigest()
        if full_text.strip()
        else "",
        "full_text": full_text,
    }
    return records, doc_record


def duplicate_analysis(documents: pd.DataFrame, pages: pd.DataFrame, config: Config):
    pairs = []
    shingles = {}
    for doc in documents.to_dict("records"):
        tokens = doc["full_text"].split()
        shingles[doc["doc_id"]] = (
            {
                tuple(tokens[i : i + config.shingle_size])
                for i in range(max(0, len(tokens) - config.shingle_size + 1))
            }
            if len(tokens) >= config.duplicate_min_tokens
            else set()
        )
    groups = {doc_id: doc_id for doc_id in documents["doc_id"]}

    def root(value):
        while groups[value] != value:
            value = groups[value]
        return value

    for first, second in combinations(documents.to_dict("records"), 2):
        left, right = first["doc_id"], second["doc_id"]
        a, b = shingles[left], shingles[right]
        similarity = len(a & b) / len(a | b) if a and b else None
        exact_file = first["file_sha256"] == second["file_sha256"]
        exact_text = bool(first["document_text_sha256"]) and (
            first["document_text_sha256"] == second["document_text_sha256"]
        )
        if (
            exact_file
            or exact_text
            or (similarity is not None and similarity >= config.near_duplicate_jaccard)
        ):
            pairs.append(
                {
                    "doc_a": left,
                    "doc_b": right,
                    "exact_file": exact_file,
                    "exact_text": exact_text,
                    "shingle_jaccard": similarity,
                }
            )
            groups[root(right)] = root(left)
    pair_frame = pd.DataFrame(
        pairs,
        columns=[
            "doc_a",
            "doc_b",
            "exact_file",
            "exact_text",
            "shingle_jaccard",
        ],
    )
    # Short boilerplate is excluded; equal extracted text does not imply equal images.
    repeated = pages[
        (pages["token_count"] >= config.duplicate_min_tokens)
        & pages["text_sha256"].ne("")
    ].copy()
    repeated = repeated[repeated.duplicated("text_sha256", keep=False)]
    group_frame = pd.DataFrame(
        [
            {
                "doc_id": doc_id,
                "duplicate_group_candidate": root(doc_id),
                "family_group_reviewed": "",
            }
            for doc_id in groups
        ]
    )
    return pair_frame, repeated.drop(columns=["text"]), group_frame


def russian_ranges(pages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc_id, subset in pages.groupby("doc_id", sort=True):
        start = end = None
        for row in subset.sort_values("page_num").to_dict("records"):
            if row["russian_candidate"]:
                if start is None:
                    start = row["page_num"]
                end = row["page_num"]
            elif start is not None:
                rows.append({"doc_id": doc_id, "start_page": start, "end_page": end})
                start = end = None
        if start is not None:
            rows.append({"doc_id": doc_id, "start_page": start, "end_page": end})
    return pd.DataFrame(rows, columns=["doc_id", "start_page", "end_page"])


def question_analysis(
    path: Path | None, pages: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    if path is None or not path.exists():
        return pd.DataFrame(), {"status": "no_annotations", "total": 0}
    questions = pd.read_csv(path, keep_default_na=False, dtype=str)
    if questions.empty:
        return questions, {"status": "no_annotations", "total": 0}
    required = {
        "q_id",
        "question",
        "doc_id",
        "target_page",
        "question_type",
        "answer",
        "requires_visual",
        "evidence_notes",
    }
    if not required.issubset(questions.columns):
        raise ValueError(
            f"Missing QA columns: {sorted(required - set(questions.columns))}"
        )
    if questions["q_id"].eq("").any() or questions["q_id"].duplicated().any():
        raise ValueError("q_id must be non-empty and unique")
    if questions["question"].str.strip().eq("").any():
        raise ValueError("question must be non-empty")
    valid_targets = set(zip(pages["doc_id"], pages["page_num"]))
    for row in questions.to_dict("records"):
        number = float(row["target_page"])
        if not number.is_integer() or (row["doc_id"], int(number)) not in valid_targets:
            raise ValueError(f"Invalid QA target: {row['q_id']}")
        if row["question_type"] not in {
            "text",
            "table",
            "schematic",
            "mixed",
            "unknown",
        }:
            raise ValueError(f"Unknown question_type: {row['q_id']}")
        if row["requires_visual"] not in {"yes", "no", "unknown", ""}:
            raise ValueError(f"Invalid requires_visual: {row['q_id']}")
        if row["requires_visual"] in {"yes", "no"} and (
            not str(row["evidence_notes"]).strip() or not str(row["answer"]).strip()
        ):
            raise ValueError(f"Answer and evidence_notes required: {row['q_id']}")
    counts = Counter(questions["requires_visual"])
    total = len(questions)
    known = counts["yes"] + counts["no"]
    proportion = counts["yes"] / known if known else None
    interval = None
    if known:
        z = 1.959963984540054
        denominator = 1 + z * z / known
        center = (proportion + z * z / (2 * known)) / denominator
        half = (
            z
            * math.sqrt(proportion * (1 - proportion) / known + z * z / (4 * known**2))
            / denominator
        )
        interval = [max(0, center - half), min(1, center + half)]
    return questions, {
        "status": "annotated_sample",
        "total": total,
        "reviewed": known,
        "visual_yes": counts["yes"],
        "unknown": total - known,
        "visual_share_among_reviewed": proportion,
        "wilson_95_conditional": interval,
        "full_sample_bounds": [
            counts["yes"] / total,
            (counts["yes"] + total - known) / total,
        ],
        "question_types": dict(Counter(questions["question_type"])),
        "warning": "Descriptive sample only; Wilson interval assumes independent random sampling.",
    }


def review_queue(pages: pd.DataFrame, config: Config) -> pd.DataFrame:
    # Include every document, then add rare strata. This is a review queue, not a random prevalence sample.
    per_doc = pages.groupby("doc_id", group_keys=False).sample(
        n=1, random_state=config.seed
    )
    per_type = pd.concat(
        [
            frame.sample(n=min(8, len(frame)), random_state=config.seed)
            for _, frame in pages.groupby("page_type_candidate")
        ]
    )
    queue = (
        pd.concat([per_doc, per_type]).drop_duplicates("page_id").drop(columns=["text"])
    )
    queue = queue.sort_values(["doc_id", "page_num"])
    for column in [
        "has_picture",
        "has_table",
        "has_schematic",
        "text_only",
        "is_scan",
        "language_reviewed",
        "review_notes",
    ]:
        queue[column] = ""
    return queue


def enrich_text_quality(summary: dict, pages: pd.DataFrame, output: Path) -> None:
    """Measure whitespace padding from saved text, without repeating PDF extraction."""
    nonspace = {}
    with (output / "pages_text.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            nonspace[row["page_id"]] = len("".join(row["text"].split()))
    pages["nonspace_char_count"] = pages.page_id.map(nonspace)
    if pages.nonspace_char_count.isna().any():
        raise ValueError("Missing extracted text for a page")
    pages["whitespace_share"] = 1 - pages.nonspace_char_count.div(
        pages.char_count.replace(0, 1)
    )
    pages.loc[pages.char_count.eq(0), "whitespace_share"] = 0.0
    assert pages.whitespace_share.between(0, 1).all()
    flagged = pages[pages.whitespace_share.ge(0.9)].sort_values(
        "char_count", ascending=False
    )
    flagged.drop(columns=["text"], errors="ignore").to_csv(
        output / "text_quality_review.csv", index=False
    )
    pages.drop(columns=["text"], errors="ignore").to_csv(
        output / "pages.csv", index=False
    )
    summary["text_quality"] = {
        "whitespace_90pct_pages": len(flagged),
        "maximum_raw_chars": int(pages.char_count.max()),
        "maximum_nonspace_chars": int(pages.nonspace_char_count.max()),
        "note": "Raw char_count includes PDF alignment spaces, especially on RTL pages.",
    }
    summary["postprocessing_source_hash"] = sha256_file(Path(__file__))


def make_charts(pages: pd.DataFrame, documents: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 130})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].hist(
        documents["page_count"], bins=min(25, len(documents)), color="#34699a"
    )
    axes[0, 0].set(
        xlabel="Страниц в PDF", ylabel="Документов", title="Размер инструкций"
    )
    axes[0, 1].hist(
        pages["nonspace_char_count"].map(lambda n: math.log10(n + 1)),
        bins=50,
        color="#3b987d",
    )
    axes[0, 1].set(
        xlabel="log10(символов без пробелов + 1)",
        ylabel="Страниц",
        title="Извлечённый текст",
    )
    counts = pages["page_type_candidate"].value_counts()
    labels = {
        "scan_candidate": "Кандидат в сканы",
        "no_text_other": "Нет текста: прочее",
        "table_candidate": "Кандидат с таблицей",
        "graphics_candidate": "С графикой (эвристика)",
        "text_without_detected_graphics": "Текст, графика не выявлена",
    }
    axes[1, 0].barh([labels[v] for v in counts.index], counts.values, color="#b17a43")
    axes[1, 0].set(xlabel="Страниц", title="Взаимоисключающие автоматические категории")
    languages = pages["language_candidate"].value_counts().head(12)
    axes[1, 1].bar(languages.index, languages.values, color="#7970ae")
    axes[1, 1].tick_params(axis="x", labelrotation=45)
    axes[1, 1].set(
        xlabel="Код языка / unknown",
        ylabel="Страниц",
        title="Язык: автоматическая оценка",
    )
    fig.savefig(output / "overview.png")
    plt.close(fig)


def render_previews(pages: pd.DataFrame, pdf_dir: Path, output: Path) -> list[dict]:
    """Bounded contact sheet examples, preserving PDF page numbers (1-based)."""
    import matplotlib.pyplot as plt

    selected = pd.concat(
        [frame.head(2) for _, frame in pages.groupby("page_type_candidate", sort=True)]
    )
    preview_dir = output / "previews"
    preview_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(
        math.ceil(len(selected) / 3), 3, figsize=(13, 5 * math.ceil(len(selected) / 3))
    )
    rows = []
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, selected.to_dict("records")):
        with pymupdf.open(pdf_dir / row["filename"]) as document:
            page = document[row["page_num"] - 1]
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(0.8, 0.8), alpha=False)
            destination = preview_dir / f"{row['page_id']}.png"
            pixmap.save(destination)
        ax.imshow(plt.imread(destination))
        ax.set_title(
            f"{row['doc_id'][:38]}\np.{row['page_num']} · {row['page_type_candidate']}",
            fontsize=8,
        )
        rows.append(
            {
                k: row[k]
                for k in ["page_id", "doc_id", "page_num", "page_type_candidate"]
            }
        )
    fig.tight_layout()
    fig.savefig(output / "contact_sheet.png", dpi=110)
    plt.close(fig)
    return rows


def attach_document_review(summary: dict, documents: pd.DataFrame, root: Path) -> None:
    """Attach separately versioned human quality decisions without modifying the PDFs."""
    path = root / "data/meta/document_review.csv"
    review = (
        pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()
    )
    excluded = []
    for row in review.to_dict("records"):
        filename = row["doc_id"] + ".pdf"
        if summary["file_hashes"].get(filename) != row["file_sha256"]:
            raise ValueError(f"Review refers to a different PDF version: {filename}")
        if row["decision"] == "exclude":
            excluded.append(row["doc_id"])
    summary["document_review"] = {
        "review_file_hash": sha256_file(path) if path.exists() else None,
        "review_processing_source_hash": sha256_file(Path(__file__)),
        "excluded_documents": excluded,
        "excluded_pages": int(
            documents.loc[documents.doc_id.isin(excluded), "page_count"].sum()
        ),
        "notes": review.to_dict("records"),
        "scope": "Raw corpus statistics retain rejected files; remaining files are not all manually verified.",
    }


def report(summary: dict, output: Path) -> None:
    n = summary["page_count"]

    def rate(count):
        return f"{count} / {n} ({count / n:.1%})" if n else "нет страниц"

    qa = summary["qa"]
    review = summary.get("document_review", {})
    review_text = (
        "\n".join(
            f"- `{row['doc_id']}`: {row['decision']}. {row['reason']} Страницы: {row['evidence_pages']}."
            for row in review.get("notes", [])
        )
        or "Сплошная ручная проверка подлинности инструкций не выполнена."
    )
    question_text = (
        "Размеченных вопросов нет. Распределение типов и доля ответов, недоступных по тексту, "
        "**не определены**. Заполните `data/meta/benchmark_qa.csv` по инструкции в `docs/eda.md`."
        if not qa["total"]
        else json.dumps(qa, ensure_ascii=False, indent=2)
    )
    text = (
        f"""# EDA корпуса технических инструкций

Источник: https://drive.google.com/drive/folders/1rfNnKmHLSG3mkVbMDcy040v4phgAtNbp

Чеклист: https://docs.google.com/document/d/1eHSE9a3qsbgTl_w8lufTnGSJUmajbBfG4FGfDq1cbIg/edit

## Охват

- PDF в снимке Drive: {summary["expected_documents"]}.
- PDF найдены локально: {summary["found_documents"]}.
- Успешно прочитаны: {summary["document_count"]}; ошибки: {len(summary["errors"])}.
- Страниц в прочитанных документах: {n}. Пропущенные/повреждённые PDF не входят в знаменатель.
- Проверка охвата: {summary["coverage_status"]}.

### Ручная проверка качества источников

{review_text}

Решения записаны в `data/meta/document_review.csv` с SHA-256 проверенного файла.
Рекомендуется исключить документов: {len(review.get("excluded_documents", []))};
их страниц: {review.get("excluded_pages", 0)}. Статистика ниже описывает **исходные {summary["expected_documents"]} PDF**,
включая отклонённые источники; остальные документы не объявляются проверенными.
Исходники сохранены. Перед построением бенчмарка замените отклонённые документы настоящими инструкциями.

## 1. Страницы, таблицы, картинки и схемы

Категории ниже — автоматические кандидаты, с приоритетом скан → нет текста → таблица → графика → текст.
Они взаимоисключающие; исходные признаки в `pages.csv` могут пересекаться.

"""
        + "\n".join(
            f"- `{key}`: {rate(value)}." for key, value in summary["page_types"].items()
        )
        + f"""

Таблицы найдены геометрическим алгоритмом по линиям: {rate(summary["table_candidate_pages"])}.
Ошибок детектора: {summary["table_errors"]}; слишком сложных страниц пропущено: {summary["table_skipped"]}.
Таблицы без рамок и таблицы в сканах могут быть пропущены. Векторные линии включают рамки,
логотипы и украшения: это не классификатор электрических схем.
Точное число страниц с картинками, таблицами, схемами и только текстом требует ручной разметки.
Очередь `page_review_queue.csv` содержит по одной случайной странице каждого PDF и редкие категории;
её нельзя использовать для несмещённой оценки долей по всему корпусу.

## 2. Текстовый слой и сканы

- Нет извлекаемого текста: {rate(summary["no_text_pages"])}.
- Кандидат в скан (нет текста, объединённая площадь растров ≥80%): {rate(summary["scan_candidates"])}.
- Мало текста (<100 символов): {rate(summary["sparse_pages"])}.

Пустая страница, иллюстрация и скан могут иметь одинаковые признаки. OCR не запускался.
Скан с OCR-слоем попадёт в страницы с текстом; качество OCR автоматически не измерено.
Символы извлечены в порядке `sort=True`; это не гарантирует правильный порядок колонок.

### Артефакты пробелов

В текстовом слое некоторых страниц, в частности арабских, `sort=True` добавляет большое
число пробелов выравнивания. Максимальная исходная длина: {summary.get("text_quality", {}).get("maximum_raw_chars", "не рассчитано")}
символов; страниц с ≥90% пробельных символов: {summary.get("text_quality", {}).get("whitespace_90pct_pages", "не рассчитано")}.
Это артефакт извлечения, а не сопоставимый объём содержания.
`nonspace_char_count` и `whitespace_share` в `pages.csv` отделяют эту проблему;
`text_quality_review.csv` содержит страницы для проверки. Гистограмма показывает
`log10(nonspace_char_count + 1)` и сохраняет все страницы, включая пустые.
Исходный текст сохранён; токенизация для дубликатов схлопывает пробелы.

## 3. Языки и русские разделы

Языки-кандидаты: {json.dumps(summary["languages"], ensure_ascii=False)}.
Страниц с кандидатом `ru`: {summary["russian_candidates"]}.
`russian_ranges.csv` содержит непрерывные диапазоны страниц-кандидатов, нумерация PDF с 1.
Это очередь проверки, не готовые границы обрезки. Текстовые подсказки и вероятности `langdetect`
не калиброваны; обложки и страницы с несколькими языками требуют проверки.
Для общих схем и смешанных страниц нужен выбор блоков; удалять их целиком нельзя.
Автоматическое вырезание русского корпуса не выполнялось.

## 4. Дубликаты и разделение данных

- Пар совпадающих файлов SHA-256: {summary["exact_file_pairs"]}.
- Пар с равным нормализованным текстом документа: {summary["exact_text_pairs"]}.
- Всего пар-кандидатов (включая близкий текст): {summary["duplicate_candidate_pairs"]}.
- Повторяющихся страниц с ≥100 токенами: {summary["repeated_text_pages"]}.

Для близких документов используется Jaccard множеств 5-грамм токенов ≥0.65.
Это эвристика: общая техника безопасности может повысить сходство, сканы без текста не сравниваются.
Равный текст страниц не доказывает совпадение картинок. Проверяйте `duplicate_pairs.csv` и
`duplicate_groups.csv`; дополните группы семействами моделей вручную. Разделять страницы случайно
нельзя: все вопросы и страницы семейства/дубликата должны принадлежать одной части train/dev/test.
Общий корпус поиска может быть доступен всем запросам, но тестовые QA нельзя использовать
для выбора порогов, промптов, чанкинга или параметров ретривера. Разделение пока не выполнялось.

## 5–6. Вопросы и обоснование мультимодальности

{question_text}

`requires_visual=yes` означает: ответ подтверждён PDF, но извлечённого текста страницы
недостаточно для этого ответа (например, важны стрелки, взаимное расположение или обозначения).
Табличный вопрос может быть доступен по тексту; наличие изображения само по себе не означает `yes`.
Отдельно различайте недостаток извлечения текста/OCR и необходимость пространственного анализа.
Главный вывод о мультимодальности появится после сравнения ответов по PDF и текстовому слою
на заранее выбранной и описанной выборке. Метрики retrieval и accuracy здесь не вычислялись.

## Воспроизводимость

Параметры, SHA-256 файлов и версии зависимостей записаны в `summary.json` и `documents.csv`.
Seed: {summary["config"]["seed"]}. Все вычисления локальные; обучения, OCR, API и VLM нет.
Обзорные графики: `overview.png`; примеры исходных страниц: `contact_sheet.png`.
"""
    )
    (output / "report.md").write_text(text, encoding="utf-8")


def run(root: Path, *, refresh: bool = False) -> dict:
    config = Config()
    pdf_dir = root / "data/raw/drive_manuals"
    output = root / "reports/eda"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data/meta/drive_corpus.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = sorted(pdf_dir.glob("*.pdf"))
    if not files:
        raise FileNotFoundError(
            f"No PDFs in {pdf_dir}; see docs/eda.md for corpus preparation"
        )
    qa_path = root / "data/meta/benchmark_qa.csv"
    hashes = {p.name: sha256_file(p) for p in files}
    provenance = {
        "config": asdict(config),
        "file_hashes": hashes,
        "manifest_hash": sha256_file(manifest_path),
        "source_hash": sha256_file(Path(__file__)),
        "qa_hash": sha256_file(qa_path) if qa_path.exists() else None,
        "versions": {
            name: version(name)
            for name in ["pymupdf", "pandas", "langdetect", "matplotlib"]
        },
    }
    cache_key = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()
    ).hexdigest()
    summary_path = output / "summary.json"
    artifacts = [
        "pages.csv",
        "pages_text.jsonl",
        "documents.csv",
        "report.md",
        "overview.png",
        "contact_sheet.png",
        "duplicate_pairs.csv",
        "duplicate_groups.csv",
        "repeated_text_pages.csv",
        "russian_ranges.csv",
        "page_review_queue.csv",
    ]
    if (
        not refresh
        and summary_path.exists()
        and all((output / p).exists() for p in artifacts)
    ):
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("cache_key") == cache_key:
            print(
                f"Verified input hashes; using EDA cache: {previous['document_count']} PDFs"
            )
            return previous
    expected = {f["filename"]: f for f in manifest["files"]}
    missing = sorted(set(expected) - set(hashes))
    extra = sorted(set(hashes) - set(expected))
    wrong_size = [
        p.name
        for p in files
        if p.name in expected and p.stat().st_size != expected[p.name]["size_bytes"]
    ]
    records, docs, errors = [], [], []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}", flush=True)
        try:
            page_records, doc_record = extract_pdf(path, config)
            if not page_records:
                raise ValueError("PDF has no pages")
            records.extend(page_records)
            docs.append(doc_record)
        except Exception as error:  # noqa: BLE001 - isolate damaged PDFs and report coverage
            errors.append(
                {"filename": path.name, "error": f"{type(error).__name__}: {error}"}
            )
    if not records:
        raise ValueError(f"No readable PDF pages: {errors}")
    pages = pd.DataFrame(records)
    documents = pd.DataFrame(docs)
    assert not pages["page_id"].duplicated().any()
    assert len(pages) == int(documents["page_count"].sum())
    assert pages["image_coverage"].between(0, 1 + 1e-9).all()
    pairs, repeated, groups = duplicate_analysis(documents, pages, config)
    questions, qa = question_analysis(qa_path, pages)
    pages.drop(columns=["text"]).to_csv(output / "pages.csv", index=False)
    with (output / "pages_text.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    {k: record[k] for k in ["page_id", "doc_id", "page_num", "text"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    documents.drop(columns=["full_text"]).to_csv(output / "documents.csv", index=False)
    pairs.to_csv(output / "duplicate_pairs.csv", index=False)
    repeated.to_csv(output / "repeated_text_pages.csv", index=False)
    groups.to_csv(output / "duplicate_groups.csv", index=False)
    russian_ranges(pages).to_csv(output / "russian_ranges.csv", index=False)
    review_queue(pages, config).to_csv(output / "page_review_queue.csv", index=False)
    if not questions.empty:
        questions.to_csv(output / "questions.csv", index=False)
    previews = render_previews(pages, pdf_dir, output)
    summary = {
        **provenance,
        "cache_key": cache_key,
        "expected_documents": len(expected),
        "found_documents": len(files),
        "document_count": len(documents),
        "page_count": len(pages),
        "coverage_status": "complete"
        if not (missing or extra or wrong_size or errors)
        else "incomplete",
        "missing_files": missing,
        "extra_files": extra,
        "size_mismatches": wrong_size,
        "errors": errors,
        "page_types": {
            k: int(v) for k, v in pages["page_type_candidate"].value_counts().items()
        },
        "no_text_pages": int(pages["no_text_layer"].sum()),
        "scan_candidates": int(pages["scan_candidate"].sum()),
        "sparse_pages": int(pages["sparse_text"].sum()),
        "table_candidate_pages": int(pages["table_count_candidate"].gt(0).sum()),
        "table_errors": int(pages["table_detection_status"].eq("error").sum()),
        "table_skipped": int(
            pages["table_detection_status"].eq("skipped_complex_page").sum()
        ),
        "languages": {
            k: int(v) for k, v in pages["language_candidate"].value_counts().items()
        },
        "russian_candidates": int(pages["russian_candidate"].sum()),
        "exact_file_pairs": int(pairs["exact_file"].sum()),
        "exact_text_pairs": int(pairs["exact_text"].sum()),
        "duplicate_candidate_pairs": len(pairs),
        "repeated_text_pages": len(repeated),
        "qa": qa,
        "preview_pages": previews,
    }
    enrich_text_quality(summary, pages, output)
    make_charts(pages, documents, output)
    attach_document_review(summary, documents, root)
    report(summary, output)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Finished: {len(documents)} PDFs, {len(pages)} pages; {len(errors)} PDF errors",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), refresh=args.refresh)


if __name__ == "__main__":
    main()
