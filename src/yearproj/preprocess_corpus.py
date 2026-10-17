#!/usr/bin/env python3
"""
Скрипт предобработки PDF-мануалов:
1. Постраничное извлечение текста с метаданными (doc_id, page_num, char_count).
2. Рендеринг страниц в PNG высокого разрешения (150 DPI) для разметки Bounding Box и VLM.
3. Сохранение структуры корпуса в JSONL для бейзлайна (BM25 / Dense / ColPali).

Требования:
    pip install pymupdf tqdm pillow
Запуск:
    python preprocess_corpus.py
"""

import json
import os

import fitz  # PyMuPDF
from tqdm import tqdm

PDF_DIR = "./data/pdf_manuals"
OUTPUT_PAGES_DIR = "./data/corpus_pages_images"
PROCESSED_JSONL = "./data/jsons/corpus_pages.jsonl"
CORPUS_META_JSON = "./data/jsons/corpus_manifest.json"

os.makedirs(OUTPUT_PAGES_DIR, exist_ok=True)

# Ищем все PDF в директории
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
pdf_files.sort()

if not pdf_files:
    print(
        f"[!] В папке {PDF_DIR} не найдено PDF-файлов. Убедитесь, что положили мануалы в {PDF_DIR}."
    )
    exit(1)

print(f"[*] Найдено {len(pdf_files)} PDF-документов для обработки.")
print(f"[*] Изображения страниц будут сохранены в: {OUTPUT_PAGES_DIR}")
print(f"[*] Текстовый индекс страниц: {PROCESSED_JSONL}")
print("=" * 70)

all_pages_records = []
manifest = []

# Коэффициент масштабирования для рендера страниц (150 DPI = 150 / 72 ≈ 2.08)
ZOOM_FACTOR = 150 / 72
mat = fitz.Matrix(ZOOM_FACTOR, ZOOM_FACTOR)

total_pages_processed = 0

for doc_idx, pdf_name in enumerate(pdf_files, start=1):
    pdf_path = os.path.join(PDF_DIR, pdf_name)
    doc_id = os.path.splitext(pdf_name)[0]

    # Создаем персональную папку для рендера страниц документа
    doc_images_dir = os.path.join(OUTPUT_PAGES_DIR, doc_id)
    os.makedirs(doc_images_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Не удалось открыть {pdf_name}: {e}")
        continue

    num_pages = len(doc)
    doc_char_count = 0

    print(f"[{doc_idx:02d}/{len(pdf_files)}] Обработка: {doc_id} ({num_pages} стр.)...")

    for page_idx in tqdm(range(num_pages), desc=f"   Стр. {doc_id[:25]}", leave=False):
        page_num = page_idx + 1
        page = doc[page_idx]

        # 1. Извлечение текста страницы
        text = page.get_text("text").strip()
        doc_char_count += len(text)

        # 2. Рендеринг страницы в PNG для разметки BBox и мультимодального поиска
        img_filename = f"page_{page_num:03d}.png"
        img_rel_path = os.path.join(doc_images_dir, img_filename)

        if not os.path.exists(img_rel_path):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(img_rel_path)

        page_record = {
            "page_id": f"{doc_id}_p{page_num:03d}",
            "doc_id": doc_id,
            "page_num": page_num,
            "text": text,
            "char_count": len(text),
            "image_path": img_rel_path,
            "page_width": page.rect.width,
            "page_height": page.rect.height,
            "has_images": len(page.get_images()) > 0,
        }
        all_pages_records.append(page_record)
        total_pages_processed += 1

    manifest.append(
        {
            "doc_id": doc_id,
            "filename": pdf_name,
            "pages_count": num_pages,
            "total_chars": doc_char_count,
            "avg_chars_per_page": round(doc_char_count / num_pages, 1)
            if num_pages > 0
            else 0,
        }
    )
    doc.close()

# Сохранение текстовой базы в JSONL (одна страница на строку)
with open(PROCESSED_JSONL, "w", encoding="utf-8") as f:
    f.writelines(
        json.dumps(rec, ensure_ascii=False) + "\n" for rec in all_pages_records
    )

# Сохранение метаданных корпуса
with open(CORPUS_META_JSON, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("=" * 70)
print("Готово!")
print(f" Всего обработано документов: {len(manifest)}")
print(f" Всего страниц в базе:       {total_pages_processed}")
print(f" Изображения страниц:        {OUTPUT_PAGES_DIR}/")
print(f" Текстовый индекс страниц:   {PROCESSED_JSONL}")
print(f" Манифест корпуса:           {CORPUS_META_JSON}")
