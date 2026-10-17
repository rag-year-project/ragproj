#!/usr/bin/env python3
"""
Скрипт текстового бейзлайна (Чекпоинт 3):
1. Индексация текстовых страниц (BM25 + TF-IDF).
2. Загрузка benchmark_qa.json.
3. Прогон поисковых запросов и расчет метрик:
   - HitRate@1, HitRate@3, HitRate@5 (находит ли ретривер нужную страницу).
   - MRR (Mean Reciprocal Rank).
   - Анализ просадки по типам вопросов (text vs table vs schematic).

Требования:
    pip install rank_bm25 tqdm
Запуск:
    python evaluate_text_baseline.py
"""

import json
import os
import re

from rank_bm25 import BM25Okapi
from tqdm import tqdm

PAGES_JSONL = "./corpus_pages.jsonl"
BENCHMARK_JSON = "./benchmark_qa.json"

if not os.path.exists(PAGES_JSONL):
    print(f"[!] Файл {PAGES_JSONL} не найден. Сначала запустите preprocess_corpus.py!")
    exit(1)

if not os.path.exists(BENCHMARK_JSON):
    print(
        f"[!] Файл {BENCHMARK_JSON} не найден. Разметьте вопросы через labeling_tool.html!"
    )
    exit(1)


# Простая токенизация
def tokenize(text):
    return re.findall(r"\w+", text.lower())


print("[*] Загрузка страниц корпуса...")
corpus_pages = []
with open(PAGES_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        corpus_pages.append(json.loads(line))

print(f"[*] Загружено {len(corpus_pages)} страниц. Построение BM25 индекса...")
tokenized_corpus = [tokenize(p["text"]) for p in tqdm(corpus_pages, desc="Токенизация")]
bm25 = BM25Okapi(tokenized_corpus)

print(f"[*] Загрузка тестового бенчмарка: {BENCHMARK_JSON}...")
with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
    queries = json.load(f)

print(f"[*] Запуск оценки на {len(queries)} вопросах...")

top_k_hits = {1: 0, 3: 0, 5: 0, 10: 0}
mrr_sum = 0.0

category_stats = {}

for q in queries:
    q_text = q["question"]
    target_page = q["target_page"]
    doc_id = q.get("doc_id", "")
    q_type = q.get("question_type", "general")

    if q_type not in category_stats:
        category_stats[q_type] = {"total": 0, "hit1": 0, "hit5": 0}
    category_stats[q_type]["total"] += 1

    tokenized_query = tokenize(q_text)
    scores = bm25.get_scores(tokenized_query)

    # Сортировка по убыванию релевантности
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    retrieved_pages = [corpus_pages[i] for i in top_indices]

    # Проверка ранга правильной страницы
    rank = None
    for r_idx, r_page in enumerate(retrieved_pages, start=1):
        # Совпадение по номеру страницы и doc_id (если указан)
        if r_page["page_num"] == target_page:
            if not doc_id or doc_id in r_page["doc_id"]:
                rank = r_idx
                break

    if rank is not None:
        mrr_sum += 1.0 / rank
        for k in top_k_hits:
            if rank <= k:
                top_k_hits[k] += 1
        if rank <= 1:
            category_stats[q_type]["hit1"] += 1
        if rank <= 5:
            category_stats[q_type]["hit5"] += 1

total_q = len(queries)
print("=" * 65)
print("РЕЗУЛЬТАТЫ ТЕКСТОВОГО БЕЙЗЛАЙНА (BM25 Retrieval)")
print("=" * 65)
print(f"Total Questions:  {total_q}")
print(f"HitRate@1:        {top_k_hits[1] / total_q:.2%}  ({top_k_hits[1]}/{total_q})")
print(f"HitRate@3:        {top_k_hits[3] / total_q:.2%}  ({top_k_hits[3]}/{total_q})")
print(f"HitRate@5:        {top_k_hits[5] / total_q:.2%}  ({top_k_hits[5]}/{total_q})")
print(f"MRR (Mean Rec.):  {mrr_sum / total_q:.3f}")
print("-" * 65)
print("РАЗБИВКА ПО ТИПАМ ВОПРОСОВ (Где проседает текст):")
for cat, st in category_stats.items():
    t = st["total"]
    h1_pct = (st["hit1"] / t) * 100 if t > 0 else 0
    h5_pct = (st["hit5"] / t) * 100 if t > 0 else 0
    print(
        f" • {cat:<18}: {t:>2} вопр. | Hit@1: {h1_pct:>5.1f}% | Hit@5: {h5_pct:>5.1f}%"
    )
print("=" * 65)
