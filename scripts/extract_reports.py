# -*- coding: utf-8 -*-
"""把 10 份银行年报 PDF 转成逐页 Markdown（表格保留行列），供切块和索引使用。"""
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pymupdf
import pymupdf4llm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE, "data", "pdfs")
OUT_DIR = os.path.join(BASE, "data", "pages")
os.makedirs(OUT_DIR, exist_ok=True)

CHUNK = 30


def parse_name(fn):
    m = re.match(r"(\d{6})_(.+)_2023\.pdf$", fn)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def extract_chunk(pdf_path, start, end):
    doc = pymupdf.open(pdf_path)
    out = []
    for i in range(start, min(end, doc.page_count)):
        try:
            md = pymupdf4llm.to_markdown(pdf_path, pages=[i])
        except Exception as e:  # noqa: BLE001
            md = doc[i].get_text()
            md = "[extract fallback %r]\n%s" % (e, md)
        out.append({"page": i + 1, "markdown": md})
    doc.close()
    return out


def main():
    files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    tasks = []
    for fn in files:
        code, name = parse_name(fn)
        if not code:
            print("skip", fn, file=sys.stderr)
            continue
        path = os.path.join(PDF_DIR, fn)
        doc = pymupdf.open(path)
        for start in range(0, doc.page_count, CHUNK):
            tasks.append((code, name, path, start, min(start + CHUNK, doc.page_count)))
        print("%s %s pages=%d" % (code, name, doc.page_count))
        doc.close()

    started = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(extract_chunk, path, start, end): (code, name, start, end)
                for code, name, path, start, end in tasks}
        done = 0
        for fut in as_completed(futs):
            code, name, start, end = futs[fut]
            try:
                pages = fut.result()
            except Exception as e:  # noqa: BLE001
                print("FAIL", code, start, end, repr(e), file=sys.stderr)
                continue
            results.setdefault((code, name), []).extend(pages)
            done += 1
            if done % 10 == 0:
                print("progress %d/%d %.1fs" % (done, len(tasks), time.time() - started), file=sys.stderr)

    for (code, name), pages in results.items():
        pages.sort(key=lambda x: x["page"])
        path = os.path.join(OUT_DIR, "%s_%s.jsonl" % (code, name))
        with open(path, "w", encoding="utf-8") as f:
            for p in pages:
                f.write(json.dumps({"code": code, "company": name, "year": 2023,
                                    "page": p["page"], "markdown": p["markdown"]},
                                   ensure_ascii=False) + "\n")
        print("saved", path, len(pages))
    print("total %.1fs" % (time.time() - started))


if __name__ == "__main__":
    main()
