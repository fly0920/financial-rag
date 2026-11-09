# -*- coding: utf-8 -*-
"""从东方财富公告接口下载 10 家银行 2023 年年度报告。

作业要求仓库里不放 PDF，只放这个脚本；运行后 PDF 落在 data/pdfs/。
"""
import os
import re
import sys
import time

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE, "data", "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

COMPANIES = [
    ("600000", "浦发银行"), ("600016", "民生银行"), ("600036", "招商银行"),
    ("601166", "兴业银行"), ("601288", "农业银行"), ("601328", "交通银行"),
    ("601398", "工商银行"), ("601939", "建设银行"), ("601988", "中国银行"),
    ("601998", "中信银行"),
]

API = "https://np-anotice-stock.eastmoney.com/api/security/ann"
PDF = "https://pdf.dfcfw.com/pdf/H2_{}_1.pdf"
BAD = ("摘要", "补充", "更正", "英文", "H股", "公告", "问询", "说明")


def find_report(code):
    for page in range(1, 9):
        params = {"sr": -1, "page_size": 100, "page_index": page, "ann_type": "A",
                  "client_source": "web", "stock_list": code, "f_node": 1}
        r = requests.get(API, params=params, timeout=20)
        r.raise_for_status()
        items = r.json()["data"]["list"]
        if not items:
            break
        for it in items:
            title = it.get("title", "")
            date = (it.get("notice_date") or "")[:10]
            if "2023" not in title or "年度报告" not in title:
                continue
            if any(b in title for b in BAD):
                continue
            if not date.startswith("2024"):
                continue
            return it["art_code"], title, date
        if (items[-1].get("notice_date") or "")[:4] < "2024":
            break
    return None, None, None


def main():
    for code, name in COMPANIES:
        out = os.path.join(PDF_DIR, "%s_%s_2023.pdf" % (code, name))
        if os.path.exists(out) and os.path.getsize(out) > 2_000_000:
            print("SKIP", code, name, os.path.getsize(out))
            continue
        art, title, date = find_report(code)
        if not art:
            print("NOT FOUND", code, name, file=sys.stderr)
            continue
        r = requests.get(PDF.format(art), timeout=120)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF") or len(r.content) < 2_000_000:
            print("BAD FILE", code, title, len(r.content), file=sys.stderr)
            continue
        with open(out, "wb") as f:
            f.write(r.content)
        print("OK", code, name, date, title, len(r.content))
        time.sleep(0.8)


if __name__ == "__main__":
    main()
