# -*- coding: utf-8 -*-
"""把逐页 Markdown 切成带公司/页码/章节元数据的块。"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE_DIR = os.path.join(BASE, "data", "pages")
OUT = os.path.join(BASE, "data", "chunks.jsonl")

HEAD_RE = re.compile(r"^(#{1,6})\s+(.+)$")
NUM_HEAD_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\s+([^\n]{2,40})$")
NOISE = re.compile(r"^(第?\s*\d+\s*页?|\d{1,3}|招商银行股份有限公司|中国工商银行股份有限公司|"
                   r"中国建设银行股份有限公司|中国农业银行股份有限公司|中国银行股份有限公司|"
                   r"交通银行股份有限公司|兴业银行股份有限公司|上海浦东发展银行股份有限公司|"
                   r"中信银行股份有限公司|中国民生银行股份有限公司|20\d\d\s*年?度?报告.*)$")


def clean_page(md):
    lines = []
    for line in md.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if NOISE.match(s):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines)


def is_table_block(block):
    rows = [l for l in block.splitlines() if l.strip().startswith("|")]
    return len(rows) >= 2


def split_prose(block, max_chars=800):
    """长段落按句号/分号切开，避免超过嵌入模型 512 token 被截断。"""
    parts = re.split(r"(?<=[。；！？])", block)
    out, buf = [], ""
    for part in parts:
        if not part:
            continue
        if len(buf) + len(part) > max_chars and buf:
            out.append(buf)
            buf = part
        else:
            buf += part
    if buf:
        out.append(buf)
    return out


def split_table(block, max_chars=1800):
    lines = [l for l in block.splitlines() if l.strip()]
    if len(block) <= max_chars:
        return [block]
    header = lines[:2]
    out, buf = [], list(header)
    for row in lines[2:]:
        if sum(len(x) for x in buf) + len(row) > max_chars and len(buf) > 2:
            out.append("\n".join(buf))
            buf = list(header)
        buf.append(row)
    if len(buf) > 2:
        out.append("\n".join(buf))
    return out


def flush(buf, meta, section, chunks):
    text = "\n".join(buf).strip()
    if len(text) < 40:
        return
    chunks.append({"text": text, "company": meta["company"], "code": meta["code"],
                   "year": meta["year"], "page": meta["page"], "section": section})


def main():
    files = sorted(f for f in os.listdir(PAGE_DIR) if f.endswith(".jsonl"))
    chunks = []
    for fn in files:
        section = ""
        with open(os.path.join(PAGE_DIR, fn), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                md = clean_page(rec.get("markdown", ""))
                blocks = re.split(r"\n\s*\n", md)
                buf = []
                for block in blocks:
                    block = block.strip("\n")
                    if not block:
                        continue
                    m = HEAD_RE.match(block.splitlines()[0]) if block.splitlines() else None
                    n = NUM_HEAD_RE.match(block.splitlines()[0]) if block.splitlines() else None
                    if m:
                        flush(buf, rec, section, chunks); buf = []
                        section = m.group(2).strip()[:60]
                    elif n and len(block.splitlines()) == 1:
                        flush(buf, rec, section, chunks); buf = []
                        section = (n.group(1) + " " + n.group(2)).strip()[:60]
                    if is_table_block(block):
                        flush(buf, rec, section, chunks); buf = []
                        for part in split_table(block):
                            chunks.append({"text": part, "company": rec["company"], "code": rec["code"],
                                           "year": rec["year"], "page": rec["page"], "section": section})
                        continue
                    if len(block) > 1200:
                        flush(buf, rec, section, chunks); buf = []
                        for part in split_prose(block):
                            if len(part.strip()) >= 40:
                                chunks.append({"text": part.strip(), "company": rec["company"],
                                               "code": rec["code"], "year": rec["year"],
                                               "page": rec["page"], "section": section})
                        continue
                    buf.append(block)
                    if sum(len(x) for x in buf) > 900:
                        flush(buf, rec, section, chunks)
                        # 保留最后一段作为重叠
                        buf = [buf[-1]] if len(buf[-1]) < 300 else []
                flush(buf, rec, section, chunks)
        print("done", fn, "chunks", len(chunks), file=sys.stderr)

    # 去重（同公司同页同文本）
    seen = set()
    unique = []
    for c in chunks:
        key = (c["code"], c["page"], c["text"][:120])
        if key in seen:
            continue
        seen.add(key)
        c["id"] = len(unique)
        unique.append(c)

    with open(OUT, "w", encoding="utf-8") as f:
        for c in unique:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("chunks", len(unique), "->", OUT)


if __name__ == "__main__":
    main()
