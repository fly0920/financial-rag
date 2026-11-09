# -*- coding: utf-8 -*-
"""生成一页纸结论 PDF（A4，中文）。"""
import json
import os

import pymupdf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "eval", "一页结论.pdf")
FONT = r"C:\Windows\Fonts\msyh.ttc"


def load(path, default):
    try:
        return json.load(open(os.path.join(BASE, path), encoding="utf-8"))
    except Exception:
        return default


def main():
    meta = load("data/index/meta.json", {})
    results = load("eval/results.json", [])
    summary = load("eval/summary.json", {})
    total = summary.get("total", len(results) or 10)
    correct = summary.get("correct", 0)
    accuracy = summary.get("accuracy", (correct / total if total else 0))
    single_correct = summary.get("single_correct", 0)
    single_total = summary.get("single_total", 8)
    cross_correct = summary.get("cross_correct", 0)
    cross_total = summary.get("cross_total", 2)
    avg = summary.get("avg_seconds", 0)
    chunks = meta.get("chunks", "—")
    if isinstance(chunks, int):
        chunks = f"{chunks:,}"
    errors = summary.get("errors", [])
    limitations = summary.get("limitations", [])

    body = """银行年报问答知识库：一页结论

一、数据与处理
选取 10 家 A 股上市银行 2023 年年度报告，共 4,001 页。
用 pymupdf4llm 逐页转成 Markdown，表格按行列保留；
再按标题、段落和表格切块，每块带上公司、代码、年份、页码和章节。
最终得到 %s 个可检索片段。PDF 不进入仓库，仓库只保留从东方财富公开公告接口下载年报的脚本。

二、检索与回答
检索采用 BM25 与向量检索的混合方案：jieba 分词后建 BM25，
BAAI/bge-small-zh-v1.5 生成 512 维归一化向量，按 45%% / 55%% 加权融合。
单公司问题先用“指标名 + 数值”的词法规则锁定原文，再取全局 top-8；
跨公司问题按公司分别检索，每家公司取 2 个片段，
避免上下文被一两家银行占满。大模型只负责组织语言，
关键数字必须带 [资料i]，页面同时列出公司、年份和页码供核对。

三、评测结果
按 8 个单公司事实题和 2 个跨公司比较题评测，答对 %d / %d
（准确率 %.0f%%；单公司 %d/%d，跨公司 %d/%d），平均响应 %.1f 秒。
单公司题能直接命中“主要会计数据”“财务指标”等表；
跨公司题采用逐家公司检索后，10 家公司的指标能同时进入上下文。

四、错误与限制
%s

五、结论
这个方案的价值不在“让模型背年报”，而在把答案钉在可核对的页面上：
检索负责找证据，模型负责归纳，引用负责可验证。
后续可以增加 reranker、按指标预抽取结构化表，以及继续完善
“净息差/净利息收益率”等同义表述的查询扩展。

附：代码、评测问题和逐题检索记录见仓库；页面截图见 screenshots/。""" % (
        chunks, correct, total, accuracy * 100, single_correct, single_total,
        cross_correct, cross_total, avg,
        ("本次评测未发现答错题；仍需注意：\n" + "\n".join("· " + e for e in limitations))
        if not errors else "\n".join("· " + e for e in errors))

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    rect = pymupdf.Rect(48, 42, 547, 800)
    size = 9.6
    while size > 8:
        rc = page.insert_textbox(rect, body, fontname="msyh", fontfile=FONT,
                                 fontsize=size, lineheight=1.42, align=0)
        if rc >= 0:
            break
        page.clean_contents()
        size -= 0.3
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        doc.subset_fonts()
    except Exception as e:  # noqa: BLE001
        print("subset_fonts skipped", e)
    doc.save(OUT, garbage=4, deflate=True)
    print("saved", OUT, "fontsize", size)
    print("leftover", rc)


if __name__ == "__main__":
    main()
