# -*- coding: utf-8 -*-
"""跑完索引后，抽查每家银行的关键指标检索结果，便于人工核对标准答案。"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
import qa_server  # noqa: E402

qa_server.load()
companies = sorted({c["company"] for c in qa_server.state["chunks"]})
out = {}
for metric in ("不良贷款率", "净息差"):
    out[metric] = {}
    for company in companies:
        hits = qa_server.retrieve("%s 2023年 %s" % (company, metric), 2)
        out[metric][company] = [{"company": h["company"], "page": h["page"],
                                 "section": h["section"], "text": h["text"][:700]}
                                for h in hits]
json.dump(out, open(os.path.join(BASE, "eval", "metric_probe.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("saved eval/metric_probe.json")


