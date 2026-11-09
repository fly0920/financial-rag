# -*- coding: utf-8 -*-
"""依次跑 10 个评测问题，保存答案和检索快照（需先启动 qa_server）。"""
import json
import os
import time

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.getenv("QA_API", "http://127.0.0.1:8010/api/ask")


def main():
    questions = [json.loads(l) for l in open(os.path.join(BASE, "eval", "questions.jsonl"),
                                             encoding="utf-8")]
    out = []
    for q in questions:
        started = time.time()
        try:
            r = requests.post(API, json={"question": q["question"]}, timeout=300)
            r.raise_for_status()
            data = r.json()
        except Exception as e:  # noqa: BLE001
            data = {"answer": "ERROR: %r" % e, "citations": [], "retrieved": []}
        rec = dict(q)
        rec.update({"answer": data.get("answer", ""),
                    "citations": data.get("citations", []),
                    "retrieved": data.get("retrieved", []),
                    "seconds": round(time.time() - started, 1),
                    "correct": None, "note": ""})
        out.append(rec)
        print(q["id"], q["question"], "%.1fs" % rec["seconds"], flush=True)
        print(rec["answer"][:300].replace("\n", " "), flush=True)
        print("-" * 60, flush=True)
    path = os.path.join(BASE, "eval", "results.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
