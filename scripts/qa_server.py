# -*- coding: utf-8 -*-
"""财报问答页面：BM25 + 向量混合检索，答案带公司/页码出处。"""
import json
import os
import pickle
import re
import shutil
import subprocess
import sys

import jieba
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(BASE, "data", "index")
WEB = os.path.join(BASE, "web", "index.html")

app = FastAPI(title="银行年报问答")
state = {}


def tokenize(text):
    return [w for w in jieba.cut_for_search(text) if w.strip()]


@app.on_event("startup")
def load():
    chunks = [json.loads(l) for l in open(os.path.join(INDEX, "chunks.jsonl"), encoding="utf-8")]
    emb = np.load(os.path.join(INDEX, "embeddings.npy"))
    with open(os.path.join(INDEX, "bm25.pkl"), "rb") as f:
        bm = pickle.load(f)
    model = SentenceTransformer(json.load(open(os.path.join(INDEX, "meta.json"), encoding="utf-8"))["model"],
                                device="cpu")
    state.update(chunks=chunks, emb=emb, bm25=bm["bm25"], model=model)
    print("loaded", len(chunks), "chunks")


AGG_RE = re.compile(r"最高|最低|排名|排序|比较|对比|哪些|哪家|各家|分别|谁|全部|所有")
METRICS = ["核心一级资本充足率", "归属于本行股东的净利润", "不良贷款率", "净利息收益率",
           "净息差", "资本充足率", "拨备覆盖率", "营业收入", "净利润", "总资产",
           "贷款总额", "存款总额"]
# 同一指标在不同年报里的叫法不同，检索时一起找
METRIC_SYNONYMS = {
    "不良贷款率": ["不良贷款率", "不良率"],
    "净息差": ["净息差", "净利息收益率", "NIM"],
    "净利息收益率": ["净利息收益率", "净息差", "NIM"],
    "归属于本行股东的净利润": ["归属于本行股东的净利润", "归母净利润"],
    "营业收入": ["营业收入", "营收"],
    "净利润": ["净利润"],
    "资本充足率": ["资本充足率"],
    "核心一级资本充足率": ["核心一级资本充足率"],
    "拨备覆盖率": ["拨备覆盖率"],
    "总资产": ["总资产"],
    "贷款总额": ["贷款总额", "贷款余额"],
    "存款总额": ["存款总额", "存款余额"],
}


def detect_metric(question):
    for m in METRICS:
        if m in question:
            return m
    return ""


def _value_re(syn):
    """指标名后紧跟的百分比数值；遇到数字就停，避免越过第一个数去匹配后面的指标。"""
    return re.compile(re.escape(syn) + r"[^0-9%\n]{0,40}?(\d+(?:\.\d+)?)\s*%")


SEGMENT_WORDS = ("小微", "零售", "个人住房", "信用卡", "房地产", "公司类", "公司贷款",
                 "县域", "境外", "子公司", "分行", "行业", "普惠", "消费贷", "按揭", "对公")
PERCENT_METRICS = ("不良贷款率", "净息差", "净利息收益率", "拨备覆盖率",
                   "资本充足率", "核心一级资本充足率")
LEVEL_METRICS = ("不良贷款率", "净息差", "净利息收益率")


def _amount_res(syn):
    return [
        re.compile(re.escape(syn) + r"[^0-9\n]{0,40}?([0-9][0-9,]*(?:\.\d+)?)\s*(?:亿元|百万元|万元)"),
        # 表格行：|营业收入|210,831|...
        re.compile(re.escape(syn) + r"[^0-9\n]{0,12}?\|?\s*([0-9][0-9,]{2,})"),
    ]


def _lexical_candidates(company, synonyms, limit=8, metric=""):
    """按“指标名 + 百分比”做词法筛选，保证每家银行都能找到指标原文。

    同一个 chunk 里可能出现多个百分比（分项、上年数、公式），因此逐个匹配，
    用匹配点前后的局部文字判断是不是“集团/全行”口径的水平值。
    """
    chunks = state["chunks"]
    rows = []
    for i, c in enumerate(chunks):
        if c["company"] != company:
            continue
        text = c["text"]
        clean = re.sub(r"<sup>.*?</sup>", "", text)
        # PDF 抽取常在汉字之间夹空格/换行（如“不 良率”），匹配前先去掉
        clean = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", clean)
        sec = c.get("section", "")
        best = None
        for syn in synonyms:
            patterns = [_value_re(syn)] if metric in PERCENT_METRICS else _amount_res(syn)
            for pat in patterns:
                for m in pat.finditer(clean):
                    val = m.group(1)
                    # 只取同一句/同一表格行，避免把后一句的“房地产”“个人”算进分项
                    lefts = [clean.rfind(p, 0, m.start()) for p in ("。", "；", "\n")]
                    left = max(max(lefts), m.start() - 45, 0)
                    rights = [clean.find(p, m.end()) for p in ("。", "；", "\n")]
                    rights = [r for r in rights if r != -1]
                    right = min(rights) if rights else min(len(clean), m.end() + 25)
                    ctx = clean[left:right]
                    # 公式、口径说明里的 100% 之类
                    if any(w in ctx for w in ("计算", "公式", "×100", "＝", "=")):
                        continue
                    if metric in LEVEL_METRICS and (float(val) >= 20 or float(val) <= 0):
                        continue
                    s = 0.0
                    if "集团" in ctx:
                        s += 3.0
                    if "全行" in ctx:
                        s += 3.0
                    if "本行" in ctx:
                        s += 1.5
                    if "资产质量" in ctx:
                        s += 2.0
                    if any(k in sec for k in ("主要会计数据", "财务指标", "财务摘要", "资产质量", "净息差", "净利息", "经营业绩")):
                        s += 2.0
                    if any(k in ctx for k in SEGMENT_WORDS):
                        s -= 4.0
                    if len(text) < 400:
                        s += 0.8
                    # 表格行：指标名在同一行靠前位置
                    line_start = clean.rfind("\n", 0, m.start()) + 1
                    line = clean[line_start:clean.find("\n", m.end()) if clean.find("\n", m.end()) != -1 else len(clean)]
                    if line.lstrip().startswith("|") and syn in line[:40]:
                        s += 2.0
                    if best is None or s > best[0]:
                        best = (s, i, val, ctx.replace("\n", " "))
        if best:
            rows.append(best)
    # 同一页的重复 chunk 只留分数最高的一个
    rows.sort(key=lambda x: (-x[0], len(chunks[x[1]]["text"])))
    out, seen = [], set()
    for r in rows:
        page = chunks[r[1]]["page"]
        if page in seen:
            continue
        seen.add(page)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def _score(question, idx, qv=None, bm_all=None):
    """在给定候选下标上计算向量 + BM25 混合分。"""
    chunks, emb, bm25, model = state["chunks"], state["emb"], state["bm25"], state["model"]
    if qv is None:
        qv = model.encode([question], normalize_embeddings=True)[0]
    if bm_all is None:
        bm_all = np.array(bm25.get_scores(tokenize(question)), dtype=float)
    vec_scores = emb[idx] @ qv
    bm_scores = bm_all[idx]

    def norm(x):
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo + 1e-9)

    return 0.55 * norm(vec_scores) + 0.45 * norm(bm_scores), vec_scores, bm_scores


def _pack(i, hybrid, vec_scores, bm_scores, pos):
    c = state["chunks"][int(i)]
    return {"id": c["id"], "company": c["company"], "code": c["code"],
            "year": c["year"], "page": c["page"], "section": c.get("section", ""),
            "score": round(float(hybrid[pos]), 4),
            "vector": round(float(vec_scores[pos]), 4),
            "bm25": round(float(bm_scores[pos]), 2),
            "text": c["text"]}


def _metric_bonus(text, section, metric):
    """让“不良贷款率 1.36%”“净息差为1.59%”这类带数值的句子排到前面。"""
    if not metric:
        return 0.0
    bonus = 0.0
    if metric in text:
        bonus += 0.12
        if re.search(re.escape(metric) + r"[^。；\n]{0,25}?\d", text):
            bonus += 0.10
    if "主要会计数据" in section or "财务摘要" in section or "财务指标" in section:
        bonus += 0.05
    return bonus


def retrieve_cross(question, per_company=2):
    """跨公司问题：每家公司各检索一次，避免某一家银行占满上下文。"""
    chunks, emb, bm25, model = state["chunks"], state["emb"], state["bm25"], state["model"]
    metric = detect_metric(question)
    synonyms = METRIC_SYNONYMS.get(metric, [metric] if metric else [])
    # 用“指标 + 2023年 + 集团”检索，避免“最低/哪家”这类词把讨论段落顶上来
    sub_q = (metric + " 2023年 集团 主要会计数据 财务指标") if metric else question
    qv = model.encode([sub_q], normalize_embeddings=True)[0]
    bm_all = np.array(bm25.get_scores(tokenize(sub_q)), dtype=float)
    companies = sorted({c["company"] for c in chunks})
    picked = []
    for company in companies:
        idx = np.array([i for i, c in enumerate(chunks) if c["company"] == company])
        pos_map = {int(i): p for p, i in enumerate(idx)}
        hybrid, vec_scores, bm_scores = _score(question, idx, qv=qv, bm_all=bm_all)

        chosen = []
        # 先用“指标名 + 数值”的词法结果，确保水平值进入上下文
        if synonyms:
            for _, i, _, _ in _lexical_candidates(company, synonyms, limit=8, metric=metric):
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= per_company:
                    break
        # 不够再用混合检索补齐
        if len(chosen) < per_company:
            for pos in np.argsort(-hybrid):
                i = int(idx[pos])
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= per_company:
                    break
        for i in chosen:
            picked.append(_pack(i, hybrid, vec_scores, bm_scores, pos_map[i]))
    return picked


def retrieve(question, top_k=8):
    chunks = state["chunks"]
    named = {c["company"] for c in chunks if c["company"] in question}
    if not named and AGG_RE.search(question):
        return retrieve_cross(question)

    idx = np.arange(len(chunks))
    hybrid, vec_scores, bm_scores = _score(question, idx)
    # 问题里点名某家银行时，优先该公司
    if named:
        hybrid = hybrid + np.array([0.08 if c["company"] in named else 0 for c in chunks])

    order = []
    metric = detect_metric(question)
    # 单家公司 + 明确指标：先把“指标名 + 数值”的原文找出来
    if named and metric:
        synonyms = METRIC_SYNONYMS.get(metric, [metric])
        for company in sorted(named):
            for _, i, _, _ in _lexical_candidates(company, synonyms, limit=top_k, metric=metric):
                if int(i) not in order:
                    order.append(int(i))
    # 先补同公司，再补全局
    if named:
        for i in np.argsort(-hybrid):
            if len(order) >= top_k:
                break
            if chunks[int(i)]["company"] in named and int(i) not in order:
                order.append(int(i))
    for i in np.argsort(-hybrid):
        if len(order) >= top_k:
            break
        if int(i) not in order:
            order.append(int(i))
    return [_pack(i, hybrid, vec_scores, bm_scores, i) for i in order[:top_k]]


def call_claude(prompt, timeout=240):
    exe = shutil.which("claude") or shutil.which("claude.cmd")
    if not exe:
        return None
    cmd = [os.environ.get("COMSPEC", "cmd.exe"), "/c", exe, "-p", "--max-turns", "1"]
    try:
        p = subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", errors="replace",
                           capture_output=True, timeout=timeout)
        if p.returncode == 0 and (p.stdout or "").strip():
            return p.stdout.strip()
    except Exception as e:  # noqa: BLE001
        print("claude error", repr(e), file=sys.stderr)
    return None


class Ask(BaseModel):
    question: str


@app.post("/api/ask")
def ask(req: Ask):
    q = (req.question or "").strip()
    if not q:
        return JSONResponse({"error": "问题不能为空"}, status_code=400)
    hits = retrieve(q, 8)
    ctx = []
    for i, h in enumerate(hits, 1):
        ctx.append("[资料%d] %s %d年报 第%d页（%s）\n%s" % (
            i, h["company"], h["year"], h["page"], h["section"] or "正文", h["text"][:1000]))
    prompt = """你是银行年报分析助手。只根据下面检索到的年报片段回答问题。

要求：
1. 先给结论，再给必要的数据和解释；
2. 每个关键数字后用 [资料i] 标注出处，并写明“公司 年报 第X页”；
3. 如果片段里没有答案，直接说“检索到的年报片段中没有这个信息”，不要编造；
4. 中文回答。单家公司问题控制在 300 字以内；跨公司比较先给一句话结论，再用紧凑列表列出各家公司数据，控制在 600 字以内。

问题：%s

%s""" % (q, "\n\n".join(ctx))
    answer = call_claude(prompt)
    if not answer:
        answer = "模型调用失败，先返回检索到的原文片段；请根据右侧出处自行核对。"
    citations = [{"company": h["company"], "year": h["year"], "page": h["page"],
                  "section": h["section"], "score": h["score"]} for h in hits]
    return {"answer": answer, "citations": citations,
            "retrieved": [{**{k: v for k, v in h.items() if k != "text"},
                           "text": h["text"][:800]} for h in hits]}


@app.get("/api/health")
def health():
    return {"ok": True, "chunks": len(state.get("chunks", []))}


@app.get("/", response_class=HTMLResponse)
def home():
    return open(WEB, encoding="utf-8").read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
