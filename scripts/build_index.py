# -*- coding: utf-8 -*-
"""建 BM25 + 向量索引；只需要跑一次。"""
import json
import os
import pickle
import sys
import time

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS = os.path.join(BASE, "data", "chunks.jsonl")
INDEX = os.path.join(BASE, "data", "index")
os.makedirs(INDEX, exist_ok=True)

MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")


def tokenize(text):
    return [w for w in jieba.cut_for_search(text) if w.strip()]


def main():
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    print("chunks", len(chunks))
    started = time.time()

    print("tokenizing...")
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    print("embedding...")
    model = SentenceTransformer(MODEL, device="cpu")
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=32, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)

    with open(os.path.join(INDEX, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(os.path.join(INDEX, "embeddings.npy"), emb)
    with open(os.path.join(INDEX, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    meta = {"model": MODEL, "chunks": len(chunks), "dim": int(emb.shape[1]),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.time() - started, 1)}
    json.dump(meta, open(os.path.join(INDEX, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(meta)


if __name__ == "__main__":
    main()
