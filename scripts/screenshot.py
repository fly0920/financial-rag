# -*- coding: utf-8 -*-
"""用 Edge headless 打开带问题的问答页并截图。"""
import os
import subprocess
import sys
from urllib.parse import quote

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT = os.path.join(BASE, "screenshots", "qa_page.png")
QUESTION = sys.argv[1] if len(sys.argv) > 1 else "招商银行2023年营业收入和归属于本行股东的净利润分别是多少？"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
url = "http://127.0.0.1:8010/?q=" + quote(QUESTION)
cmd = [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
       "--user-data-dir=" + os.path.join(os.environ.get("TEMP", r"D:\temp"), "edge_qa_profile"),
       "--window-size=1440,1250", "--virtual-time-budget=120000",
       "--screenshot=" + OUT, url]
p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
print("rc", p.returncode)
print((p.stderr or "")[-500:])
print("saved", OUT, os.path.getsize(OUT) if os.path.exists(OUT) else "MISSING")
