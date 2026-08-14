# -*- coding: utf-8 -*-
"""一鍵依序執行整條 pipeline：01 -> 06。用法：python run_all.py"""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["01_clean_panel.py", "02_embed_text.py", "03_build_weights.py",
         "04_spatial_models.py", "05_robustness.py", "06_figures.py"]
for s in ORDER:
    print("\n==> 執行 %s" % s, flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, s)], check=True)
print("\n完成：outputs/ 與 figures/ 已更新。")
