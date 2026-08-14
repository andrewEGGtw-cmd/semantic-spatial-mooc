# -*- coding: utf-8 -*-
"""
A3｜Bloom 三維標記（Claude Opus 4.8 + GPT-4o 互驗）

前置：
  pip install --break-system-packages anthropic openai
  export ANTHROPIC_API_KEY="sk-ant-..."
  export OPENAI_API_KEY="sk-..."

執行：
  cd analysis_pipeline/scripts
  python3 A3_run_bloom.py          # 全部 175 門
  python3 A3_run_bloom.py --smoke  # 只跑前 5 門測試
  python3 A3_run_bloom.py --resume # 接續已跑到一半的結果

產出：
  outputs/bloom_labels_raw.csv     # 每列 = 一門課，claude_*/gpt_* 三維 + reasoning
  outputs/bloom_agreement.json     # Kappa / ICC / MAE / disagree 樣本
  analysis_panel.csv (in-place)    # 灌回 bloom_low/mid/high（兩模型平均）
"""
import os, sys, json, time, argparse, re
import pandas as pd
import numpy as np

# ---- system prompt（見 A3_bloom_prompt.md）----
SYSTEM_PROMPT = """你是一位教育測驗與課程設計專家，熟悉 Bloom 認知教育目標分類法（Bloom's Taxonomy, revised 2001 版）。你的任務是分析一門線上課程的教學文本（課名、簡介、目標、大綱），評估該課程的教學內容分別涉及認知歷程的低階、中階、高階的比例。

Bloom 六階映射為三維：
- 低階 (bloom_low)：Level 1 記憶 + Level 2 理解。典型動詞：定義、描述、列舉、辨識、說明、解釋、摘要。教學重點在掌握事實、概念與基本原理。
- 中階 (bloom_mid)：Level 3 應用 + Level 4 分析。典型動詞：計算、操作、實作、運用、比較、對比、拆解、判別、推論。教學重點在運用所學到新情境、拆解問題結構。
- 高階 (bloom_high)：Level 5 評鑑 + Level 6 創造。典型動詞：評估、批判、辯護、決策、判斷、設計、規劃、產出、綜合、創作、提案。教學重點在做出價值判斷、產生原創作品或新方案。

輸出規則：
1. 三維比例加總必須等於 1.00（保留兩位小數）。
2. 若一門課同時包含多層次教學活動，比例應反映其相對時間或權重。
3. 若文本資訊不足以判斷高階活動，high 應保守給 0.0-0.1，不要為湊三維而虛報。

推論流程（Chain-of-Thought）：
步驟 1：從課程大綱與目標中找出所有動詞或動作描述。
步驟 2：依動詞表把每個活動歸類到低/中/高。
步驟 3：估計三類活動在整門課的時間或篇幅權重。
步驟 4：換算為三維比例，加總=1。
步驟 5：以 JSON 格式輸出。"""

USER_TEMPLATE = """請分析以下課程並依規則輸出 Bloom 三維比例。

課程名稱：{course_title}
課程簡介：{course_description}
學習目標：{course_objectives}
課程大綱：{course_syllabus}

（注意：若「課程大綱」為「[未提供]」，請主要依課程簡介與學習目標推論。）

請先以自然語言簡述你的推論（步驟 1-4），最後在末尾以下方 JSON 格式給出結論：

```json
{{"bloom_low": 0.XX, "bloom_mid": 0.XX, "bloom_high": 0.XX, "reasoning": "一句話總結"}}
```"""


def truncate(s, n=2000):
    s = str(s) if s is not None else ""
    return s if len(s) <= n else s[:n] + "..."


def parse_json_from_response(text):
    """從模型回覆中抓出 JSON block，回傳 dict 或 None。"""
    if not text:
        return None
    # 找 ```json ... ``` block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        # 找最後一個 { ... } block
        matches = re.findall(r"\{[^{}]*bloom_low[^{}]*\}", text, re.DOTALL)
        if not matches:
            return None
        candidate = matches[-1]
    try:
        obj = json.loads(candidate)
        for k in ["bloom_low", "bloom_mid", "bloom_high"]:
            if k not in obj:
                return None
            obj[k] = float(obj[k])
        total = obj["bloom_low"] + obj["bloom_mid"] + obj["bloom_high"]
        if abs(total - 1.0) > 0.05:
            # 微調至加總=1
            for k in ["bloom_low", "bloom_mid", "bloom_high"]:
                obj[k] = round(obj[k] / total, 2)
        obj["reasoning"] = obj.get("reasoning", "")
        return obj
    except Exception:
        return None


def call_claude(client, user_msg, retries=3):
    """呼叫 Claude Opus 4.8"""
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=800,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = r.content[0].text if r.content else ""
            return text
        except Exception as e:
            if attempt == retries - 1:
                return f"__ERROR__: {e}"
            time.sleep(2 ** attempt)


def call_gpt(client, user_msg, retries=3):
    """呼叫 GPT-4o"""
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                seed=20260701,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                return f"__ERROR__: {e}"
            time.sleep(2 ** attempt)


def compute_agreement(df):
    """計算 Cohen's Kappa（dominant class）、ICC、MAE"""
    from sklearn.metrics import cohen_kappa_score
    ok = df.dropna(subset=["claude_low", "gpt_low"]).copy()
    dims = ["low", "mid", "high"]
    for src in ["claude", "gpt"]:
        cols = [f"{src}_{d}" for d in dims]
        ok[f"{src}_dominant"] = ok[cols].idxmax(axis=1).str.replace(f"{src}_", "", regex=False)
    kappa = float(cohen_kappa_score(ok["claude_dominant"], ok["gpt_dominant"]))
    maes = {d: float(np.mean(np.abs(ok[f"claude_{d}"] - ok[f"gpt_{d}"]))) for d in dims}
    icc_num = np.var((ok[[f"claude_{d}" for d in dims]].values + ok[[f"gpt_{d}" for d in dims]].values) / 2)
    icc_den = np.var(ok[[f"claude_{d}" for d in dims]].values) + np.var(ok[[f"gpt_{d}" for d in dims]].values)
    icc = float(icc_num / (icc_den + 1e-9))
    diff_arr = np.abs(ok[[f"claude_{d}" for d in dims]].values -
                       ok[[f"gpt_{d}" for d in dims]].values).max(axis=1)
    disagree_mask = (ok["claude_dominant"] != ok["gpt_dominant"]) | (diff_arr > 0.30)
    disagree_ids = ok.loc[disagree_mask, "panel_idx"].tolist()
    return {
        "n_labeled": int(len(ok)),
        "cohen_kappa_dominant": round(kappa, 3),
        "mae_low": round(maes["low"], 3),
        "mae_mid": round(maes["mid"], 3),
        "mae_high": round(maes["high"], 3),
        "icc_approx": round(icc, 3),
        "n_disagree": len(disagree_ids),
        "disagree_panel_idx": disagree_ids,
        "kappa_target": 0.75,
        "kappa_passed": kappa >= 0.75,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="only run first 5 rows")
    ap.add_argument("--resume", action="store_true", help="skip rows already labeled")
    ap.add_argument("--limit", type=int, default=0, help="cap number of rows")
    args = ap.parse_args()

    import config as C
    from anthropic import Anthropic
    from openai import OpenAI

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY environment variable")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Missing OPENAI_API_KEY environment variable")

    claude = Anthropic()
    gpt = OpenAI()

    llm_in = pd.read_csv(os.path.join(C.OUT, "llm_input_fields.csv"))
    print(f"[A3] input rows: {len(llm_in)}")

    raw_path = os.path.join(C.OUT, "bloom_labels_raw.csv")
    if args.resume and os.path.exists(raw_path):
        done = pd.read_csv(raw_path)
        done_ids = set(done["panel_idx"].tolist())
        print(f"[A3] resume: skipping {len(done_ids)} rows")
    else:
        done = pd.DataFrame()
        done_ids = set()

    if args.smoke:
        llm_in = llm_in.head(5)
    if args.limit:
        llm_in = llm_in.head(args.limit)

    records = done.to_dict("records") if len(done) else []
    for i, row in llm_in.iterrows():
        if int(row["panel_idx"]) in done_ids:
            continue
        syllabus_val = row.get("course_syllabus")
        if pd.isna(syllabus_val) or str(syllabus_val).strip() == "":
            syllabus_val = "[未提供]"
        user_msg = USER_TEMPLATE.format(
            course_title=truncate(row.get("course_title"), 200),
            course_description=truncate(row.get("course_description"), 1500),
            course_objectives=truncate(row.get("course_objectives"), 1500),
            course_syllabus=truncate(syllabus_val, 3000),
        )
        print(f"[{i+1}/{len(llm_in)}] panel_idx={row['panel_idx']} name={str(row.get('course_title',''))[:30]}...")

        c_text = call_claude(claude, user_msg)
        g_text = call_gpt(gpt, user_msg)
        c_obj = parse_json_from_response(c_text)
        g_obj = parse_json_from_response(g_text)

        rec = {
            "panel_idx": int(row["panel_idx"]),
            "course_id": int(row["course_id"]),
            "course_name": row.get("course_title"),
            "claude_low": c_obj["bloom_low"] if c_obj else None,
            "claude_mid": c_obj["bloom_mid"] if c_obj else None,
            "claude_high": c_obj["bloom_high"] if c_obj else None,
            "claude_reasoning": c_obj["reasoning"] if c_obj else c_text[:500],
            "gpt_low": g_obj["bloom_low"] if g_obj else None,
            "gpt_mid": g_obj["bloom_mid"] if g_obj else None,
            "gpt_high": g_obj["bloom_high"] if g_obj else None,
            "gpt_reasoning": g_obj["reasoning"] if g_obj else g_text[:500],
        }
        records.append(rec)

        if (i + 1) % 10 == 0 or (i + 1) == len(llm_in):
            pd.DataFrame(records).to_csv(raw_path, index=False, encoding="utf-8-sig")
            print(f"  ...saved progress ({len(records)} rows)")

        time.sleep(0.5)

    df = pd.DataFrame(records)
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"[A3] saved raw labels to {raw_path}")

    ok = df.dropna(subset=["claude_low", "gpt_low"])
    if len(ok) < 10:
        print("[A3] not enough labeled rows to compute agreement (<10).")
        return

    agreement = compute_agreement(df)
    with open(os.path.join(C.OUT, "bloom_agreement.json"), "w", encoding="utf-8") as f:
        json.dump(agreement, f, ensure_ascii=False, indent=2)
    print("[A3] agreement:")
    print(json.dumps(agreement, ensure_ascii=False, indent=2))

    if not args.smoke:
        panel_path = os.path.join(C.OUT, "analysis_panel.csv")
        panel = pd.read_csv(panel_path)
        avg = df.copy()
        for d in ["low", "mid", "high"]:
            avg[f"avg_{d}"] = (avg[f"claude_{d}"] + avg[f"gpt_{d}"]) / 2
        panel = panel.drop(columns=["bloom_low", "bloom_mid", "bloom_high"], errors="ignore")
        panel = panel.merge(
            avg[["panel_idx", "avg_low", "avg_mid", "avg_high"]].rename(
                columns={"avg_low": "bloom_low", "avg_mid": "bloom_mid", "avg_high": "bloom_high"}
            ),
            on="panel_idx",
            how="left",
        )
        panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
        print(f"[A3] merged bloom_low/mid/high into {panel_path}")


if __name__ == "__main__":
    main()
