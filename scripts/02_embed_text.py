# -*- coding: utf-8 -*-
"""
02 text cleaning + SBERT semantic vectorization
- merge title + summary + objectives + syllabus
- segment-level removal of SOS administrative boilerplate
- extract English course name (semantic signal)
- multilingual Sentence-BERT, L2-normalized (cosine = dot product)
- also save clean 4-field text for future LLM Bloom labeling
"""
import re, os, numpy as np, pandas as pd
import config as C

ADMIN_KW = [
    "訂單","繳費","退費","換課","退選","停課","結帳","成班","開班人數","選課人數上限",
    "報名","名額","學分費","學分費用","證書","直播","試閱","學生指南","學習平台功能教學",
    "平台操作","ewant學習簡單上手","簽到","請假","查詢校內","認抵","提醒","注意","上課時間",
    "正式開課時間","開課時間","授課語言","全線上","無面授","衝堂","報名須知","重要時程",
    "評分標準後再選課","前往繳費","繳費完畢","頁面反應","核發學分","成績登分","下載證書",
    "如何選課","如何報名","如何上課","如何繳費","必讀","額滿","錄取","作日後使用",
]
ADMIN_RE = re.compile("|".join(map(re.escape, ADMIN_KW)))
SEG_SPLIT = re.compile(r"[。！!？?\n；;｜|]|●|▶|◆|★|\*|【[^】]*】|\s{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
PAREN_TAG_RE = re.compile(r"（[^）]*(SOS|計畫|20\d{2})[^）]*）|\([^)]*(SOS|20\d{2})[^)]*\)")
EN_NAME_RE = re.compile(r"英文課名[：:]\s*([A-Za-z][A-Za-z0-9 ,\-&/'’.()]+)")

def clean_title(s):
    s = HTML_RE.sub(" ", str(s))
    s = PAREN_TAG_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

def strip_admin(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    t = HTML_RE.sub(" ", text)
    t = URL_RE.sub(" ", t)
    keep = []
    for s in SEG_SPLIT.split(t):
        if s is None:
            continue
        s = s.strip(" 　,，、.．:：-–—~～")
        if len(s) < 2:
            continue
        if ADMIN_RE.search(s):
            continue
        keep.append(s)
    return " ".join(keep)

def extract_en_name(summary):
    if not isinstance(summary, str):
        return ""
    m = EN_NAME_RE.search(summary)
    return m.group(1).strip() if m else ""

def main():
    panel = pd.read_csv(os.path.join(C.OUT, "analysis_panel.csv"))
    raw = pd.read_csv(C.RAW_CSV)[C.TEXT_COLS].copy()
    raw["src_row"] = raw.index.values
    df = panel[["panel_idx","src_row","course_id","course_name","year","term_id"]].merge(raw, on="src_row", how="left")
    df = df.sort_values("panel_idx").reset_index(drop=True)
    assert len(df) == len(panel), "merge inflated rows: %d != %d" % (len(df), len(panel))

    rows, texts = [], []
    for _, r in df.iterrows():
        title = clean_title(r["course_name_clean"] if pd.notna(r.get("course_name_clean")) else r["course_name"])
        en   = extract_en_name(r["summary_clean"])
        desc = strip_admin(r["summary_clean"])
        obj  = strip_admin(r["object_clean"])
        syll = strip_admin(r["sections_clean"])
        combined = re.sub(r"\s+", " ", " ".join([x for x in [title, en, obj, syll, desc] if x])).strip()
        texts.append(combined)
        rows.append({"panel_idx": r["panel_idx"], "course_id": r["course_id"], "course_title": title,
                     "en_name": en,
                     "raw_len": len(str(r["summary_clean"]))+len(str(r["object_clean"]))+len(str(r["sections_clean"])),
                     "clean_len": len(combined),
                     "course_description": desc, "course_objectives": obj, "course_syllabus": syll})
    rep = pd.DataFrame(rows)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(C.SBERT_MODEL)
    emb = np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False), dtype=np.float32)
    assert emb.shape[0] == len(panel), "emb rows %d != panel %d" % (emb.shape[0], len(panel))

    np.save(os.path.join(C.OUT, "course_embeddings.npy"), emb)
    rep[["panel_idx","course_id","course_title","en_name","raw_len","clean_len"]].to_csv(
        os.path.join(C.OUT, "text_clean_report.csv"), index=False, encoding="utf-8-sig")
    rep[["panel_idx","course_id","course_title","course_description","course_objectives","course_syllabus"]].to_csv(
        os.path.join(C.OUT, "llm_input_fields.csv"), index=False, encoding="utf-8-sig")
    meta = df[["panel_idx","course_id","course_name","year","term_id"]].copy()
    meta["sbert_model"] = C.SBERT_MODEL; meta["dim"] = emb.shape[1]
    meta.to_csv(os.path.join(C.OUT, "embeddings_meta.csv"), index=False, encoding="utf-8-sig")

    print("[02] embeddings:", emb.shape, "model:", C.SBERT_MODEL)
    print("text len raw->clean median: %.0f -> %.0f" % (rep["raw_len"].median(), rep["clean_len"].median()))
    print("en_name extracted:", int((rep["en_name"].fillna("").str.len()>0).sum()), "/", len(rep))
    from sklearn.metrics.pairwise import cosine_similarity
    y0 = (meta["year"]==2023).values
    sub = emb[y0]; ids = meta.loc[y0,"course_name"].tolist()
    S = cosine_similarity(sub); np.fill_diagonal(S,-1)
    i,j = np.unravel_index(np.argmax(S), S.shape)
    print("2023 most-similar pair (cos=%.3f): %s || %s" % (S[i,j], ids[i], ids[j]))

if __name__ == "__main__":
    main()
