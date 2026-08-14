# -*- coding: utf-8 -*-
"""
共用設定檔 (config)
碩論：基於 LLM 與語意空間計量模型探討線上學分課程市場之競爭與外溢效應
資料：ewant SOS 暑期線上學分課程 panel (2021-2025)，目前 175 門。

設計原則：結構可擴充。
- TERM_KEYS 目前只有各年度 summer；日後加入春/秋班，只要原始資料含對應 course_term，
  下游（W 矩陣、固定效果）會自動依 term_id 分組，無需改碼。
- 認知難度 (Bloom) 與 學分數/學分費 先保留為空白佔位欄，待 LLM 標記 / 向 ewant 申請後填入。
"""
import os

# ---- 路徑 ----
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # analysis_pipeline/
ROOT = os.path.dirname(BASE)                                          # 碩論數據整理/
RAW_CSV = os.path.join(ROOT, "0623原始檔", "committee_moodle_course_panel.csv")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ---- 分析單位與時間 ----
# 開課期別 = (年度, 期別)。目前資料全為 summer，故 term_id 實質等於年度。
# 日後若有春/秋班，term_id 會自然展開為 2021_spring / 2021_summer / 2021_fall ...
ID_COL      = "course_id"
YEAR_COL    = "course_year"
TERM_COL    = "course_term"     # spring / summer / fall

# ---- 依變項 ----
ENROLL_RAW  = "student_count_for_model"   # 修課人數（DV 來源）

# ---- 目前可用的控制變項（論文表3-1 中已存在於後台者）----
VIDEO_COL   = "video_count_with_url_proxy"  # 影片數
MAT_COL     = "resource_module_count"       # 教材數

# ---- 額外可用課程屬性（延伸/穩健用，主模型不一定納入）----
EXTRA_COVARS = {
    "hours":       "course_hours_raw",        # 學習時數
    "instructors": "instructor_count_lead_only",  # 主授教師數
    "quiz":        "quiz_module_count",       # 測驗數
}

# ---- 學分數 / 學分費 推算常數 ----
# credit_est = hours / CREDIT_HOURS_PER  依教育部 MOOCs 規範 1 學分 = 18 學習時數
# credit_fee = credit_est * CREDIT_FEE_TWD
#   2021–2025 SOS 學分費：每學分 NT$750（2022、2025 SOS 學生指南手冊確認；
#   2021、2023、2024 已去信 ewant 課規師確認中，暫沿用 750。）
#   2026 起調為 NT$900，本樣本不涵蓋，故本常數固定 750。
CREDIT_HOURS_PER = 18
CREDIT_FEE_TWD   = 750

# ---- 保留佔位（Bloom 待 LLM 標記，A3）----
RESERVED_EMPTY = ["bloom_low", "bloom_mid", "bloom_high"]

# ---- 分群/固定效果用 ----
FIELD_COL   = "field_category"          # 學門
SCHOOL_COL  = "school_fixed_effect_id"  # 學校固定效果 id
SCHOOL_NAME = "school_name"
QUALITY_COL = "data_quality_flag"       # good / bad_no_grade / bad_low_n / bad_extreme

# ---- 文本欄（SBERT / 未來 LLM 輸入）----
TEXT_COLS = ["course_name_clean", "summary_clean", "object_clean", "sections_clean"]

# ---- SBERT 設定 ----
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 多語、含中文，384 維

# ---- 權重矩陣設定 ----
KNN_LIST = [5, 10]      # 主模型 k=10，穩健性檢驗 k=5
KNN_MAIN = 10

# ---- 隨機種子 ----
SEED = 20260626
