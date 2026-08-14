# 語意空間計量分析：線上學分課程市場之競爭與外溢效應

本 repo 為碩士論文《基於大型語言模型與語意空間計量模型探討線上學分課程市場之競爭與外溢效應》
之**完整分析程式碼**，供學術重製與方法檢驗之用。

研究對象為 ewant 育網 SOS 暑期線上學分課程（2021–2025，共 175 門），
以 SBERT 將課程文本轉為語意向量、據以建構「語意鄰近」空間權重矩陣，
估計 Panel Spatial Durbin Model（SDM）並分解直接／間接（外溢）效果。

---

## 方法流程

```
原始 Moodle 課程 panel
  └─ 01_clean_panel.py      資料前處理：面板建構、變項計算、敘述統計、VIF
       └─ 02_embed_text.py  文本清理 → SBERT 384 維語意向量
            ├─ A3_run_bloom.py   LLM 雙模型標記 Bloom 認知難度三維（Claude × GPT 互驗）
            └─ 03_build_weights.py  逐年度餘弦相似度 → kNN 列標準化語意權重矩陣 W
                 └─ 04_spatial_models.py  Moran's I、LM 檢定、OLS/SAR/SEM/SDM、效果分解
                      ├─ 05_robustness.py      穩健性檢驗（k、DV、樣本、FE、控制變項）
                      ├─ 07_mse_comparison.py  MSE / RMSE / MAE 與 5-fold 交叉驗證
                      ├─ B4_tobit_appendix.py  左設限（零修課人數）之 Tobit 對照附錄
                      └─ 06_figures.py         論文用 300dpi 圖檔
```

## 目錄結構

| 路徑 | 內容 |
|---|---|
| `scripts/config.py` | 單一設定點：路徑、欄位名、SBERT 模型、k 值、學分費常數、亂數種子 |
| `scripts/01_clean_panel.py` | **資料前處理**：建「課程–開課期別」面板、依變項與控制變項計算、敘述統計、VIF |
| `scripts/02_embed_text.py` | **SBERT 向量轉換**：文本清理 + `paraphrase-multilingual-MiniLM-L12-v2` 編碼（175×384） |
| `scripts/A1_A2_apply.py` | 補算學分數 / 學分費欄（1 學分 = 18 學習時數；每學分 NT$750） |
| `prompts/A3_bloom_prompt.md` | **LLM 提示詞規格**：Bloom 三維標記之 system / user prompt、CoT 步驟、參數與互驗設計 |
| `scripts/A3_run_bloom.py` | **LLM 提示詞腳本**：呼叫 Claude 與 GPT 雙模型標記、一致性（Kappa / ICC / MAE）與歧異仲裁 |
| `scripts/03_build_weights.py` | **空間權重矩陣**：逐年度分塊、kNN 列標準化語意 W（跨年度權重恆為 0） |
| `scripts/04_spatial_models.py` | **空間計量主模型**：Moran's I、LM 檢定、OLS/SAR/SEM/SDM、direct-indirect-total 分解 |
| `scripts/05_robustness.py` | 穩健性檢驗九項規格 |
| `scripts/07_mse_comparison.py` | 預測誤差比較（純 numpy 實作空間模型 MLE，不依賴 spreg，作為交叉驗證） |
| `scripts/B4_tobit_appendix.py` | 零修課人數左設限之 Tobit / 排除零值 SDM 三方對照 |
| `scripts/06_figures.py` | 論文圖 4-1～4-7 之 300dpi 產製 |
| `scripts/run_all.py` | 一鍵依序執行 01 → 06 |
| `replication/k4_rerun_20260806/` | **k = 4 重現紀錄**：以論文最終設定重跑表 4-3 / 4-4 / 4-6，含逐數字比對報告 |

## 環境需求

- Python 3.10+
- `pip install -r requirements.txt`
  （`torch` 為 CPU 版；若安裝失敗請用 `pip install torch --index-url https://download.pytorch.org/whl/cpu`）
- 首次執行 `02_embed_text.py` 會自動下載 SBERT 模型（約 0.5 GB，之後可離線）
- 繁體中文圖檔字型：`06_figures.py` 讀取 `fonts/NotoSansCJKtc-Regular.otf`，
  因授權與檔案大小未納入版本控制，請自 [noto-cjk](https://github.com/notofonts/noto-cjk) 下載後放入 `fonts/`

## 執行方式

```bash
pip install -r requirements.txt

# 主流程
cd scripts
python run_all.py                 # 或逐一執行 01 → 06

# LLM Bloom 標記（需金鑰，見下節）
python A3_run_bloom.py --smoke    # 先跑前 5 門測試
python A3_run_bloom.py            # 全樣本 175 門
python A3_run_bloom.py --resume   # 接續中斷的結果

# 附錄與檢驗
python 07_mse_comparison.py
python B4_tobit_appendix.py
```

k = 4 重現腳本以**當前工作目錄**讀取 `analysis_panel.csv` 與 `course_embeddings.npy`，
請將該兩檔複製到 `replication/k4_rerun_20260806/` 後再執行：

```bash
cd replication/k4_rerun_20260806
python 01_reproduce_table43_44.py
python 02_reproduce_table46.py
```

## API 金鑰

`A3_run_bloom.py` 需要 Anthropic 與 OpenAI 金鑰，皆由環境變數讀取，程式碼中不含任何金鑰。

```bash
cp .env.example .env    # 填入自己的金鑰
set -a && source .env && set +a
```

## 資料取得

本 repo **不含原始課程資料**（`analysis_panel.csv`、`course_embeddings.npy`、
權重矩陣 `.npy` 與各項 `outputs/`），因涉及 ewant 育網平台之課程後台資料使用規範。

`scripts/config.py` 之 `RAW_CSV` 指向原始 panel 檔路徑，請自行替換為自有資料，
或依 `01_clean_panel.py` 所需欄位（見 `config.py` 之欄位常數）自建同格式輸入。
如需研究資料以進行重製，請聯繫作者並依平台資料使用規範辦理。

## 可重現性

- 亂數種子固定於 `config.py`：`SEED = 20260626`
- 套件版本鎖定於 `requirements.txt`
- Moran's I 之 p 值採 999 次隨機化置換，因置換亂數不同會有 ±0.03 浮動；I 值本身為決定性計算
- `replication/k4_rerun_20260806/README.md` 記錄了論文表 4-3 / 4-4 / 4-6 之逐數字重現比對

## 已知限制

- 全樣本為暑期單一期別，開課期別固定效果實質等同年度固定效果
- 空間權重為「語意鄰近」而非地理鄰近，`W` 依年度分塊，跨年度不設連結
- 完整空間 Tobit（Xu & Lee 2015/2018）未實作；`B4` 以「排除零值 SDM」與「非空間 Tobit」兩項對照替代

## 引用

見 `CITATION.cff`。

## 授權

`LICENSE`（MIT）。若你的機構或指導教授另有規定，請逕行替換。
