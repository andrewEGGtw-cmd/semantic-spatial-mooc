# A3｜Bloom 三維標記 Prompt 重現包（Reproducibility Package）

**目的**：讓未來研究者以完全相同之提示詞、模型參數與流程重現本研究之課程認知要求標記結果。
**原始版本**：2026-07-02（王祈翰）
**校正版本**：2026-08-15 —— 已逐字對照 `scripts/A3_run_bloom.py` 校正；差異清單見文末附錄「校正紀錄」。
**作者**：王祈翰
**對應論文附錄**：附錄 A（LLM 標記提示詞與推論參數設定）

> 本文件之提示詞與參數，均以 `scripts/A3_run_bloom.py` 之原始碼為準。凡本文件與該程式不一致者，以程式為準。

---

## 一、模型與 API 版本

| 項目 | Rater 1（Claude） | Rater 2（GPT） |
|---|---|---|
| 供應商 | Anthropic | OpenAI |
| 模型識別碼 | `claude-opus-4-8` | `gpt-4o` |
| API 端點 | Anthropic Messages API | OpenAI Chat Completions API |
| Python SDK | `anthropic` ≥ 0.30.0 | `openai` ≥ 2.0.0 |
| 標記執行日期 | 2026 年 6–7 月 | 2026 年 6–7 月 |

仲裁者（見第七節）：Claude Opus 4.7，於 Claude Cowork 介面執行，精確服務端識別字未保存。

---

## 二、System Prompt（兩模型共用，完全一致）

以下內容取自 `A3_run_bloom.py` 之 `SYSTEM_PROMPT` 常數，逐字相同（含全形／半形標點與換行）：

```
你是一位教育測驗與課程設計專家，熟悉 Bloom 認知教育目標分類法（Bloom's Taxonomy, revised 2001 版）。你的任務是分析一門線上課程的教學文本（課名、簡介、目標、大綱），評估該課程的教學內容分別涉及認知歷程的低階、中階、高階的比例。

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
步驟 5：以 JSON 格式輸出。
```

---

## 三、User Prompt Template（每門課填入四個變數）

以下內容取自 `A3_run_bloom.py` 之 `USER_TEMPLATE` 常數，逐字相同。
變數插槽為 `{course_title}`、`{course_description}`、`{course_objectives}`、`{course_syllabus}`。
（範例 JSON 中的雙大括號為 Python `str.format` 之跳脫寫法，實際送出時為單大括號。）

````
請分析以下課程並依規則輸出 Bloom 三維比例。

課程名稱：{course_title}
課程簡介：{course_description}
學習目標：{course_objectives}
課程大綱：{course_syllabus}

（注意：若「課程大綱」為「[未提供]」，請主要依課程簡介與學習目標推論。）

請先以自然語言簡述你的推論（步驟 1-4），最後在末尾以下方 JSON 格式給出結論：

```json
{"bloom_low": 0.XX, "bloom_mid": 0.XX, "bloom_high": 0.XX, "reasoning": "一句話總結"}
```
````

**變數截斷長度**（`truncate()` 函式，超長時截斷並附加 `...`）：

| 變數 | 對應欄位 | 最大字元數 |
|---|---|---|
| `course_title` | `course_title` | 200 |
| `course_description` | `course_description` | 1,500 |
| `course_objectives` | `course_objectives` | 1,500 |
| `course_syllabus` | `course_syllabus`；若為 NaN 或空字串，先替換為 `[未提供]` | 3,000 |

**實際實作片段**（`A3_run_bloom.py`）：

```python
def truncate(s, n=2000):
    s = str(s) if s is not None else ""
    return s if len(s) <= n else s[:n] + "..."

syllabus_val = row.get("course_syllabus")
if pd.isna(syllabus_val) or str(syllabus_val).strip() == "":
    syllabus_val = "[未提供]"
user_msg = USER_TEMPLATE.format(
    course_title=truncate(row.get("course_title"), 200),
    course_description=truncate(row.get("course_description"), 1500),
    course_objectives=truncate(row.get("course_objectives"), 1500),
    course_syllabus=truncate(syllabus_val, 3000),
)
```

---

## 四、推論參數（Model Parameters）

**Anthropic Claude Opus 4.8**：

```python
client.messages.create(
    model="claude-opus-4-8",
    max_tokens=800,
    temperature=0.0,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_msg}],
)
```

**OpenAI GPT-4o**：

```python
client.chat.completions.create(
    model="gpt-4o",
    temperature=0.0,
    seed=20260701,
    max_tokens=800,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ],
)
```

**參數說明**：

- `temperature = 0.0`：降低隨機性，追求最大穩定輸出。
- `seed = 20260701`：GPT-4o 額外設定，強化重現性。Claude 不支援 seed 參數。
- `max_tokens = 800`：足夠輸出簡短推論加 JSON，避免截斷。
- `top_p`、`frequency_penalty`、`presence_penalty` 均未於程式中設定，採 API 預設值（1、0、0）。
- 失敗重試：兩個模型皆設 `retries = 3`，採指數退避（`2 ** attempt` 秒）。

---

## 五、輸出解析規則

模型輸出為自然語言推論後接 JSON。解析採兩段式：先抓 ```json 圍欄區塊，抓不到再退回抓最後一個含 `bloom_low` 之 `{...}`：

````python
def parse_json_from_response(text):
    if not text:
        return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
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
            for k in ["bloom_low", "bloom_mid", "bloom_high"]:
                obj[k] = round(obj[k] / total, 2)
        obj["reasoning"] = obj.get("reasoning", "")
        return obj
    except Exception:
        return None
````

即：三維加總與 1 之差超過 ±0.05 時，按比例正規化至加總為 1。

**解析失敗處置**：解析失敗之樣本以更簡化之提示（僅要求 JSON、不含自然語言推論）重新呼叫。
最終產出之 `bloom_labels_raw.csv` 中，兩個模型 175 門之三維比例**均無缺失值**，即所有樣本最終皆解析成功。

---

## 六、跨模型一致性檢驗指標（仲裁前）

| 指標 | 目標 | 本研究實測 |
|---|---|---|
| Cohen's Kappa（dominant class） | ≥ 0.75 | 0.491（moderate，Landis & Koch, 1977） |
| MAE 低階 / 中階 / 高階 | ≤ 0.15 | 0.097 / 0.088 / 0.039 |
| 平均 MAE | ≤ 0.15 | 0.075 |
| Pearson r（三維串接） | — | 0.836 |
| Dominant class 一致率 | — | 76.0%（133 / 175） |

以上數值可由 `outputs/bloom_labels_raw.csv` 逐筆重算，並與 `outputs/bloom_agreement.json` 對照。

**Kappa 悖論說明**：Kappa 僅達中度，係主要認知層級之邊際差易翻轉所致（比例最高者僅較次高者高 ≤ 0.05 即可翻轉）；連續型指標（MAE、Pearson r）顯示兩模型對三維分布之整體判斷一致程度較高。相關討論見 Feinstein 與 Cicchetti（1990）、Cicchetti 與 Feinstein（1990）。

---

## 七、爭議觀察之仲裁協議（Post-hoc Arbitration Protocol）

### 觸發條件

以下任一條件成立之樣本，進入仲裁（由程式判定，不經人工挑選）：

1. 兩模型之 dominant class 判定不一致（各取三維中比例最高者）
2. 任一維（low / mid / high）之模型間絕對差 > 0.30

判定程式為 `A3_run_bloom.py` 之 `compute_agreement()`。本研究 175 門中，45 門符合觸發條件，其 `panel_idx` 清單見 `outputs/bloom_agreement.json` 之 `disagree_panel_idx`。

### 執行環境

仲裁於 **Claude Cowork 介面**以 **Claude Opus 4.7** 執行，逐筆進行，未以 API 腳本化。
因此**仲裁環節不存在固定的提示詞模板**：仲裁指令為對話式，且模型可直接讀取課程文本與兩個初始模型之輸出檔。
本節之協議即為該環節之規格說明，可供查核者為觸發規則（程式碼）、判準（同第二節）、以及逐筆裁決結果與理由（`bloom_arbitration.csv`）。

### 仲裁流程

1. 仲裁時同時提供：
   - 該課程之原始文字（課名、簡介、目標、大綱）
   - Claude Opus 4.8 之三維比例與 reasoning
   - GPT-4o 之三維比例與 reasoning
2. 依課程文字重新判讀，做出以下四種決定之一：
   - 採 Claude 標記
   - 採 GPT 標記
   - 採兩者之折中平均
   - 另行擬定共識值（若兩者皆不夠準確）
3. 判準與初始標記完全相同（即第二節之三類聚合準則）；輸出為三維比例（加總為 1，保留兩位小數）與一句理由，記錄於 `bloom_arbitration.csv` 之 `arb_note` 欄。

### 實際裁決分布（45 門）

| 裁決型態 | 門數 |
|---|---|
| 折中平均 | 32 |
| 折中而偏向 GPT-4o | 6 |
| 保留 Claude Opus 4.8 值 | 5 |
| 折中而偏向 Claude Opus 4.8 | 1 |
| 逕採 GPT-4o 值 | 1 |
| **合計** | **45** |

45 門之三維比例加總均為 1.00。

### 仲裁後聚合規則

- 通過仲裁之 45 門：兩模型原值改為仲裁共識值（統一雙 rater 值）。
- 未觸發仲裁之 130 門：保留兩模型原值。
- 最終每門課之 Bloom 三維 = (Claude 值 + GPT 值) / 2。

上述規則已逐筆驗證：`analysis_panel.csv` 之 45 門仲裁樣本三欄與 `bloom_arbitration.csv` 完全相同；其餘 130 門與 `bloom_labels_raw.csv` 之雙模型算術平均完全相同。

### 仲裁後一致性

| 指標 | 值 |
|---|---|
| 平均 MAE | 0.042 |
| Dominant class 一致率 | 100% |

註：仲裁後之一致性為定義上的結果（爭議樣本雙 rater 值已被統一），**不構成獨立的效度證據**，亦非人類專家驗證。

---

## 八、完整重現流程（給未來研究者）

### 環境需求

```bash
pip install anthropic openai pandas scikit-learn numpy
```

### 環境變數

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-proj-..."
```

### 輸入資料格式

`llm_input_fields.csv` 需含以下欄位：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `panel_idx` | int | 課程於 panel 中之編號 |
| `course_id` | int | 課程平台端 ID |
| `course_title` | str | 課名（已清 HTML）|
| `course_description` | str | 課程簡介（已清 HTML）|
| `course_objectives` | str | 學習目標（已清 HTML）|
| `course_syllabus` | str \| NaN | 課程大綱（已清 HTML；可為缺失）|

### 執行順序

1. `python A3_run_bloom.py --smoke` — 5 門測試
2. `python A3_run_bloom.py` — 全 175 門標記（約 30–45 分鐘、成本約 US$7）
3. 讀 `bloom_agreement.json` 檢查初步一致性
4. 依 `disagree_panel_idx` 清單逐門仲裁，寫入 `bloom_arbitration.csv`
5. 重算一致性、灌回 `analysis_panel.csv` 之 `bloom_low` / `bloom_mid` / `bloom_high` 三欄

`--resume` 可略過已完成之 `panel_idx`；每 10 筆自動存檔一次。

### 輸出資料

| 檔案 | 內容 |
|---|---|
| `bloom_labels_raw.csv` | 兩模型各 175 門之三維比例 + reasoning |
| `bloom_arbitration.csv` | 45 門仲裁結果與理由 |
| `bloom_agreement.json` | 仲裁前一致性指標 |
| `bloom_agreement_post.json` | 仲裁後一致性指標 |
| `bloom_labels_final.csv` | 最終雙 rater 值（仲裁樣本已統一）|
| `analysis_panel.csv` | 主分析面板，含 `bloom_low`、`bloom_mid`、`bloom_high` 三欄 |

---

## 九、已知限制與未來擴充建議

1. **`temperature = 0.0` 不保證逐位元重現**：API 供應商之後端路由、模型版本更新、量化精度皆可能微幅影響輸出。
2. **仲裁環節之重現性低於初始標記環節**：初始標記有逐字提示詞與腳本，仲裁則於 Cowork 介面逐筆對話執行，提示詞原文與逐筆回應未保存，僅留觸發規則與裁決結果。
3. **中文專有名詞邊界**：兩模型對台灣特定領域術語（如「磨課師」、「跨校通識」）之判別未經專門校準，可能導致個別課程之認知層級判定偏移。
4. **提示詞敏感性**：本研究提示詞未做正式的 ablation 測試（如 few-shot、CoT on/off），未來可對此展開系統性研究。
5. **提示詞第 3 條之單向保守指令**：輸出規則第 3 條僅對 `bloom_high` 設下保守指示，未對低階、中階設對稱規則，可能使高階比例的估計偏低，進而影響以高階比例為核心自變項之結論。
6. **模型更迭**：Anthropic 與 OpenAI 定期釋出新版本。未來重現時建議明確記錄 API 回傳之 `model` 欄位。
7. **人類專家對照**：本研究未進行人類專家標記之對照試驗。未來可招募 3 位以上教育專家對子樣本（30–50 門）三方標記，計算 LLM ↔ 人類 Kappa 與 MAE。

---

## 十、引用建議

若使用本重現包，請引用：

> 王祈翰（2026）。**基於大型語言模型與語意空間計量模型探討線上學分課程生態之語意鄰近關聯與外溢效果**。碩士論文，國立陽明交通大學教育研究所。

以及方法論支持文獻：

- Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A taxonomy for learning, teaching, and assessing: A revision of Bloom's taxonomy of educational objectives* (Complete ed.). Longman.
- Cicchetti, D. V., & Feinstein, A. R. (1990). High agreement but low kappa: II. Resolving the paradoxes. *Journal of Clinical Epidemiology*, 43(6), 551–558.
- Feinstein, A. R., & Cicchetti, D. V. (1990). High agreement but low kappa: I. The problems of two paradoxes. *Journal of Clinical Epidemiology*, 43(6), 543–549.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174.

---

## 十一、聯絡窗口

- 研究者：王祈翰
- Email：nycu20260422@gmail.com
- 資料集開放請求：附上使用計畫，經同意後提供 `outputs/` 全部原始檔

---

## 附錄　校正紀錄（2026-08-15）

本文件之 2026-07-02 版與 `scripts/A3_run_bloom.py` 有下列七處不符，已依程式碼校正。原版數值一併列出以供追溯。

| # | 項目 | 2026-07-02 版 | 實際程式（本版採用） |
|---|---|---|---|
| 1 | `max_tokens` | 500 | **800**（兩模型皆是；與論文表 A-1 一致） |
| 2 | `course_description` 截斷 | 1,200 字元 | **1,500** |
| 3 | `course_objectives` 截斷 | 1,200 字元 | **1,500** |
| 4 | `course_syllabus` 截斷 | 2,500 字元 | **3,000** |
| 5 | System Prompt 之「推論流程」 | 壓縮為一行「先找動詞→歸類→估權重→算比例→輸出 JSON」 | **五個編號步驟**（已改為逐字版） |
| 6 | User Prompt 措辭與變數名 | `{title}`／`{desc}`／`{obj}`／`{syl}`，且少「依規則」「（步驟 1-4）」「注意：」等字 | **`{course_title}` 等四個變數名與逐字文本** |
| 7 | 解析函式 | 僅有「抓最後一個含 bloom_low 之區塊」一支 | **先抓 ```json 圍欄，再退回上述分支** |

另有兩處非程式面之更新：

- 第十節「引用建議」原載舊論文題目（「…線上學分課程**市場之競爭與外溢效應**」），已更新為現行題目（「…線上學分課程**生態之語意鄰近關聯與外溢效果**」）。
- 第五節原載「僅 GPT 於 175 門中出現 2 筆解析失敗」。`bloom_labels_raw.csv` 之兩模型三維比例均無缺失值，與「失敗後重試成功」相容，但無法由現存輸出檔獨立佐證失敗筆數，故改為僅陳述可查證之事實。

校正方式：第二、三節之提示詞係以正規式自 `A3_run_bloom.py` 之 `SYSTEM_PROMPT` 與 `USER_TEMPLATE` 常數直接抽出，非人工重打。
