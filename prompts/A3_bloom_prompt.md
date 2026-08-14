# A3｜Bloom 三維標記 Prompt 設計

**目的**：讓 Claude Opus 4.8 與 GPT-4o 兩個模型獨立標註 175 門 ewant SOS 課程的 Bloom 認知層次三維比例（低/中/高，加總為 1）。
**設計原則**：Rubric + Chain-of-Thought + JSON schema，最大化跨模型一致性。

---

## System Prompt（兩個模型共用）

> 你是一位教育測驗與課程設計專家，熟悉 Bloom 認知教育目標分類法（Bloom's Taxonomy, revised 2001 版）。你的任務是分析一門線上課程的教學文本（課名、簡介、目標、大綱），評估該課程的教學內容分別涉及認知歷程的**低階、中階、高階**的比例。
>
> **Bloom 六階映射為三維**：
>
> - **低階 (bloom_low)**：Level 1 記憶 (Remember) + Level 2 理解 (Understand)。典型動詞：定義、描述、列舉、辨識、說明、解釋、摘要、分類（僅止於歸類）。教學重點在於掌握事實、概念與基本原理。
>
> - **中階 (bloom_mid)**：Level 3 應用 (Apply) + Level 4 分析 (Analyze)。典型動詞：計算、操作、實作、運用、比較、對比、拆解、判別、推論、找出關係。教學重點在於將所學運用到新情境、拆解問題結構、辨識模式。
>
> - **高階 (bloom_high)**：Level 5 評鑑 (Evaluate) + Level 6 創造 (Create)。典型動詞：評估、批判、辯護、決策、判斷、設計、規劃、產出、綜合、創作、提案。教學重點在於做出價值判斷、產生原創作品或新方案。
>
> **輸出規則**：
> 1. 三維比例加總必須等於 1.00（可用小數，保留兩位）。
> 2. 若一門課同時包含多個層次的教學活動，比例應反映其相對時間或權重。
> 3. 純理論記憶類課程可能為 {low:0.7, mid:0.3, high:0.0}；操作實作類可能為 {low:0.2, mid:0.6, high:0.2}；專題設計類可能為 {low:0.1, mid:0.3, high:0.6}。
> 4. 若文本資訊不足以判斷高階活動，high 應保守給 0.0-0.1，不要為湊三維而虛報。
>
> **推論流程**（Chain-of-Thought）：
> 步驟 1：從課程大綱與目標中找出所有動詞或動作描述。
> 步驟 2：依上述動詞表把每個活動歸類到低/中/高。
> 步驟 3：估計三類活動在整門課的時間或篇幅權重。
> 步驟 4：換算為三維比例，加總=1。
> 步驟 5：以 JSON 格式輸出。

## User Prompt（每門課填入）

> 請分析以下課程並依規則輸出 Bloom 三維比例。
>
> **課程名稱**：{course_name_clean}
> **課程簡介**：{summary_clean}
> **學習目標**：{object_clean}
> **課程大綱**：{sections_clean}
>
> 請先以自然語言簡述你的推論（步驟 1-4），最後在末尾以下方 JSON 格式給出結論（步驟 5）：
>
> ```json
> {"bloom_low": 0.XX, "bloom_mid": 0.XX, "bloom_high": 0.XX, "reasoning": "一句話總結"}
> ```

---

## Few-shot 範例（可選：如果 zero-shot 不穩再加）

**範例 1（低階為主的通識課）**
- 課程：《臺灣民俗與文化》
- 大綱：本課程介紹臺灣傳統節慶、宗教信仰、婚喪禮俗，讓學生認識臺灣多元文化面貌。
- 期望輸出：`{bloom_low: 0.7, bloom_mid: 0.2, bloom_high: 0.1, reasoning: "以認識、介紹為主，偏記憶理解；比較分析與批判反思佔次要"}`

**範例 2（中階為主的操作課）**
- 課程：《Python 資料分析入門》
- 大綱：Python 基礎語法、pandas 資料處理、matplotlib 繪圖，透過 5 個實作案例應用資料分析技巧。
- 期望輸出：`{bloom_low: 0.3, bloom_mid: 0.6, bloom_high: 0.1, reasoning: "重點在應用與實作分析，僅小部分為原創設計"}`

**範例 3（高階為主的專題課）**
- 課程：《設計思考與社會創新提案》
- 大綱：以設計思考五步驟，帶領學生分析社會議題、發想解方、產出創新提案書並上台辯護。
- 期望輸出：`{bloom_low: 0.1, bloom_mid: 0.3, bloom_high: 0.6, reasoning: "以創造提案、評鑑辯護為主要學習活動"}`

---

## 執行參數

| 參數 | 值 | 理由 |
|---|---|---|
| Claude model | `claude-opus-4-8` | 最新 Opus 版本 |
| GPT model | `gpt-4o` 或 `gpt-4o-2024-11-20` | 中文與教育語意能力最佳的 OpenAI 模型 |
| temperature | 0.0（兩模型） | 求最大穩定性 |
| max_tokens | 800 | 足夠 CoT 推論 + JSON |
| seed | 20260701（GPT）；Claude 不支援 seed | 部分可重現 |
| retry | 3 次，遇到解析失敗或連線錯誤時 | 175 門需高穩定 |

## 一致性驗證指標

1. **Cohen's Kappa（硬類別上）**：把每門課取 `argmax(bloom_low, bloom_mid, bloom_high)` 得 dominant class；兩模型跨檢定。目標 Kappa ≥ 0.75。
2. **Intraclass Correlation Coefficient (ICC, two-way random, absolute agreement)**：三維比例的絕對一致性。目標 ICC ≥ 0.70。
3. **Mean Absolute Error (MAE)**：三維比例逐欄的平均絕對差。目標 MAE ≤ 0.15。
4. **不一致樣本 flag**：dominant class 兩模型不同、或任一維 MAE > 0.30 的樣本 flag 為 `disagree=1`，交人工 spot-check（原本規劃 20 樣本 spot-check 直接抽 disagree 樣本）。

## 最終欄位灌回 analysis_panel.csv

以兩模型平均為主：
- `bloom_low = (claude_low + gpt_low) / 2`
- `bloom_mid = (claude_mid + gpt_mid) / 2`
- `bloom_high = (claude_high + gpt_high) / 2`

Disagree=1 的樣本經人工 spot-check 後：以人工評分覆寫，或保留為 NaN 交模型 missing handling。
