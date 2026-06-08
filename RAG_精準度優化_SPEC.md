# RAG 精準度優化 SPEC — 從回饋系統到輕量知識圖譜

> 狀態：草案（Draft）
> 建立日期：2026-06-08
> 關聯文件：[RAG_實作紀錄.md](RAG_實作紀錄.md) 第 13 節、[rag_app.py](rag_app.py)、[eval_rag.py](eval_rag.py)、[config.py](config.py)

---

## 1. 問題定義

目前系統已具備基本 RAG 功能與離線評估框架（`eval_rag.py`），但缺少一套**能持續、系統性提升回答精準度**的機制。核心困境：

> 當回答「不準」時，無法判斷問題出在哪一個環節 —— 是檢索找錯資料、排序錯誤、生成幻覺、知識庫缺漏，還是使用者問題本身模糊？

這個問題本質上不是「RAG 架構」問題，而是**回饋系統（Feedback System）設計**問題：回饋的品質、速度、頻率，決定了系統能否真正變準，而不是單純堆疊更複雜的架構（如 GraphRAG）。

### 1.1 設計原則

- **先讓系統知道自己錯在哪，再決定怎麼修。** 不貿然引入 GraphRAG / MemGraphRAG 等複雜架構。
- **回饋必須「可歸因」**，單純的 👍/👎 訊號雜訊太大，無法導向具體修正動作。
- **資源限制下的優化順位**：地端只能用較小的 LLM（`llama3.2:3b`）與 embedding（`nomic-embed-text`），生成端（LLM 理解力）改善空間有限；應把優化重心放在**檢索端**（query rewriting、embedding 選型、chunking、reranking），這些對小模型限制的依賴較低，CP 值較高。
- **回饋機制不能犧牲使用者體驗**：「蒐集回饋」是系統優化（實驗）的目的，但對一般使用者而言是額外的負擔與干擾。兩者目的不同，**不應預設綁在一起**。因此回饋相關功能一律以「自願參與的 toggle」形式呈現 —— 預設關閉、不打擾，使用者可自行選擇是否開啟並參與（詳見第 4 節 4.4）。

---

## 2. 現況盤點

| 項目 | 現況 | 對應檔案 |
|---|---|---|
| Baseline RAG pipeline | 已具備：chunking → embedding → ChromaDB 檢索 → LLM 生成 | `rag_app.py` |
| 離線評估框架 | 已具備：hit_rate、MRR、avg_similarity、keyword_coverage、faithfulness（LLM-as-judge） | `eval_rag.py` |
| 實驗版本追蹤 | 已具備：`experiments/<version>/` + `optimization_log.md` leaderboard | `eval_rag.py` → `save_experiment()` / `update_leaderboard()` |
| 使用者回饋機制 | 已具備雛形：👍/👎 + 自由文字意見，寫入 `feedback_log.jsonl`（JSONL，含 question/answer/top_k/retrieved_sources/rating/comment） | `rag_app.py` → `save_feedback()`（[RAG_實作紀錄.md](RAG_實作紀錄.md) 第 13 節） |
| 知識庫擴充 | 已具備：手動上傳 PDF/TXT、PubMed 搜尋導入 | `rag_app.py` → `upload_file()` / `search_pubmed()` / `import_pubmed_articles()` |
| **錯誤可歸因分類** | ❌ 尚未具備 — 目前的回饋只知道「不好」，不知道「哪裡不好」 | — |
| **Evaluation Dataset（從真實使用萃取）** | ❌ 尚未具備 — 現有 `test_questions.json` 是預先設計的固定題庫 | — |
| **Feedback Memory（跨次經驗累積）** | ❌ 尚未具備 | — |
| **輕量知識圖譜 / GraphRAG** | ❌ 尚未具備，且**現階段不建議投入** | — |

---

## 3. 分階段路線圖（Phase 0 → Phase 8）

> 原則：循序漸進，每個 Phase 都要看到具體產出再進到下一階段，不可跳階。

### Phase 0：定義「準確」的標準（已部分完成）
**目標：** 把「回答準不準」轉成可量化的指標（即 loss function）。
**現況：** `eval_rag.py` 已涵蓋檢索面（hit/MRR/相似度）、內容面（關鍵字覆蓋）、生成面（忠實度）三個維度。
**待補：** 可考慮加入「是否答到問題核心」「是否漏掉重要脈絡」等更貼近使用者體感的評分構面。

### Phase 1：最小版 RAG（已完成）
資料 → chunk → embedding → vector DB → 檢索 top-k → LLM 生成 + 引用來源。
**現況：** `rag_app.py` 完整具備。

### Phase 2：觀測紀錄（雛形已完成，需強化結構）
**目標：** 每次問答都完整記錄「問題、檢索片段、分數、回答、回饋」，作為後續所有分析的原始資料。
**現況：** `feedback_log.jsonl` 已記錄 question / answer / top_k / retrieved_sources（含 source/topic/score）/ rating / comment。
**待補（見第 4 節詳細規格）：** 加入「失敗類型分類」欄位，讓紀錄從「結果」升級為「可歸因的診斷資料」。

### Phase 3：拆解錯誤類型（下一步重點）
把每次負面回饋標成以下五類之一：

| 代碼 | 類型 | 說明 | 對應修正方向 |
|---|---|---|---|
| A | Retrieval failure | 找錯資料、檢索到的內容根本不相關 | 調 chunking、換 embedding 模型、加入 query rewriting |
| B | Ranking failure | 找到對的資料，但排序錯誤（該排前面的排後面） | 加入 reranker、調整 top_k、改善相似度計算 |
| C | Generation failure | 資料正確，但 LLM 回答曲解或產生幻覺 | 改善 prompt、要求列出根據（evidence-first）、加入答案驗證 |
| D | Knowledge gap | 知識庫根本沒有相關主題的資料 | 補充資料來源（PDF/PubMed 導入）、建立缺口清單 |
| E | Query ambiguity | 使用者問題本身模糊、難以對應到知識庫內容 | 加入澄清式提問、產生多個查詢解讀 |

**實作方式：** 詳見第 4 節「回饋分類 UI 規格」。

### Phase 4：針對錯誤類型做對應修正
依照 Phase 3 累積的分類統計，優先修正「出現頻率最高」的錯誤類型，而非平均施力。

### Phase 5：建立 Evaluation Dataset
從 Phase 2-3 累積的真實回饋中，篩選出「有代表性的失敗案例」（建議 50-200 題），補上 `expected_sources` / `expected_keywords`，擴充進 `test_questions.json`，讓 `eval_rag.py` 的離線評估更貼近真實使用情境（而非僅依賴預先設計的固定題庫）。

### Phase 6：Feedback Memory（跨次經驗累積）
不只記錄「這次回饋」，而是累積成可查詢的經驗庫，例如：

- `query_pattern → preferred_sources`（這類問題該優先檢索哪些來源）
- `query_pattern → bad_chunks`（這類問題容易誤檢索到哪些片段）
- `failure_case → fix_strategy`（過去同類錯誤是怎麼修好的）

### Phase 7：輕量知識圖譜（概念關係表）
在真正導入 GraphRAG 之前，先用簡單的「概念 → 關係 → 概念」表格捕捉知識庫內的主題關聯，例如：

```
糖尿病 → 常合併 → 高血壓
Metformin → 禁忌症 → 慢性腎臟病晚期
急性心肌梗塞 → 需鑑別 → 心絞痛
```

檢索時除了找 chunk，也找相關概念與關係路徑，讓回答更有「脈絡感」。

### Phase 8：GraphRAG / MemGraphRAG（長期，現階段不建議投入）
chunk 檢索 + entity 檢索 + 關係檢索 + 記憶檢索的整合架構。**只有在 Phase 0-7 都已落實，且知識庫規模與使用量足夠大時才考慮**，否則複雜度的 ROI 過低。

---

## 4. 下一步具體規格：回饋分類 UI（Phase 2 → 3 銜接）

### 4.1 目標
把現有「👎 + 自由文字」升級成「👎 + 結構化失敗分類」，讓每筆負面回饋直接對應到第 3 節表格中的 A-E 類型，使資料具備「可直接導向修正動作」的價值。

### 4.2 UI 變更（已實作，[rag_app.py:614-672](rag_app.py#L614-L672)）

**流程修正（重要）：** 最初版本是「點擊👍/👎當下立刻寫入紀錄」，但實測發現這會導致使用者來不及選擇失敗類型、填寫意見就已送出空白內容（後續補填也不會重新送出）。已改為**兩段式流程**：
1. 點擊 👍/👎 只是「選擇評分」（存入 `selected_rating_state`，並用 `rating_display` 顯示目前選擇），👎 會額外顯示失敗類型選項
2. 使用者可從容選擇失敗類型、填寫意見
3. 按下「📨 送出回饋」按鈕才真正呼叫 `save_feedback()` 寫入紀錄（一次性帶入評分、失敗類型、意見）

**UI 元件：**
- `gr.Radio`（單選，`failure_type_radio`）：僅在選擇 👎 後顯示，選項為：
  - 🔍 檢索到的資料不相關（A）
  - 📊 資料相關，但排序怪異（B）
  - 🤖 資料正確，但 AI 回答曲解／編造（C）
  - 📭 知識庫沒有這個主題的資料（D）
  - ❓ 我的問題問得不夠清楚（E）
  - 🤷 不確定 / 其他（unknown）
- 選擇 👍 則隱藏並清空失敗類型選項（正面回饋不需要分類）
- 每次送出新問題時，自動重置 `selected_rating_state`／`rating_display`／`failure_type_radio`／`feedback_comment`／`feedback_status`，避免殘留上一輪狀態

**設計決策：單選而非多選。** 從分析角度，單選能產生乾淨的分類分布，直接反映「該優先修哪一類」；多選會造成組合爆炸（A+C、B+D+E...），難以排序優先順序。從使用者角度，選「最主要、最根本的問題」比列出所有覺得不對的地方更省力，符合「不增加使用者負擔」的原則。多重原因之間通常有因果鏈（例如「檢索錯」導致「回答曲解」），讓使用者選最根本的一個即可，其餘細節交給自由文字意見欄補充。

**操作回饋（已實作，[rag_app.py:82-86](rag_app.py#L82-L86)）：** 原本送出評分後僅顯示「感謝你的回饋，已記錄！」，使用者無法確認自己填寫的意見是否真的被記錄。已改為在確認訊息中**回顯使用者輸入的意見內容**（例如：「👎 已記錄你的評分，意見：「檢索到的內容跟問題不太相關」，感謝你的回饋！」），讓使用者能直接確認文字已送出，同時也間接提示「意見回饋是跟著評分按鈕一起送出」的操作邏輯。
- 自由文字意見欄保留，作為補充說明

### 4.3 資料結構變更（已實作，`feedback_log.jsonl`）
`save_feedback()` 在現有 entry 中新增 `failure_type` 欄位（[rag_app.py:64-83](rag_app.py#L64-L83)）：

```json
{
  "timestamp": "...",
  "question": "...",
  "answer": "...",
  "top_k": 3,
  "retrieved_sources": [...],
  "rating": "down",
  "failure_type": "A",          // A/B/C/D/E/unknown；rating="up" 時為 null
  "comment": "..."
}
```

### 4.4 參與方式：自願 Toggle（已實作）
**設計緣由：** 回饋蒐集是「實驗／系統優化」目的，與「使用者體驗」目的不同，不應強加在一般使用流程中。

**實作方式（[rag_app.py:609-631](rag_app.py#L609-L631)）：**
- 在結果區下方提供一個 `gr.Checkbox`：「🗳️ 我願意針對 AI 回答提供回饋，協助改善這個系統」，預設**未勾選**
- 整個回饋區塊（含👍/👎、意見欄、失敗分類）預設**隱藏**，僅在使用者主動勾選後才顯示（`feedback_toggle.change()` 動態切換 `feedback_group` 的 `visible`）
- 旁註明確告知：用途為系統優化研究、自願參與、不影響正常問答功能

**影響：** 蒐集到的回饋資料會是「主動參與者」的子集，樣本數可能較少。但反過來說，主動勾選參與的使用者通常更有意願認真填答；相較之下，若強制每個人都要評分，被迫參與者很可能隨意亂答（敷衍式回饋），反而會在資料中混入大量雜訊，降低整體回饋品質。因此「樣本數較少但品質較高」未必是缺點，甚至可能比「樣本數多但雜訊高」更有參考價值。後續若需要更多回饋資料，應優先考慮如何提升「已開啟回饋者」的填答意願與深度（例如更明確地說明回饋的用途與貢獻），而非擴大強制顯示的範圍。

### 4.5 後續應用：回饋資料如何真正「起作用」

**重要前提：`feedback_log.jsonl` 本身只是「紀錄」，不會自動讓系統變準。** 真正讓它發揮作用的，是「定期回頭分析這些資料、找出規律、據此調整系統」這個循環。具體分為四步：

**Step 1：定期做「失敗類型統計」（對應 Phase 3-4）**
寫一個分析腳本讀取 `feedback_log.jsonl`，統計 `rating="down"` 案例中 `failure_type`（A-E）的分布比例，藉此判斷該優先把心力放在哪裡，而非憑感覺猜：
- A（檢索不相關）佔比高 → 優先檢討 `chunk_size` / `embedding_model` / 加入 query rewriting
- D（知識庫缺漏）佔比高 → 優先補充對應主題的文件或 PubMed 文獻
- C（生成幻覺）佔比高 → 優先調整 prompt、要求「先列出根據再回答」

**Step 2：篩選代表性案例擴充評估題庫（對應 Phase 5）**
從負評案例（特別是 A/B 類、且能明確指出「應該要找到的正確來源」）整理成新測試題，補上 `expected_sources` / `expected_keywords` 後加入 `test_questions.json`，讓 `eval_rag.py` 的離線評估愈來愈貼近真實使用情境，而非僅依賴固定的預設題庫。

**Step 3：作為「修改前後比較」的依據**
每次調整 `config.py` 參數（chunk_size、top_k、embedding 模型等）後重跑 `eval_rag.py`，重點看「過去曾經失敗的真實案例」是否獲得改善——這比單看 hit_rate 數字更有說服力，因為它直接對應到「曾經讓使用者不滿意的具體問題」有沒有被解決。

**Step 4：（更後期）萃取規律建立 Feedback Memory（對應 Phase 6）**
當資料量足夠時，進一步從中找出規律，例如：「問到『XX 藥物的禁忌症』這類問題，系統經常檢索不到正確的慢性腎臟病相關文件」——這種規律可回頭指導「該往知識庫補哪些主題」或「該對哪類問題做特殊處理（如 query rewriting）」。

**待實作：** 一個讀取 `feedback_log.jsonl` 並輸出 `failure_type` 分布統計的分析腳本（Step 1 的起點），可在累積到一定資料量後優先建立。

---

## 5. 待討論 / 待決策事項

- [ ] Phase 0 的評估構面是否需要擴充（例如加入「是否漏掉重要脈絡」的人工評分）？
- [ ] `failure_type` 分類是否需要更細緻（例如把 A 拆成「完全不相關」vs「部分相關但不夠完整」）？
- [ ] Evaluation Dataset（Phase 5）擴充後，是否需要重新調整 `eval_rag.py` 的指標權重？
- [ ] Feedback Memory（Phase 6）的儲存形式 —— 簡單的查詢表（dict/JSON）即可，或需要更結構化的儲存（如 SQLite）？

---

## 6. 變更紀錄

| 日期 | 內容 |
|---|---|
| 2026-06-08 | 初版建立，整理自「MLflow/Optuna vs. 回饋機制」與「RAG 精準度優化路線圖」討論 |
| 2026-06-08 | 新增設計原則「回饋機制不能犧牲使用者體驗」，並落實「自願 Toggle」設計（4.4），預設隱藏回饋區塊，使用者需主動勾選才參與 |
