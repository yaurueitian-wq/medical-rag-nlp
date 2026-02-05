# RAG 實作練習紀錄 — 醫療知識問答系統

> 課程：北醫 醫療 AI 實戰力養成班：從數據到臨床的智慧應用
> 日期：2026-02-05

---

## 1. 專案概述

| 項目 | 說明 |
|------|------|
| **目標** | 實作基於 RAG (Retrieval-Augmented Generation) 的醫療知識問答系統 |
| **技術架構** | Ollama (本地 LLM) + ChromaDB (向量資料庫) + LangChain (RAG 框架) |
| **LLM 模型** | llama3.2:3b (生成回答) + nomic-embed-text (向量嵌入) |
| **Python** | 3.13.5 |

---

## 2. 討論過程與決策

在開始實作前，我們針對三個關鍵技術選擇進行了討論：

| 議題 | 選擇 | 理由 |
|------|------|------|
| LLM 選擇 | 本地模型 Ollama | 不需 API key、隱私性高、適合學習、離線可用 |
| 資料來源 | 示範用假資料 | 方便快速學習 RAG 流程，不需額外準備資料 |
| 向量資料庫 | ChromaDB | 輕量級、易上手、純 Python 安裝、適合原型開發 |

---

## 3. 環境建置過程

### 3.1 安裝 Ollama

使用 Homebrew 安裝並啟動服務：

```bash
brew install ollama
brew services start ollama
```

### 3.2 下載模型

- **LLM 模型：** `ollama pull llama3.2:3b`（2GB，用於生成回答）
- **Embedding 模型：** `ollama pull nomic-embed-text`（274MB，用於文字轉向量）

### 3.3 安裝 Python 套件

```bash
pip3 install chromadb langchain langchain-ollama langchain-chroma langchain-text-splitters
```

---

## 4. RAG 架構說明

### 4.1 RAG 是什麼？

**RAG = Retrieval-Augmented Generation（檢索增強生成）**

核心理念：不是讓 LLM 靠記憶回答，而是先從知識庫中「檢索」相關資料，再讓 LLM 根據這些資料「生成」回答。

好處：
- 減少幻覺 (Hallucination)，回答有據可查
- 可使用最新或專業的知識，不受模型訓練資料截止日限制
- 知識庫可隨時更新，不需重新訓練模型

### 4.2 RAG 流程圖

```
使用者問題 → Embedding 模型(nomic-embed-text) → 向量化 → ChromaDB 相似度搜尋
  → 取出 Top-3 相關片段 → 組合成 Prompt → Ollama LLM(llama3.2:3b) → 生成回答
```

---

## 5. 實作步驟詳解

### Step 1：匯入套件

使用 LangChain 生態系的套件：OllamaEmbeddings, ChatOllama, Chroma, RecursiveCharacterTextSplitter 等。

### Step 2：準備示範醫療資料

準備了 **6 份醫療文件**，涵蓋多種臨床主題：

| # | 文件主題 | 來源 | 內容摘要 |
|---|---------|------|---------|
| 1 | 糖尿病概論 | 內科學教科書 | 分類、診斷標準 |
| 2 | 糖尿病藥物治療 | 臨床藥理學 | 階梯式治療方針 |
| 3 | 高血壓分級與非藥物治療 | 心臟內科學 | ACC/AHA 指引 |
| 4 | 高血壓藥物治療 | 臨床藥理學 | ACEI, ARB, CCB 等 |
| 5 | 急性心肌梗塞 | 急診醫學 | 症狀、診斷、MONA |
| 6 | 慢性腎臟病 | 腎臟內科學 | CKD 分期與治療 |

### Step 3：文本分割 (Text Splitting)

- **工具：** RecursiveCharacterTextSplitter
- **參數：** chunk_size=300, chunk_overlap=50
- **分割優先順序：** 段落 → 換行 → 句號 → 逗號 → 空格
- **為什麼要分割：** 提高檢索精確度、符合 LLM context window 限制

### Step 4：建立向量資料庫 (Embedding + ChromaDB)

- **Embedding 模型：** nomic-embed-text（透過 Ollama 本地運行）
- **向量資料庫：** ChromaDB，儲存路徑 `./chroma_db`
- **過程：** 文字 → Embedding → 向量 → 存入 ChromaDB 建立索引

### Step 5：建立檢索器 (Retriever)

- **搜尋方式：** cosine similarity（餘弦相似度）
- **返回結果：** Top-3 最相關的文本片段

### Step 6：組裝 RAG Pipeline

使用 LangChain Expression Language (LCEL) 串接各元件：

```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

各元件功能：
- **Retriever**：檢索相關文件片段
- **format_docs**：將文件格式化為字串
- **ChatPromptTemplate**：組合 Prompt（參考資料 + 問題）
- **ChatOllama**：LLM 生成回答
- **StrOutputParser**：解析輸出為字串

---

## 6. 測試結果

### 測試問題

> 「高血壓合併糖尿病的患者應該首選什麼降壓藥？」

### 系統回答

> 「根據提供的參考資料，高血壓合併糖尿病的患者應該首選 ACE 抑制劑（如 Enalapril）或 ARB（如 Losartan），因為這些藥物具有腎臟保護作用，並且不會引起乾咳。」

### 結果分析

回答**正確**且完全基於檢索到的文件內容，沒有產生幻覺 (hallucination)。這證明 RAG 架構能有效地讓 LLM「有據可查」地回答問題。

---

## 7. 關鍵概念整理

| 概念 | 說明 |
|------|------|
| **RAG** | Retrieval-Augmented Generation，結合檢索與生成的 AI 架構 |
| **Embedding** | 將文字轉換為高維向量（數字陣列），語義相近的文字向量距離較近 |
| **向量資料庫** | 儲存和檢索向量的專用資料庫（我們使用 ChromaDB） |
| **Text Splitting** | 將長文檔切成小片段，提高檢索精準度 |
| **Retriever** | 根據問題在向量資料庫中搜尋最相關的文本片段 |
| **Prompt Template** | 定義如何將檢索結果與問題組合成 LLM 的輸入 |
| **LCEL** | LangChain Expression Language，用管道符號 (\|) 串接各元件 |
| **Hallucination** | LLM 產生不正確或虛構資訊的現象，RAG 可有效減少此問題 |
| **Cosine Similarity** | 衡量兩個向量方向相似程度的指標，用於檢索最相關的文件 |

---

## 8. 程式碼位置與執行方式

- **主程式：** `NLP/test_RAG.py`
- **向量資料庫：** `NLP/chroma_db/`

```bash
cd NLP
python3 test_RAG.py
```

程式會自動執行 Step 2-7 的測試，最後進入互動模式讓你自由問問題。輸入 `quit` 結束。

---

## 10. 進階功能：Web UI + 關聯度分數（2026-02-05 新增）

### 10.1 新增功能

在基本 RAG 完成後，我們進一步實作了：

1. **Gradio Web 前端介面** — 瀏覽器中操作的互動式問答介面
2. **關聯度分數顯示** — 每個檢索片段都附帶相似度百分比和關聯等級

### 10.2 關聯度分數原理

ChromaDB 提供 `similarity_search_with_relevance_scores()` 方法，回傳 `(document, score)` 配對：

- **score 範圍：** 0~1（越高越相關）
- 🟢 **> 85%**：高度相關
- 🟡 **70-85%**：中度相關
- 🔴 **< 70%**：低度相關

```python
results_with_scores = vectorstore.similarity_search_with_relevance_scores(
    query=question,
    k=top_k,  # 可透過滑桿調整
)
```

### 10.3 Gradio Web UI

- **安裝：** `pip3 install gradio`
- **程式：** `NLP/rag_app.py`
- **啟動：** `python3 rag_app.py`
- **網址：** http://localhost:7860

UI 功能：
- 文字輸入框 + 範例問題一鍵填入
- Top-K 滑桿（1~5）調整檢索片段數量
- 左側顯示 AI 回答，右側顯示檢索依據與分數

### 10.4 UI 優化紀錄

#### 字體優化
- 設定字體優先順序：PingFang TC → Microsoft JhengHei → Noto Sans TC → Helvetica Neue → Arial
- 程式碼區塊使用等寬字體：SF Mono → Menlo → Consolas → Courier New

#### 佈局優化
- **提問區**：用圓角邊框包裹輸入框、滑桿、範例問題，加上 `📝 提問區` 標題
- **AI 回答區**：獨立區塊，淺灰背景，`💡 AI 回答` 標題
- **檢索結果區**：獨立區塊，淺灰背景，`📚 檢索結果與關聯度` 標題
- 區塊間距 16px、內距 20px，標題使用藍色粗體搭配底線分隔

#### Gradio 6.0 相容性
- 將 `theme` 和 `css` 參數從 `gr.Blocks()` 移至 `app.launch()`，消除 deprecation 警告

### 10.5 知識庫管理功能（2026-02-05 新增）

為提高系統可信度，新增了知識庫管理功能：

#### 知識庫來源清單
- 以表格顯示所有知識庫來源：來源名稱、主題、片段數、類型、加入時間
- 預設顯示 6 份內建醫療文件的來源資訊
- 每次上傳新文件後自動更新清單

#### 檔案上傳擴充知識庫
- 支援 **PDF** 和 **TXT** 格式的檔案上傳
- 使用 `PyPDFLoader`（來自 langchain-community）處理 PDF
- 上傳時可自訂來源名稱與主題名稱（選填，不填則自動使用檔名）
- 上傳後自動：文本分割 → Embedding → 加入 ChromaDB → 更新來源清單
- UI 使用可收合的 Accordion 面板，不影響主要問答區域

#### 新增安裝的套件
- `pip3 install pypdf langchain-community`

### 10.6 討論決策紀錄

- **為什麼用 Gradio？** 輕量、幾行程式碼就能建立 Web UI，適合快速原型開發
- **為什麼顯示關聯度？** 讓使用者能判斷 AI 回答是否基於高品質的檢索結果，增加可信度
- **為什麼顯示知識庫清單？** 讓使用者清楚知道 AI 的回答是基於哪些來源，提高透明度與可信度
- **為什麼用 Accordion？** 知識庫管理是進階功能，預設收合不干擾主要問答流程

### 10.7 參數設定檔 config.py（2026-02-05 新增）

將所有可調參數從 `rag_app.py` 抽出至獨立的 `config.py`，方便實驗不同參數組合：

| 參數 | 設定變數 | 目前值 | 說明 |
|------|----------|--------|------|
| LLM 模型 | `LLM_MODEL` | llama3.2:3b | Ollama 模型名稱 |
| Temperature | `LLM_TEMPERATURE` | 0.3 | 生成溫度（0=保守, 1=創意） |
| Embedding 模型 | `EMBEDDING_MODEL` | nomic-embed-text | 文字轉向量模型 |
| 片段大小 | `CHUNK_SIZE` | 300 | 每個文本片段最大字元數 |
| 重疊字元 | `CHUNK_OVERLAP` | 50 | 相鄰片段重疊量 |
| 分割符號 | `CHUNK_SEPARATORS` | 段落→換行→句號→逗號→空格 | 分割優先順序 |
| 預設 Top-K | `DEFAULT_TOP_K` | 3 | 預設檢索片段數 |
| 高度相關門檻 | `SCORE_HIGH` | 85 | > 此值為 🟢 |
| 中度相關門檻 | `SCORE_MID` | 70 | ≥ 此值為 🟡 |
| Prompt | `SYSTEM_PROMPT` | （見 config.py） | LLM 系統提示詞 |

**使用方式：** 修改 `config.py` 後重啟 `python3 rag_app.py` 即可。

---

## 11. RAG 優化路線規劃（2026-02-05 新增）

### 11.1 優化不只是調參數

RAG 優化是多維度的工程，不只是調整 chunk_size 或 temperature。我們討論後整理出一條由下而上的系統性優化路線：

### 11.2 優化金字塔（由底層往上）

```
                    ┌─────────────────────┐
        Level 5     │   進階架構           │  Corrective RAG、Agentic RAG
                    ├─────────────────────┤
        Level 4     │   生成優化           │  Prompt 工程、Context 管理
                    ├─────────────────────┤
        Level 3     │   查詢優化           │  Multi-query、HyDE
                    ├─────────────────────┤
        Level 2     │   檢索優化           │  Hybrid Search、Re-ranking
                    ├─────────────────────┤
        Level 1     │   資料層優化         │  分割策略、Metadata 豐富化
                    ├─────────────────────┤
        Level 0     │   評估框架（地基）    │  量化指標、測試集、基準線
                    └─────────────────────┘
```

### 11.3 各層級說明與順序邏輯

| 層級 | 內容 | 為什麼在這個順序 |
|------|------|-----------------|
| **L0 評估框架** | 建立量化指標、測試集、baseline 分數 | 沒有量化就無法判斷優化是「變好」還是「變差」— 這是地基 |
| **L1 資料層優化** | Semantic Chunking、Metadata 豐富化 | 檢索品質的上限取決於資料切割品質 — 切爛了，後面怎麼搜都搜不好 |
| **L2 檢索優化** | Hybrid Search、Re-ranking | 資料切好了，才有意義去改善「怎麼搜」 |
| **L3 查詢優化** | Multi-query、HyDE（假設性回答檢索） | 檢索方法穩定後，再優化「拿什麼去搜」 |
| **L4 生成優化** | Prompt 工程、Chain-of-Thought、Context 管理 | 前面都到位了，最後優化 LLM 怎麼組織和生成答案 |
| **L5 進階架構** | Corrective RAG、Agentic RAG | 當基本 RAG 已經很好，再考慮自我修正、多步推理 |

### 11.4 討論決策

- **為什麼不隨機挑優化方向？** 每一層都建立在前一層之上，就像蓋房子 — 地基（評估）→ 建材品質（資料）→ 結構（檢索）→ 裝潢（生成）
- **為什麼從評估開始？** 「You can't improve what you can't measure」— 先有 baseline 才能量化每次改動的效果
- **實作計畫：** 從 Level 0 開始，逐層往上實作

---

## 12. 延伸學習建議

1. 嘗試載入真實的 PDF 醫學文獻（使用 LangChain 的 PyPDFLoader）
2. 調整 chunk_size 和 chunk_overlap 觀察對檢索品質的影響
3. 嘗試不同的 LLM 模型（如 llama3.2:1b 或更大的模型）
4. 加入 metadata filtering 做更精確的檢索
5. 實作 multi-query retrieval 提高召回率
