# 醫療知識 RAG 問答系統

> 北醫「醫療 AI 實戰力養成班：從數據到臨床的智慧應用」課程專案
> 使用 **Ollama**（本地 LLM）+ **ChromaDB**（向量資料庫）+ **LangChain**（RAG 框架）+ **Gradio**（Web UI）建構的本地端醫療知識問答系統

---

## 專案簡介

這是一個基於 **RAG（Retrieval-Augmented Generation，檢索增強生成）** 的醫療知識問答系統：使用者提問後，系統先從醫療知識庫中檢索最相關的文本片段，再交由本地 LLM 根據這些片段生成回答，並標示每段參考資料的關聯度分數，降低幻覺、提高回答可信度。

整個系統皆在**本地端**運行，不需要外部 API key，兼顧隱私性與離線可用性。

## 技術架構

| 元件 | 使用技術 |
|---|---|
| LLM（生成回答） | Ollama — `llama3.2:3b` |
| Embedding（向量嵌入） | Ollama — `nomic-embed-text` |
| 向量資料庫 | ChromaDB |
| RAG 框架 | LangChain |
| Web 介面 | Gradio |
| 文獻檢索 | PubMed API（Biopython `Entrez`） |

## 主要功能

- **互動問答介面**：輸入醫療問題，顯示 AI 回答與檢索到的參考片段（含關聯度分數標示：🟢高度相關 / 🟡中度相關 / 🔴低度相關）
- **可調整檢索參數**：透過 Top-K 滑桿調整每次檢索的片段數量
- **知識庫管理**：
  - 顯示目前知識庫來源清單與統計
  - 支援上傳 PDF / TXT 文件擴充知識庫
  - 支援搜尋並導入 PubMed 文獻摘要
- **使用者回饋機制**（自願參與）：
  - 使用者可自行勾選是否參與回饋（預設不顯示，不干擾一般使用體驗）
  - 提供 👍/👎 評分、失敗類型分類（檢索不相關 / 排序錯誤 / 生成幻覺 / 知識庫缺漏 / 問題模糊等）、自由意見回饋
  - 回饋資料以 JSONL 格式記錄，作為後續系統優化的診斷依據

## 安裝與啟動

### 1. 安裝 Ollama 並下載模型

```bash
brew install ollama
brew services start ollama

ollama pull llama3.2:3b        # LLM 模型（約 2GB）
ollama pull nomic-embed-text   # Embedding 模型（約 274MB）
```

### 2. 安裝 Python 套件

```bash
pip3 install chromadb langchain langchain-ollama langchain-chroma \
             langchain-text-splitters langchain-community gradio biopython
```

### 3. 啟動系統

```bash
python3 rag_app.py
```

啟動後開啟瀏覽器前往 [http://localhost:7860](http://localhost:7860) 即可使用。

## 專案結構

```
.
├── rag_app.py              # 主程式：Gradio Web UI + RAG pipeline + 回饋機制
├── config.py               # 集中管理可調參數（模型、chunk 設定、檢索參數等）
├── medical_data.py         # 內建醫療知識庫資料
├── eval_rag.py             # RAG 評估腳本（離線評估框架 + 實驗追蹤系統）
├── test_questions.json     # 評估用測試題庫
├── experiments/            # 各版本評估結果與 leaderboard
├── feedback_log.jsonl      # 使用者回饋紀錄（已排除版控）
├── RAG_實作紀錄.md          # 開發過程的討論決策、實作步驟、測試結果紀錄
└── RAG_精準度優化_SPEC.md   # RAG 精準度優化路線圖與回饋系統設計規格
```

## 系統優化與評估

本專案具備一套離線評估框架（`eval_rag.py`），可量化檢索精準度（Hit Rate、MRR、相似度）、回答品質（關鍵字覆蓋率、LLM-as-Judge 忠實度評分），並將每次調整參數後的結果記錄至 `experiments/` 目錄，自動產生比較用的 leaderboard。

更完整的優化路線規劃（從基礎 RAG → 可評估 RAG → 使用者回饋機制 → 輕量知識圖譜）詳見 [RAG_精準度優化_SPEC.md](RAG_精準度優化_SPEC.md)。

## 開發紀錄

完整的討論過程、技術決策、實作步驟與測試結果，詳見 [RAG_實作紀錄.md](RAG_實作紀錄.md)。
