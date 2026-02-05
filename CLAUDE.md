# NLP 專案規則

## 1. 自動更新實作紀錄

每次討論或實作完成後，必須自動將以下內容更新至 `RAG_實作紀錄.md`：

- 新增的討論決策與理由
- 新的實作步驟或程式碼變更
- 測試結果與發現
- 遇到的問題與解決方式
- 新學到的概念或技巧

**更新方式：** 直接編輯 `RAG_實作紀錄.md`，在對應章節中新增內容。如果是全新主題，則新增章節。

**紀錄文件位置：** `NLP/RAG_實作紀錄.md`

## 2. 版本控制

本專案使用 Git 進行版控，遠端 repo：`yaurueitian-wq/medical-rag-nlp`（Private）。

### 規則

- **每次完成一個功能或修改後**，主動提醒使用者是否要 commit 並 push
- Commit message 使用中文描述，格式：`類型: 簡述`
  - 類型：`feat`（新功能）、`fix`（修復）、`docs`（文件）、`refactor`（重構）、`style`（樣式）
- 不要自動 commit/push，必須經過使用者確認
- `.gitignore` 已排除：`chroma_db/`、`__pycache__/`、`.claude/`、`.DS_Store`
