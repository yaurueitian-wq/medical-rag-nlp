"""
============================================================
RAG 醫療知識問答系統 — Gradio Web UI 版
============================================================
功能：
  1. Web 介面互動問答
  2. 顯示檢索到的文件片段與關聯度分數
  3. 支援調整檢索參數（top-k）
  4. 知識庫管理：顯示來源清單、支援上傳 PDF/TXT 擴充
============================================================
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime
import os
import gradio as gr
import config as cfg
from medical_data import medical_documents

# ============================================================
# 2. 建立向量資料庫
# ============================================================
print("⏳ 正在初始化 RAG 系統...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.CHUNK_SIZE,
    chunk_overlap=cfg.CHUNK_OVERLAP,
    separators=cfg.CHUNK_SEPARATORS,
)
splits = text_splitter.split_documents(medical_documents)

embeddings = OllamaEmbeddings(model=cfg.EMBEDDING_MODEL)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="medical_app",
)

llm = ChatOllama(model=cfg.LLM_MODEL, temperature=cfg.LLM_TEMPERATURE)

prompt = ChatPromptTemplate.from_template(cfg.SYSTEM_PROMPT)

print(f"✅ RAG 系統初始化完成（{len(splits)} 個文本片段）")

# ============================================================
# 2.5 知識庫來源追蹤
# ============================================================
kb_registry = []
INIT_DATE = datetime.now().strftime("%Y-%m-%d %H:%M")

# 從 medical_documents 中提取初始來源
_seen = {}
for doc in medical_documents:
    src = doc.metadata.get("source", "未知")
    topic = doc.metadata.get("topic", "未知")
    _seen.setdefault(src, []).append(topic)
for src, topics in _seen.items():
    kb_registry.append({
        "source": src,
        "topics": topics,
        "chunks": sum(
            1 for s in splits if s.metadata.get("source") == src
        ),
        "type": "內建",
        "date": INIT_DATE,
    })


def get_kb_inventory() -> str:
    """產生知識庫清單的 Markdown 表格。"""
    total_chunks = sum(e["chunks"] for e in kb_registry)
    md = f"**共 {len(kb_registry)} 個來源 ／ {total_chunks} 個文本片段**\n\n"
    md += "| # | 來源 | 主題 | 片段數 | 類型 | 加入時間 |\n"
    md += "|---|------|------|--------|------|----------|\n"
    for i, e in enumerate(kb_registry, 1):
        topics_str = "、".join(e["topics"])
        md += (
            f"| {i} | {e['source']} | {topics_str} | "
            f"{e['chunks']} | {e['type']} | {e['date']} |\n"
        )
    return md


def upload_file(file, source_name: str, topic_name: str):
    """處理上傳的 PDF 或 TXT 檔案，加入向量資料庫。"""
    if file is None:
        return "請先選擇檔案。", get_kb_inventory()

    file_path = file if isinstance(file, str) else file.name
    ext = os.path.splitext(file_path)[1].lower()

    # 讀取文件內容
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        raw_docs = [Document(page_content=text)]
    else:
        return f"不支援的檔案格式：{ext}（僅支援 .pdf 和 .txt）", get_kb_inventory()

    if not raw_docs or not any(d.page_content.strip() for d in raw_docs):
        return "檔案內容為空，無法加入知識庫。", get_kb_inventory()

    # 自動填入來源 / 主題
    fname = os.path.basename(file_path)
    if not source_name.strip():
        source_name = fname
    if not topic_name.strip():
        topic_name = os.path.splitext(fname)[0]

    # 加入 metadata
    for d in raw_docs:
        d.metadata["source"] = source_name
        d.metadata["topic"] = topic_name

    # 分割 → Embedding → 加入 vectorstore
    new_splits = text_splitter.split_documents(raw_docs)
    vectorstore.add_documents(new_splits)

    # 更新 registry
    kb_registry.append({
        "source": source_name,
        "topics": [topic_name],
        "chunks": len(new_splits),
        "type": "上傳 " + ext.replace(".", "").upper(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    msg = (
        f"✅ 成功加入知識庫！\n\n"
        f"- **檔案：** {fname}\n"
        f"- **來源：** {source_name}\n"
        f"- **主題：** {topic_name}\n"
        f"- **新增片段：** {len(new_splits)} 個"
    )
    return msg, get_kb_inventory()


# ============================================================
# 3. 核心功能：帶關聯度分數的檢索 + 生成
# ============================================================
def rag_query(question: str, top_k: int = 3):
    """
    執行 RAG 查詢，返回 LLM 回答和帶分數的檢索結果。

    ChromaDB 的 similarity_search_with_relevance_scores 會返回
    (document, score) 的列表，score 越高表示越相關（0~1 之間）。
    """
    if not question.strip():
        return "請輸入問題。", ""

    # --- 檢索：取得文件 + 關聯度分數 ---
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query=question,
        k=top_k,
    )

    # --- 格式化檢索結果（給使用者看） ---
    retrieval_display = ""
    for i, (doc, score) in enumerate(results_with_scores):
        # 將分數轉為百分比，更直觀
        score_pct = score * 100
        topic = doc.metadata.get("topic", "未知")
        source = doc.metadata.get("source", "未知")
        content = doc.page_content.strip()[:200]

        # 根據分數決定顏色標籤
        if score_pct > cfg.SCORE_HIGH:
            level = "🟢 高度相關"
        elif score_pct >= cfg.SCORE_MID:
            level = "🟡 中度相關"
        else:
            level = "🔴 低度相關"

        retrieval_display += f"### 片段 {i+1}  |  {level}  |  相似度：{score_pct:.1f}%\n"
        retrieval_display += f"**來源：** {source} — {topic}\n\n"
        retrieval_display += f"> {content}...\n\n"
        retrieval_display += "---\n\n"

    # --- 組合 context 送給 LLM ---
    context = "\n\n---\n\n".join(
        f"[來源: {doc.metadata.get('source', '')} - {doc.metadata.get('topic', '')}]\n{doc.page_content}"
        for doc, _ in results_with_scores
    )

    # --- LLM 生成回答 ---
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, retrieval_display


# ============================================================
# 4. Gradio Web 介面
# ============================================================
EXAMPLE_QUESTIONS = [
    "糖尿病有哪些類型？診斷標準是什麼？",
    "高血壓患者合併糖尿病，應該首選什麼降壓藥？",
    "急性心肌梗塞的典型症狀和急性處置原則是什麼？",
    "慢性腎臟病分幾期？治療原則是什麼？",
    "Metformin 的副作用和禁忌症？",
]

custom_css = """
/* === 全域字體 === */
* {
    font-family: "PingFang TC", "Microsoft JhengHei", "Noto Sans TC",
                 "Helvetica Neue", Arial, sans-serif !important;
}
code, pre, .prose code {
    font-family: "SF Mono", "Menlo", "Consolas", "Courier New", monospace !important;
}

/* === 整體容器 === */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* === 頂部橫幅 === */
.header-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%) !important;
    border-radius: 16px !important;
    padding: 28px 32px !important;
    margin-bottom: 24px !important;
}
.header-banner h1, .header-banner p, .header-banner span,
.header-banner strong, .header-banner em, .header-banner blockquote,
.header-banner * {
    color: #ffffff !important;
}
.header-banner blockquote {
    border-left-color: rgba(255,255,255,0.4) !important;
}

/* === 卡片區塊 === */
.card {
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* === 區塊標題 === */
.card-title {
    font-size: 1.1em !important;
    font-weight: 700 !important;
    color: #1e3a5f !important;
    margin-bottom: 16px !important;
    padding-bottom: 10px !important;
    border-bottom: 2px solid #e2e8f0 !important;
}

/* === 結果卡片 === */
.result-card {
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 24px !important;
    background: #f8fafc !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    min-height: 250px !important;
}

/* === 說明面板 === */
.info-panel {
    background: #f0f5ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.info-panel p, .info-panel li, .info-panel strong {
    font-size: 0.92em !important;
    line-height: 1.7 !important;
}

/* === 提問按鈕加大 === */
.submit-btn {
    min-height: 44px !important;
    font-size: 1.05em !important;
    border-radius: 10px !important;
}
"""

with gr.Blocks(
    title="醫療知識 RAG 問答系統",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=custom_css,
) as app:

    # ---- 頂部橫幅 ----
    gr.Markdown("""
    # 🏥 醫療知識 RAG 問答系統
    > 使用 **Ollama (llama3.2:3b)** + **ChromaDB** + **LangChain** 建構的本地 RAG 系統
    >
    > 知識庫涵蓋：糖尿病、高血壓、急性心肌梗塞、慢性腎臟病
    """, elem_classes="header-banner")

    # ---- 提問區 ----
    with gr.Group(elem_classes="card"):
        gr.Markdown("### 📝 提問區", elem_classes="card-title")
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="輸入你的醫療問題",
                    placeholder="例如：高血壓合併糖尿病應該用什麼藥？",
                    lines=2,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=cfg.MAX_TOP_K, value=cfg.DEFAULT_TOP_K, step=1,
                    label="檢索片段數 (Top-K)",
                    info="從知識庫中檢索最相關的 K 個文本片段",
                )
            with gr.Column(scale=1, min_width=220):
                gr.Markdown(f"""
**💡 使用說明**
1. 輸入你的醫療問題
2. 調整 Top-K 檢索片段數
3. 按「提問」或按 Enter
4. 下方顯示 AI 回答與檢索依據

**📊 關聯度**
- 🟢 > {cfg.SCORE_HIGH}% 高度相關
- 🟡 {cfg.SCORE_MID}-{cfg.SCORE_HIGH}% 中度相關
- 🔴 < {cfg.SCORE_MID}% 低度相關
                """, elem_classes="info-panel")
        submit_btn = gr.Button(
            "🔍 提問", variant="primary",
            elem_classes="submit-btn",
        )

    # ---- 快速範例 ----
    with gr.Group(elem_classes="card"):
        gr.Markdown("### 💬 快速提問範例", elem_classes="card-title")
        examples = gr.Examples(
            examples=EXAMPLE_QUESTIONS,
            inputs=question_input,
        )

    # ---- 結果區 ----
    with gr.Row(equal_height=True):
        with gr.Column():
            with gr.Group(elem_classes="result-card"):
                gr.Markdown("### 💡 AI 回答", elem_classes="card-title")
                answer_output = gr.Markdown()
        with gr.Column():
            with gr.Group(elem_classes="result-card"):
                gr.Markdown("### 📚 檢索結果與關聯度", elem_classes="card-title")
                retrieval_output = gr.Markdown()

    # ---- 知識庫管理區 ----
    with gr.Accordion("📦 知識庫管理", open=False):
        with gr.Group(elem_classes="card"):
            gr.Markdown("### 📋 目前知識庫來源", elem_classes="card-title")
            kb_display = gr.Markdown(value=get_kb_inventory)

        with gr.Group(elem_classes="card"):
            gr.Markdown("### 📤 上傳新文件至知識庫", elem_classes="card-title")
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="選擇檔案（PDF 或 TXT）",
                        file_types=[".pdf", ".txt"],
                    )
                with gr.Column(scale=1):
                    source_input = gr.Textbox(
                        label="來源名稱（選填）",
                        placeholder="例如：臨床指引 2024",
                    )
                    topic_input = gr.Textbox(
                        label="主題名稱（選填）",
                        placeholder="例如：心房顫動治療",
                    )
            upload_btn = gr.Button(
                "📤 上傳並加入知識庫", variant="secondary",
                elem_classes="submit-btn",
            )
            upload_status = gr.Markdown()

    # 綁定事件
    submit_btn.click(
        fn=rag_query,
        inputs=[question_input, top_k_slider],
        outputs=[answer_output, retrieval_output],
    )
    question_input.submit(
        fn=rag_query,
        inputs=[question_input, top_k_slider],
        outputs=[answer_output, retrieval_output],
    )
    upload_btn.click(
        fn=upload_file,
        inputs=[file_input, source_input, topic_input],
        outputs=[upload_status, kb_display],
    )

# ============================================================
# 5. 啟動
# ============================================================
if __name__ == "__main__":
    print("\n🚀 啟動 Web 介面...")
    print("   開啟瀏覽器前往: http://localhost:7860")
    app.launch(server_name=cfg.SERVER_HOST, server_port=cfg.SERVER_PORT)
