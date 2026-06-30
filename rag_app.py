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
import json
import os
import gradio as gr
import config as cfg
from medical_data import medical_documents

# PubMed API
from Bio import Entrez

# 設定 NCBI Email（必須，用於識別請求來源）
Entrez.email = "medical-rag@example.com"

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
# 2.5 使用者回饋紀錄
# ============================================================
FEEDBACK_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_log.jsonl")


def compute_score_distribution(retrieved_sources: list) -> dict:
    """
    從檢索結果的分數列表，計算分數分佈統計，作為回饋資料的輔助診斷訊號。

    - top1_score：最高分（最相關片段的相似度）
    - score_gap：Top1 與 Top2 的分數差距（陡降 vs. 平坦的量化指標）
    - avg_score：平均分數
    - score_std：分數的標準差（離散程度，越小代表分佈越平坦）
    """
    scores = [s["score"] for s in retrieved_sources]
    if not scores:
        return {"top1_score": None, "score_gap": None, "avg_score": None, "score_std": None}

    n = len(scores)
    avg = sum(scores) / n
    variance = sum((s - avg) ** 2 for s in scores) / n
    gap = scores[0] - scores[1] if n >= 2 else None

    return {
        "top1_score": round(scores[0], 1),
        "score_gap": round(gap, 1) if gap is not None else None,
        "avg_score": round(avg, 1),
        "score_std": round(variance ** 0.5, 1),
    }


def save_feedback(state: dict, rating: str, failure_type: str, comment: str) -> str:
    """將使用者對單次問答的回饋（👍/👎＋失敗類型＋意見）以 JSONL 格式追加寫入。"""
    if not state or not state.get("question"):
        return "尚未有可回饋的問答，請先提問。"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": state["question"],
        "answer": state["answer"],
        "top_k": state["top_k"],
        "retrieved_sources": state["retrieved_sources"],
        "score_distribution": compute_score_distribution(state["retrieved_sources"]),
        "rating": rating,
        "failure_type": failure_type if rating == "down" else None,
        "comment": comment.strip() if comment else "",
    }
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    icon = "👍" if rating == "up" else "👎"
    msg = f"{icon} 已記錄你的評分"
    if entry["comment"]:
        msg += f"，意見：「{entry['comment']}」"
    msg += "，感謝你的回饋！"
    return msg


# ============================================================
# 2.6 知識庫來源追蹤
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
# 2.7 PubMed 搜尋與導入
# ============================================================
# 暫存搜尋結果（用於導入）
_pubmed_cache = {}


def search_pubmed(query: str, max_results: int = 10):
    """搜尋 PubMed 並返回摘要列表。"""
    if not query.strip():
        return "請輸入搜尋關鍵字。", gr.update(choices=[], value=[])

    try:
        # 搜尋 PubMed
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()

        id_list = record.get("IdList", [])
        if not id_list:
            return f"找不到與「{query}」相關的文獻。", []

        # 取得文獻詳細資料
        handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        results = []
        for article in records.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            article_data = medline.get("Article", {})
            pmid = str(medline.get("PMID", ""))

            # 標題
            title = article_data.get("ArticleTitle", "無標題")

            # 摘要
            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            if abstract_parts:
                # 有些摘要是結構化的（分段），有些是純文字
                abstract = " ".join(
                    str(p) if isinstance(p, str) else str(p)
                    for p in abstract_parts
                )
            else:
                abstract = ""

            # 期刊和年份
            journal = article_data.get("Journal", {}).get("Title", "未知期刊")
            pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year", "") or pub_date.get("MedlineDate", "")[:4] if pub_date.get("MedlineDate") else ""

            # 作者
            author_list = article_data.get("AuthorList", [])
            if author_list:
                first_author = author_list[0]
                author_name = first_author.get("LastName", "") + " " + first_author.get("Initials", "")
                if len(author_list) > 1:
                    author_name += " et al."
            else:
                author_name = "未知作者"

            results.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "author": author_name,
            })

        # 快取結果
        _pubmed_cache.clear()
        for r in results:
            _pubmed_cache[r["pmid"]] = r

        # 格式化：摘要資訊顯示在選項標籤中
        total_with_abstract = sum(1 for r in results if r["abstract"])
        display = f"### 找到 {len(results)} 篇文獻（{total_with_abstract} 篇有摘要可導入）\n"

        # 建立選項：每個選項包含完整資訊
        choices = []
        for r in results:
            if r["abstract"]:
                label = f"[{r['pmid']}] {r['title'][:60]}{'...' if len(r['title']) > 60 else ''} | {r['author']} | {r['journal']} ({r['year']})"
                choices.append(label)

        return display, gr.update(choices=choices, value=[])

    except Exception as e:
        return f"❌ 搜尋失敗：{str(e)}", gr.update(choices=[], value=[])


def import_pubmed_articles(selected_items: list):
    """將選中的 PubMed 文獻導入知識庫。"""
    if not selected_items:
        return "請先選擇要導入的文獻。", get_kb_inventory()

    imported = []
    for item in selected_items:
        # 從 "[PMID] Title..." 格式提取 PMID
        pmid = item.split("]")[0].replace("[", "").strip()
        if pmid not in _pubmed_cache:
            continue

        article = _pubmed_cache[pmid]
        if not article["abstract"]:
            continue

        # 組合文獻內容
        content = f"""標題：{article['title']}

作者：{article['author']}
期刊：{article['journal']} ({article['year']})
PMID：{article['pmid']}

摘要：
{article['abstract']}
"""
        # 建立 Document
        doc = Document(
            page_content=content,
            metadata={
                "source": f"PubMed:{article['pmid']}",
                "topic": article["title"][:50],
                "journal": article["journal"],
                "year": article["year"],
            }
        )

        # 分割並加入向量庫
        new_splits = text_splitter.split_documents([doc])
        vectorstore.add_documents(new_splits)

        # 更新 registry
        kb_registry.append({
            "source": f"PubMed:{article['pmid']}",
            "topics": [article["title"][:30] + "..."],
            "chunks": len(new_splits),
            "type": "PubMed",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

        imported.append(article["pmid"])

    if imported:
        msg = f"✅ 成功導入 {len(imported)} 篇文獻！\n\n"
        msg += "**已導入 PMID：** " + ", ".join(imported)
    else:
        msg = "❌ 沒有成功導入任何文獻。請確認選擇的文獻有摘要。"

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
        return "請輸入問題。", "", {}

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

    # --- 紀錄本次問答內容，供回饋按鈕使用 ---
    query_state = {
        "question": question,
        "answer": answer,
        "top_k": top_k,
        "retrieved_sources": [
            {
                "source": doc.metadata.get("source", "未知"),
                "topic": doc.metadata.get("topic", "未知"),
                "score": round(score * 100, 1),
            }
            for doc, score in results_with_scores
        ],
    }

    return answer, retrieval_display, query_state


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
/* === 全域字體 + 緊湊間距 === */
* {
    font-family: "PingFang TC", "Microsoft JhengHei", "Noto Sans TC",
                 "Helvetica Neue", Arial, sans-serif !important;
}
code, pre, .prose code {
    font-family: "SF Mono", "Menlo", "Consolas", "Courier New", monospace !important;
}

/* === 整體容器：減少上下 padding === */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}

/* === 減少 Gradio 預設的區塊間距 === */
.gradio-container > .flex {
    gap: 8px !important;
}

/* === 頂部橫幅：縮小 === */
.header-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%) !important;
    border-radius: 12px !important;
    padding: 16px 24px !important;
    margin-bottom: 10px !important;
}
.header-banner h1 {
    font-size: 1.4em !important;
    margin-bottom: 4px !important;
}
.header-banner h1, .header-banner p, .header-banner span,
.header-banner strong, .header-banner em, .header-banner blockquote,
.header-banner * {
    color: #ffffff !important;
}
.header-banner blockquote {
    border-left-color: rgba(255,255,255,0.4) !important;
    margin: 4px 0 !important;
    padding: 2px 12px !important;
    font-size: 0.88em !important;
}

/* === 卡片區塊：壓縮 padding 和 margin === */
.card {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* === 區塊標題：縮小間距 === */
.card-title {
    font-size: 1em !important;
    font-weight: 700 !important;
    color: #1e3a5f !important;
    margin-bottom: 10px !important;
    padding-bottom: 6px !important;
    border-bottom: 2px solid #e2e8f0 !important;
}

/* === 結果卡片：降低最小高度 === */
.result-card {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    background: #f8fafc !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    min-height: 180px !important;
}

/* === 說明面板：縮小 === */
.info-panel {
    background: #f0f5ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
}
.info-panel p, .info-panel li, .info-panel strong {
    font-size: 0.85em !important;
    line-height: 1.5 !important;
    margin: 2px 0 !important;
}

/* === 提問按鈕 === */
.submit-btn {
    min-height: 40px !important;
    font-size: 1em !important;
    border-radius: 8px !important;
}

/* === 範例按鈕：更緊湊 === */
.example-btn {
    background: linear-gradient(135deg, #f0f5ff 0%, #e8f0fe 100%) !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 16px !important;
    padding: 6px 14px !important;
    font-size: 0.85em !important;
    color: #1e3a5f !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    text-align: left !important;
    margin: 0 !important;
}
.example-btn:hover {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
    border-color: #93c5fd !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.15) !important;
}

/* === 減少 Row 間的 gap === */
.gap-4, .gap-2 {
    gap: 8px !important;
}

/* === 輸入框：減少高度 === */
textarea {
    min-height: 50px !important;
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
        gr.Markdown("**💬 快速提問範例** — 點擊直接填入", elem_classes="card-title")
        with gr.Row(elem_classes="gap-2"):
            example_btns = []
            for q in EXAMPLE_QUESTIONS:
                btn = gr.Button(q, elem_classes="example-btn", size="sm")
                example_btns.append(btn)

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

    # ---- 回饋區（使用者可自行選擇是否參與） ----
    last_query_state = gr.State({})
    feedback_toggle = gr.Checkbox(
        label="🗳️ 我願意針對 AI 回答提供回饋，協助改善這個系統",
        value=False,
        info="此功能用於系統優化研究，採自願參與；勾選後才會顯示回饋區塊，且不影響你正常使用問答功能",
    )
    selected_rating_state = gr.State(None)
    with gr.Group(elem_classes="card", visible=False) as feedback_group:
        gr.Markdown(
            "### 這個回答對你有幫助嗎？\n"
            "*先選擇評分，視需要補充說明，最後按「送出回饋」即可*",
            elem_classes="card-title",
        )
        with gr.Row():
            feedback_up_btn = gr.Button("👍 有幫助", size="sm")
            feedback_down_btn = gr.Button("👎 沒幫助", size="sm")
        rating_display = gr.Markdown()
        failure_type_radio = gr.Radio(
            label="可以告訴我們問題出在哪裡嗎？（選填，有助於我們對症下藥）",
            choices=[
                ("🔍 檢索到的資料不相關", "A"),
                ("📊 資料相關，但排序怪異", "B"),
                ("🤖 資料正確，但 AI 回答曲解／編造", "C"),
                ("📭 知識庫沒有這個主題的資料", "D"),
                ("❓ 我的問題問得不夠清楚", "E"),
                ("🤷 不確定 / 其他", "unknown"),
            ],
            visible=False,
        )
        feedback_comment = gr.Textbox(
            label="意見回饋（選填）",
            placeholder="例如：檢索到的內容不相關、答案不夠完整...",
            lines=1,
        )
        feedback_submit_btn = gr.Button("📨 送出回饋", variant="primary", size="sm")
        feedback_status = gr.Markdown()

    feedback_toggle.change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=feedback_toggle,
        outputs=feedback_group,
    )

    # 點擊👍／👎 只是「選擇評分」，尚未送出；👎 額外顯示失敗類型選項
    feedback_up_btn.click(
        fn=lambda: ("up", "已選擇：👍 有幫助", gr.update(visible=False, value=None)),
        outputs=[selected_rating_state, rating_display, failure_type_radio],
    )
    feedback_down_btn.click(
        fn=lambda: ("down", "已選擇：👎 沒幫助", gr.update(visible=True)),
        outputs=[selected_rating_state, rating_display, failure_type_radio],
    )

    # 按下「送出回饋」才真正寫入紀錄（此時已包含選好的評分、失敗類型、意見內容）
    feedback_submit_btn.click(
        fn=save_feedback,
        inputs=[last_query_state, selected_rating_state, failure_type_radio, feedback_comment],
        outputs=feedback_status,
    )

    # ---- 知識庫管理區 ----
    with gr.Accordion("📦 知識庫管理", open=False):
        with gr.Group(elem_classes="card"):
            gr.Markdown("### 📋 目前知識庫來源", elem_classes="card-title")
            kb_display = gr.Markdown(value=get_kb_inventory)

        with gr.Group(elem_classes="card"):
            gr.Markdown("### 🔬 從 PubMed 搜尋並導入", elem_classes="card-title")
            with gr.Row():
                pubmed_query = gr.Textbox(
                    label="搜尋關鍵字",
                    placeholder="例如：diabetes treatment 2024",
                    scale=3,
                )
                pubmed_count = gr.Slider(
                    minimum=5, maximum=20, value=10, step=1,
                    label="搜尋筆數",
                    scale=1,
                )
            pubmed_search_btn = gr.Button(
                "🔍 搜尋 PubMed", variant="secondary",
                elem_classes="submit-btn",
            )
            pubmed_results = gr.Markdown(value="*輸入關鍵字後點擊搜尋*")
            pubmed_select = gr.CheckboxGroup(
                label="勾選要導入的文獻",
                choices=[],
                info="只有有摘要的文獻才會顯示在這裡",
            )
            pubmed_import_btn = gr.Button(
                "📥 導入選中的文獻", variant="primary",
                elem_classes="submit-btn",
            )
            pubmed_status = gr.Markdown()

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
        outputs=[answer_output, retrieval_output, last_query_state],
    ).then(
        fn=lambda: ("", "", None, gr.update(visible=False, value=None), ""),
        outputs=[feedback_status, rating_display, selected_rating_state, failure_type_radio, feedback_comment],
    )

    # 範例按鈕：點擊後填入問題
    for btn, q in zip(example_btns, EXAMPLE_QUESTIONS):
        btn.click(fn=lambda x=q: x, outputs=question_input)
    question_input.submit(
        fn=rag_query,
        inputs=[question_input, top_k_slider],
        outputs=[answer_output, retrieval_output, last_query_state],
    ).then(
        fn=lambda: ("", "", None, gr.update(visible=False, value=None), ""),
        outputs=[feedback_status, rating_display, selected_rating_state, failure_type_radio, feedback_comment],
    )

    upload_btn.click(
        fn=upload_file,
        inputs=[file_input, source_input, topic_input],
        outputs=[upload_status, kb_display],
    )

    # PubMed 搜尋
    pubmed_search_btn.click(
        fn=search_pubmed,
        inputs=[pubmed_query, pubmed_count],
        outputs=[pubmed_results, pubmed_select],
    )

    # PubMed 導入
    pubmed_import_btn.click(
        fn=import_pubmed_articles,
        inputs=[pubmed_select],
        outputs=[pubmed_status, kb_display],
    )

# ============================================================
# 5. 啟動
# ============================================================
if __name__ == "__main__":
    print("\n🚀 啟動 Web 介面...")
    print("   開啟瀏覽器前往: http://localhost:7860")
    app.launch(server_name=cfg.SERVER_HOST, server_port=cfg.SERVER_PORT)
