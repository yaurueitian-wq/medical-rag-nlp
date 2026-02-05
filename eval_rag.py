"""
============================================================
RAG 評估腳本 — 實驗追蹤系統
============================================================
用法：
  python3 eval_rag.py                        # 互動式（輸入版本名稱）
  python3 eval_rag.py --version v0_baseline   # 指定版本名稱
============================================================
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config as cfg
from medical_data import medical_documents

# ============================================================
# 路徑設定
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
TEST_QUESTIONS_PATH = os.path.join(BASE_DIR, "test_questions.json")
LOG_PATH = os.path.join(EXPERIMENTS_DIR, "optimization_log.md")


# ============================================================
# 1. 建立 RAG Pipeline（與 rag_app.py 相同邏輯）
# ============================================================
def build_rag_pipeline():
    """建立 RAG 系統，回傳 vectorstore, llm, prompt。"""
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
        collection_name="medical_eval",
    )

    llm = ChatOllama(model=cfg.LLM_MODEL, temperature=cfg.LLM_TEMPERATURE)
    prompt = ChatPromptTemplate.from_template(cfg.SYSTEM_PROMPT)

    print(f"✅ RAG 系統初始化完成（{len(splits)} 個文本片段）")
    return vectorstore, llm, prompt, len(splits)


# ============================================================
# 2. 評估指標
# ============================================================
def evaluate_retrieval(results_with_scores, expected_sources):
    """
    評估檢索品質。

    回傳：
    - hit: 是否命中（至少一個預期來源出現在檢索結果中）
    - mrr: Mean Reciprocal Rank（第一個命中的倒數排名）
    - avg_score: 平均相似度分數
    - top1_score: 最高相似度分數
    """
    hit = False
    mrr = 0.0
    scores = []

    for rank, (doc, score) in enumerate(results_with_scores, 1):
        scores.append(score)
        source = doc.metadata.get("source", "")
        if source in expected_sources and not hit:
            hit = True
            mrr = 1.0 / rank

    avg_score = sum(scores) / len(scores) if scores else 0
    top1_score = scores[0] if scores else 0

    return {
        "hit": hit,
        "mrr": mrr,
        "avg_score": round(avg_score * 100, 1),
        "top1_score": round(top1_score * 100, 1),
    }


def evaluate_keywords(answer, expected_keywords):
    """
    評估關鍵字覆蓋率。

    回傳：
    - coverage: 預期關鍵字出現在答案中的比例 (0~1)
    - matched: 命中的關鍵字列表
    - missed: 未命中的關鍵字列表
    """
    answer_lower = answer.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missed = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    coverage = len(matched) / len(expected_keywords) if expected_keywords else 0

    return {
        "coverage": round(coverage * 100, 1),
        "matched": matched,
        "missed": missed,
    }


def evaluate_faithfulness(llm, answer, context, question):
    """
    使用 LLM-as-Judge 評估忠實度（答案是否基於檢索到的參考資料）。

    回傳 1~5 的分數。
    """
    judge_prompt = ChatPromptTemplate.from_template(
        """你是一位評估專家。請判斷以下「AI回答」是否忠實於「參考資料」的內容。

評分標準（1-5分）：
1分：回答完全與參考資料無關，或明顯編造資訊
2分：回答部分相關，但包含明顯不在參考資料中的資訊
3分：回答大致基於參考資料，但有少量推測或不精確
4分：回答幾乎完全基於參考資料，表述準確
5分：回答完全忠實於參考資料，沒有任何編造

參考資料：
{context}

問題：{question}
AI回答：{answer}

請只回覆一個數字（1-5）："""
    )
    chain = judge_prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({
            "context": context,
            "question": question,
            "answer": answer,
        })
        score = int("".join(c for c in result.strip() if c.isdigit())[:1] or "3")
        return min(max(score, 1), 5)
    except Exception:
        return 3  # 預設中間值


# ============================================================
# 3. 執行完整評估
# ============================================================
def run_evaluation(vectorstore, llm, prompt, top_k=None):
    """對所有測試問題執行評估，回傳詳細結果。"""
    if top_k is None:
        top_k = cfg.DEFAULT_TOP_K

    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    total = len(questions)

    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        print(f"\n  [{i+1}/{total}] Q{qid}: {question}")

        # --- 檢索 ---
        t0 = time.time()
        retrieved = vectorstore.similarity_search_with_relevance_scores(
            query=question, k=top_k
        )
        retrieval_time = time.time() - t0

        # --- 生成 ---
        context = "\n\n---\n\n".join(
            f"[來源: {doc.metadata.get('source', '')} - "
            f"{doc.metadata.get('topic', '')}]\n{doc.page_content}"
            for doc, _ in retrieved
        )
        t0 = time.time()
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        generation_time = time.time() - t0

        # --- 評估 ---
        ret_eval = evaluate_retrieval(retrieved, q["expected_sources"])
        kw_eval = evaluate_keywords(answer, q["expected_keywords"])
        faith_score = evaluate_faithfulness(llm, answer, context, question)

        result = {
            "id": qid,
            "question": question,
            "category": q["category"],
            "difficulty": q["difficulty"],
            "answer": answer,
            "retrieval": ret_eval,
            "keyword": kw_eval,
            "faithfulness": faith_score,
            "retrieval_time": round(retrieval_time, 2),
            "generation_time": round(generation_time, 2),
            "retrieved_sources": [
                {
                    "source": doc.metadata.get("source", ""),
                    "topic": doc.metadata.get("topic", ""),
                    "score": round(score * 100, 1),
                }
                for doc, score in retrieved
            ],
        }
        results.append(result)

        # 即時顯示
        hit_mark = "✅" if ret_eval["hit"] else "❌"
        print(f"         檢索 {hit_mark} (Top1: {ret_eval['top1_score']}%) "
              f"| 關鍵字: {kw_eval['coverage']}% "
              f"| 忠實度: {faith_score}/5 "
              f"| 耗時: {retrieval_time + generation_time:.1f}s")

    return results


# ============================================================
# 4. 彙總統計
# ============================================================
def aggregate_results(results):
    """計算整體統計指標。"""
    n = len(results)
    return {
        "retrieval_hit_rate": round(
            sum(1 for r in results if r["retrieval"]["hit"]) / n * 100, 1
        ),
        "retrieval_mrr": round(
            sum(r["retrieval"]["mrr"] for r in results) / n, 3
        ),
        "avg_similarity": round(
            sum(r["retrieval"]["avg_score"] for r in results) / n, 1
        ),
        "avg_top1_similarity": round(
            sum(r["retrieval"]["top1_score"] for r in results) / n, 1
        ),
        "keyword_coverage": round(
            sum(r["keyword"]["coverage"] for r in results) / n, 1
        ),
        "faithfulness": round(
            sum(r["faithfulness"] for r in results) / n, 2
        ),
        "avg_retrieval_time": round(
            sum(r["retrieval_time"] for r in results) / n, 2
        ),
        "avg_generation_time": round(
            sum(r["generation_time"] for r in results) / n, 2
        ),
        "total_questions": n,
    }


# ============================================================
# 5. 儲存實驗結果
# ============================================================
def save_experiment(version_name, description, results, summary, num_chunks):
    """將實驗結果存到 experiments/version_name/ 目錄。"""
    exp_dir = os.path.join(EXPERIMENTS_DIR, version_name)
    os.makedirs(exp_dir, exist_ok=True)

    # --- config 快照 ---
    config_snapshot = {
        "llm_model": cfg.LLM_MODEL,
        "llm_temperature": cfg.LLM_TEMPERATURE,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "chunk_size": cfg.CHUNK_SIZE,
        "chunk_overlap": cfg.CHUNK_OVERLAP,
        "chunk_separators": cfg.CHUNK_SEPARATORS,
        "default_top_k": cfg.DEFAULT_TOP_K,
        "system_prompt": cfg.SYSTEM_PROMPT,
        "num_chunks": num_chunks,
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    # --- 完整結果 ---
    with open(os.path.join(exp_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "version": version_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "details": results,
        }, f, ensure_ascii=False, indent=2)

    # --- notes.md ---
    notes = f"""# {version_name}

**描述：** {description}
**日期：** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 評估結果摘要

| 指標 | 數值 |
|------|------|
| Retrieval Hit Rate | {summary['retrieval_hit_rate']}% |
| Retrieval MRR | {summary['retrieval_mrr']} |
| 平均相似度 | {summary['avg_similarity']}% |
| Top-1 相似度 | {summary['avg_top1_similarity']}% |
| 關鍵字覆蓋率 | {summary['keyword_coverage']}% |
| 忠實度 (LLM Judge) | {summary['faithfulness']}/5 |
| 平均檢索耗時 | {summary['avg_retrieval_time']}s |
| 平均生成耗時 | {summary['avg_generation_time']}s |

## 參數設定

- LLM: {cfg.LLM_MODEL} (temp={cfg.LLM_TEMPERATURE})
- Embedding: {cfg.EMBEDDING_MODEL}
- Chunk: size={cfg.CHUNK_SIZE}, overlap={cfg.CHUNK_OVERLAP}
- Top-K: {cfg.DEFAULT_TOP_K}
- 文本片段數: {num_chunks}

## 各題詳細結果

"""
    for r in results:
        hit = "✅" if r["retrieval"]["hit"] else "❌"
        notes += (
            f"### Q{r['id']}: {r['question']}\n"
            f"- 檢索: {hit} Top1={r['retrieval']['top1_score']}%\n"
            f"- 關鍵字: {r['keyword']['coverage']}% "
            f"(命中: {', '.join(r['keyword']['matched']) or '無'} / "
            f"未中: {', '.join(r['keyword']['missed']) or '無'})\n"
            f"- 忠實度: {r['faithfulness']}/5\n"
            f"- 回答: {r['answer'][:150]}...\n\n"
        )

    with open(os.path.join(exp_dir, "notes.md"), "w", encoding="utf-8") as f:
        f.write(notes)

    print(f"\n📁 實驗結果已儲存至: experiments/{version_name}/")


# ============================================================
# 6. 更新 Leaderboard
# ============================================================
def update_leaderboard():
    """掃描所有實驗目錄，重新生成 optimization_log.md。"""
    experiments = []

    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        result_path = os.path.join(EXPERIMENTS_DIR, name, "results.json")
        if os.path.isfile(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            experiments.append({
                "version": data["version"],
                "description": data["description"],
                "timestamp": data["timestamp"],
                "summary": data["summary"],
            })

    md = """# RAG 優化實驗記錄 — Leaderboard

> 自動生成，勿手動編輯。執行 `python3 eval_rag.py` 後自動更新。

## 總覽

| # | 版本 | 描述 | Hit Rate | MRR | 相似度 | 關鍵字 | 忠實度 | 日期 |
|---|------|------|:--------:|:---:|:------:|:------:|:------:|------|
"""
    for i, exp in enumerate(experiments, 1):
        s = exp["summary"]
        date = exp["timestamp"][:10]
        md += (
            f"| {i} | {exp['version']} | {exp['description']} "
            f"| {s['retrieval_hit_rate']}% | {s['retrieval_mrr']} "
            f"| {s['avg_similarity']}% | {s['keyword_coverage']}% "
            f"| {s['faithfulness']}/5 | {date} |\n"
        )

    md += f"""
## 指標說明

| 指標 | 說明 | 範圍 |
|------|------|------|
| **Hit Rate** | 檢索結果中包含預期來源的比例 | 0-100% (↑越好) |
| **MRR** | Mean Reciprocal Rank，預期來源排名的倒數平均 | 0-1 (↑越好) |
| **相似度** | 檢索片段的平均 cosine similarity | 0-100% (↑越好) |
| **關鍵字** | 回答中包含預期關鍵字的比例 | 0-100% (↑越好) |
| **忠實度** | LLM-as-Judge 評估回答是否忠實於參考資料 | 1-5 (↑越好) |

---
*最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"📊 Leaderboard 已更新: experiments/optimization_log.md")


# ============================================================
# 7. 主程式
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="RAG 評估腳本")
    parser.add_argument("--version", type=str, help="版本名稱（如 v0_baseline）")
    parser.add_argument("--desc", type=str, help="版本描述")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG 評估系統 — Level 0 實驗追蹤")
    print("=" * 60)

    # 取得版本名稱
    version = args.version
    if not version:
        version = input("\n📝 輸入版本名稱（如 v0_baseline）: ").strip()
    if not version:
        print("❌ 版本名稱不能為空")
        sys.exit(1)

    description = args.desc
    if not description:
        description = input("📝 輸入版本描述: ").strip() or version

    # 建立 RAG Pipeline
    vectorstore, llm, prompt, num_chunks = build_rag_pipeline()

    # 執行評估
    print(f"\n🔬 開始評估（版本: {version}）...")
    print("-" * 60)

    results = run_evaluation(vectorstore, llm, prompt)
    summary = aggregate_results(results)

    # 顯示彙總
    print("\n" + "=" * 60)
    print("  📊 評估結果彙總")
    print("=" * 60)
    print(f"  Retrieval Hit Rate : {summary['retrieval_hit_rate']}%")
    print(f"  Retrieval MRR      : {summary['retrieval_mrr']}")
    print(f"  平均相似度          : {summary['avg_similarity']}%")
    print(f"  Top-1 相似度        : {summary['avg_top1_similarity']}%")
    print(f"  關鍵字覆蓋率        : {summary['keyword_coverage']}%")
    print(f"  忠實度 (LLM Judge)  : {summary['faithfulness']}/5")
    print(f"  平均檢索耗時        : {summary['avg_retrieval_time']}s")
    print(f"  平均生成耗時        : {summary['avg_generation_time']}s")
    print("=" * 60)

    # 儲存
    save_experiment(version, description, results, summary, num_chunks)
    update_leaderboard()

    print("\n✅ 評估完成！")


if __name__ == "__main__":
    main()
