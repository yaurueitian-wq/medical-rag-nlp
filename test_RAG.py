"""
============================================================
RAG (Retrieval-Augmented Generation) 實作教學
============================================================
使用技術：
  - Ollama (本地 LLM: llama3.2:3b)
  - ChromaDB (向量資料庫)
  - LangChain (RAG 框架)
  - nomic-embed-text (Embedding 模型)

RAG 架構流程：
  使用者問題 → Embedding → ChromaDB 檢索相似文件 → 組合 Prompt → Ollama LLM 回答
============================================================
"""

# ============================================================
# Step 1: 匯入所需套件
# ============================================================
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# Step 2: 準備示範醫療資料
# ============================================================
# 在真實場景中，這些資料會從 PDF、資料庫或電子病歷系統載入
# 這裡我們用幾段醫療相關的示範文本

medical_documents = [
    Document(
        page_content="""
        糖尿病（Diabetes Mellitus）是一種慢性代謝疾病，主要特徵是血糖長期偏高。
        糖尿病主要分為三種類型：
        1. 第一型糖尿病：自體免疫疾病，胰臟的β細胞被破壞，無法產生胰島素。通常在兒童或青少年時期發病。
        2. 第二型糖尿病：最常見的類型，佔所有糖尿病患者約90-95%。身體產生胰島素抗性，或胰臟無法產生足夠的胰島素。
        3. 妊娠糖尿病：在懷孕期間首次出現的高血糖狀態。
        糖尿病的診斷標準包括：空腹血糖 ≥ 126 mg/dL、口服葡萄糖耐量試驗（OGTT）2小時後血糖 ≥ 200 mg/dL、
        糖化血色素（HbA1c）≥ 6.5%。
        """,
        metadata={"source": "內科學教科書", "topic": "糖尿病概論"}
    ),
    Document(
        page_content="""
        第二型糖尿病的治療策略採取階梯式治療方針：
        第一線藥物：Metformin（二甲雙胍）是首選藥物，可降低肝臟葡萄糖輸出，改善胰島素敏感性。
        常見副作用包括腸胃不適、腹瀉。禁忌症包括嚴重腎功能不全（eGFR < 30 mL/min）。
        第二線藥物選擇包括：
        - SGLT2 抑制劑（如 Empagliflozin）：可降低心血管風險，有腎臟保護作用
        - GLP-1 受體促效劑（如 Liraglutide）：可減重，降低心血管事件
        - DPP-4 抑制劑（如 Sitagliptin）：副作用較少，但效果相對溫和
        - Sulfonylureas（如 Glimepiride）：價格便宜但有低血糖風險
        當口服藥物無法控制時，需考慮加入胰島素治療。
        """,
        metadata={"source": "臨床藥理學", "topic": "糖尿病藥物治療"}
    ),
    Document(
        page_content="""
        高血壓（Hypertension）的定義：根據2017年ACC/AHA指引，血壓 ≥ 130/80 mmHg 即為高血壓。
        高血壓的分級：
        - 正常血壓：< 120/80 mmHg
        - 血壓偏高：120-129 / < 80 mmHg
        - 第一期高血壓：130-139 / 80-89 mmHg
        - 第二期高血壓：≥ 140/90 mmHg
        - 高血壓危象：> 180/120 mmHg
        高血壓是心血管疾病、中風、慢性腎病的重要危險因子。
        非藥物治療包括：減鈉飲食（每日鈉攝取 < 2300mg）、規律運動（每週150分鐘中等強度運動）、
        維持健康體重（BMI 18.5-24.9）、限制酒精攝取、戒菸。
        """,
        metadata={"source": "心臟內科學", "topic": "高血壓"}
    ),
    Document(
        page_content="""
        高血壓的藥物治療選擇：
        1. ACE抑制劑（如 Enalapril, Lisinopril）：抑制血管收縮素轉化酶，適用於合併糖尿病腎病變的患者。
           副作用：乾咳（10-15%患者）、血管性水腫（罕見但嚴重）。
        2. ARB（如 Losartan, Valsartan）：阻斷血管收縮素II受體，不會引起乾咳，適合無法耐受ACE抑制劑的患者。
        3. CCB（如 Amlodipine）：阻斷鈣離子通道，擴張血管。副作用：下肢水腫、頭痛。
        4. 利尿劑（如 Hydrochlorothiazide）：增加腎臟排鈉排水，降低血容量。副作用：低血鉀、高尿酸。
        5. β阻斷劑（如 Bisoprolol）：降低心率和心輸出量。適用於合併心衰竭或心房顫動的患者。
        合併糖尿病的高血壓患者，首選ACE抑制劑或ARB，因為具有腎臟保護作用。
        """,
        metadata={"source": "臨床藥理學", "topic": "高血壓藥物治療"}
    ),
    Document(
        page_content="""
        急性心肌梗塞（Acute Myocardial Infarction, AMI）是冠狀動脈突然阻塞導致心肌缺血壞死的急症。
        典型症狀包括：胸痛（壓迫感、緊縮感，持續超過20分鐘）、可放射至左肩、左臂、下顎。
        伴隨症狀：冒冷汗、噁心嘔吐、呼吸困難。
        非典型表現（常見於女性、糖尿病、老年患者）：上腹痛、疲倦、暈厥。
        診斷依據（三項中符合兩項即可診斷）：
        1. 典型胸痛症狀
        2. 心電圖變化（ST段上升、新出現的左束支傳導阻滯）
        3. 心肌酵素升高（Troponin I/T升高）
        急性處置原則（MONA）：Morphine（嗎啡止痛）、Oxygen（氧氣）、Nitroglycerin（硝化甘油）、Aspirin（阿斯匹靈）。
        再灌流治療：首選經皮冠狀動脈介入術（PCI），門到氣球時間（door-to-balloon time）應 < 90分鐘。
        """,
        metadata={"source": "急診醫學", "topic": "急性心肌梗塞"}
    ),
    Document(
        page_content="""
        慢性腎臟病（Chronic Kidney Disease, CKD）依腎絲球過濾率（GFR）分為五期：
        - 第一期：GFR ≥ 90，腎功能正常但有腎臟損傷（如蛋白尿）
        - 第二期：GFR 60-89，輕度腎功能下降
        - 第三期A：GFR 45-59，中度腎功能下降
        - 第三期B：GFR 30-44，中重度腎功能下降
        - 第四期：GFR 15-29，重度腎功能下降
        - 第五期：GFR < 15，末期腎臟病（需透析或移植）
        CKD的主要病因：糖尿病腎病變（最常見）、高血壓腎病變、慢性腎絲球腎炎。
        治療原則：控制血糖（HbA1c < 7%）、控制血壓（目標 < 130/80 mmHg）、使用ACEI/ARB保護腎臟、
        限制蛋白質攝取（0.6-0.8 g/kg/day）、避免腎毒性藥物（如NSAIDs）。
        """,
        metadata={"source": "腎臟內科學", "topic": "慢性腎臟病"}
    ),
]

print(f"✅ Step 2 完成：已準備 {len(medical_documents)} 份醫療文件")
for i, doc in enumerate(medical_documents):
    print(f"   文件 {i+1}: {doc.metadata['topic']} (來源: {doc.metadata['source']})")

# ============================================================
# Step 3: 文本分割 (Text Splitting)
# ============================================================
# 為什麼要分割？
# - LLM 有 context window 限制
# - 較短的文本片段能提高檢索精確度
# - 每個 chunk 應該包含完整的語義單位

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # 每個片段最大 300 字元
    chunk_overlap=50,      # 相鄰片段重疊 50 字元，避免語義被截斷
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 分割優先順序
)

# 對所有文件進行分割
splits = text_splitter.split_documents(medical_documents)

print(f"\n✅ Step 3 完成：文本分割")
print(f"   原始文件數: {len(medical_documents)}")
print(f"   分割後片段數: {len(splits)}")
print(f"   範例片段 (第1個):")
print(f"   ---")
print(f"   {splits[0].page_content[:150]}...")
print(f"   ---")

# ============================================================
# Step 4: 建立向量資料庫 (Embedding + ChromaDB)
# ============================================================
# Embedding 模型將文字轉換為向量（數字陣列），
# 語義相近的文字在向量空間中距離較近

print(f"\n⏳ Step 4: 正在建立向量資料庫（使用 nomic-embed-text 模型）...")

# 初始化 Embedding 模型（使用 Ollama 本地模型）
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 將文本片段存入 ChromaDB
# ChromaDB 會自動：1) 計算每個片段的 embedding 向量  2) 建立向量索引
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="medical_knowledge",  # 集合名稱
    persist_directory="./chroma_db",       # 資料庫儲存路徑
)

print(f"✅ Step 4 完成：向量資料庫已建立")
print(f"   儲存位置: ./chroma_db")
print(f"   集合名稱: medical_knowledge")
print(f"   向量數量: {vectorstore._collection.count()}")

# ============================================================
# Step 5: 建立檢索器 (Retriever)
# ============================================================
# Retriever 負責根據使用者問題，從向量資料庫中找出最相關的文本片段

retriever = vectorstore.as_retriever(
    search_type="similarity",  # 使用餘弦相似度搜尋
    search_kwargs={"k": 3},    # 返回最相關的 3 個片段
)

# 測試檢索功能
print(f"\n✅ Step 5 完成：檢索器已建立")
print(f"   搜尋方式: 向量相似度 (cosine similarity)")
print(f"   返回片段數: 3")

test_query = "糖尿病的治療藥物有哪些？"
print(f"\n📝 檢索測試 - 問題: '{test_query}'")
test_results = retriever.invoke(test_query)
for i, doc in enumerate(test_results):
    print(f"   結果 {i+1} [來源: {doc.metadata.get('topic', 'N/A')}]:")
    print(f"   {doc.page_content[:100]}...")
    print()

# ============================================================
# Step 6: 建立 RAG Pipeline (Retrieval + Generation)
# ============================================================
# RAG 的核心：將檢索到的相關文件作為 context，與使用者問題一起送給 LLM

# 6.1 初始化 LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3,  # 較低的 temperature 讓回答更精確
)

# 6.2 定義 Prompt Template
# 這是 RAG 的關鍵 — 告訴 LLM 如何使用檢索到的資料
prompt = ChatPromptTemplate.from_template("""
你是一位專業的醫療助理。請根據以下提供的參考資料來回答問題。
如果參考資料中沒有相關資訊，請誠實告知你無法從現有資料中找到答案。
請用繁體中文回答。

參考資料：
{context}

問題：{question}

回答：
""")


# 6.3 輔助函數：將檢索到的文件格式化為字串
def format_docs(docs):
    """將檢索到的文件列表格式化為單一字串"""
    return "\n\n---\n\n".join(
        f"[來源: {doc.metadata.get('source', '未知')} - {doc.metadata.get('topic', '未知')}]\n{doc.page_content}"
        for doc in docs
    )


# 6.4 組裝 RAG Chain
# 使用 LangChain 的 LCEL (LangChain Expression Language) 串接各元件
rag_chain = (
    {
        "context": retriever | format_docs,    # 檢索 → 格式化
        "question": RunnablePassthrough(),      # 直接傳遞使用者問題
    }
    | prompt      # 填入 Prompt Template
    | llm         # 送給 LLM 生成回答
    | StrOutputParser()  # 解析輸出為字串
)

print(f"\n✅ Step 6 完成：RAG Pipeline 已組裝完成")
print(f"   LLM 模型: llama3.2:3b")
print(f"   Pipeline: 使用者問題 → 檢索器 → Prompt → LLM → 回答")

# ============================================================
# Step 7: 測試 RAG 系統
# ============================================================
print("\n" + "=" * 60)
print("🏥 醫療知識 RAG 系統 - 測試問答")
print("=" * 60)

# 準備測試問題
test_questions = [
    "糖尿病有哪些類型？診斷標準是什麼？",
    "高血壓患者合併糖尿病，應該首選什麼降壓藥？為什麼？",
    "急性心肌梗塞的典型症狀和急性處置原則是什麼？",
]

for i, question in enumerate(test_questions):
    print(f"\n{'─' * 50}")
    print(f"❓ 問題 {i+1}: {question}")
    print(f"{'─' * 50}")

    # 呼叫 RAG Chain
    answer = rag_chain.invoke(question)
    print(f"💡 回答:\n{answer}")

# ============================================================
# Step 8: 互動模式（可選）
# ============================================================
print("\n" + "=" * 60)
print("🔄 進入互動模式（輸入 'quit' 結束）")
print("=" * 60)

while True:
    user_input = input("\n❓ 請輸入你的問題: ").strip()
    if user_input.lower() in ["quit", "exit", "q", "結束"]:
        print("👋 感謝使用！再見！")
        break
    if not user_input:
        continue

    print("⏳ 正在思考中...")
    answer = rag_chain.invoke(user_input)
    print(f"\n💡 回答:\n{answer}")
