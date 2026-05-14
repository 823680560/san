import faiss
import os
import glob
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from configs.config import (
    EMBEDDING_DIM, KB_LAW, LAW_JSON,
    LAW_PDF_DIR, LAW_CHUNK_SIZE, LAW_CHUNK_OVERLAP,
)
from server.rag.data_loader import DataLoader


def create_faiss_index(nodes, save_path):
    """
    直接接收 nodes (TextNodes) 列表进行索引构建
    """
    os.makedirs(save_path, exist_ok=True)
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM) # 创建空的 FAISS 索引。
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"开始对 {len(nodes)} 个文本片段进行向量化并构建索引...")
    
    # 使用 from_documents 也可以处理 nodes，或者直接使用 VectorStoreIndex(nodes, ...)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        # show_progress=True,
        insert_batch_size=len(nodes),
    )
    
    index.storage_context.persist(save_path)
    return index


def load_faiss_index(save_path):
    vector_store = FaissVectorStore.from_persist_dir(save_path)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=save_path)
    return load_index_from_storage(storage_context)


def build_index_from_file(file_path: str, save_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    根据文件扩展名自动选择解析方式，构建向量索引

    Args:
        file_path: 文件路径
        save_path: 索引保存路径
        chunk_size: 分片大小
        chunk_overlap: 分片重叠

    Returns:
        构建好的 VectorStoreIndex 实例
    """
    # 1. 加载文档
    documents = DataLoader.load_file(file_path)

    # 2. 分片
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"分片完成，共生成 {len(nodes)} 个文本片段")

    # 3. 构建索引
    index = create_faiss_index(nodes, save_path)
    print(f"索引构建完成，保存至: {save_path}")

    return index


def check_load_law_kb():
    print("检查加载法规向量库")
    index_file = os.path.join(KB_LAW, "default__vector_store.json")
    if not os.path.exists(index_file):
        documents = []

        # ① 加载 JSON 法规数据
        documents.extend(DataLoader.load_json_data(LAW_JSON))

        # ② 加载 PDF 法规文件
        pdf_files = glob.glob(os.path.join(LAW_PDF_DIR, "*.pdf"))
        for pdf_file in pdf_files:
            documents.extend(DataLoader.load_law_pdf_data(pdf_file))

        # ③ 加载其他格式文件（doc/docx/xlsx/xls/txt/csv）
        other_extensions = ("*.doc", "*.docx", "*.xlsx", "*.xls", "*.txt", "*.csv")
        for ext in other_extensions:
            for file_path in glob.glob(os.path.join(LAW_PDF_DIR, ext)):
                try:
                    documents.extend(DataLoader.load_file(file_path))
                    print(f"  已加载: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"  加载文件失败 {file_path}: {e}")

        print(f"共加载 {len(documents)} 个文档（JSON + PDF + 其他格式）")

        # ④ 分片
        splitter = SentenceSplitter(chunk_size=LAW_CHUNK_SIZE, chunk_overlap=LAW_CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"✅ 分片完成，共生成 {len(nodes)} 个文本片段")

        # ⑤ 预览
        print("\n🔍 分片片段详细信息预览 (前3个):")
        print("=" * 80)
        for i, node in enumerate(nodes[:3]):
            print(f"\n--- 片段 {i + 1} ---")
            print(f"节点ID: {node.node_id}")
            print(f"元数据: {node.metadata}")
            print(f"内容预览:\n{node.text[:500]}...")
        print("=" * 80)

        # ⑥ 构建索引
        index = create_faiss_index(nodes, KB_LAW)
        print("\n✅ 索引构建并保存完成")

    else:
        print("从磁盘加载法规向量库索引...")
        index = load_faiss_index(KB_LAW)
        print("✅ 法规索引加载完成")

    return index
