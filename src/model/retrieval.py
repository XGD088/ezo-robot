"""
检索相关代码
"""
from typing import Dict, Any

from langchain_community.vectorstores import Qdrant

from src.files.file_manager import load_documents, split_documents
from src.model.embeddings import TongyiEmbeddings
from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("retrieval")

def setup_vectorstore(config: Dict[str, Any]) -> Qdrant:
    """
    设置向量存储
    
    Args:
        config: 向量存储配置
        
    Returns:
        配置好的向量存储
    """
    retrieval_config = config.get("retrieval", {})
    
    # 加载文档并创建向量存储
    documents = load_documents(retrieval_config.get("docs_directory", "statics"))
    texts = split_documents(documents)
    embeddings = TongyiEmbeddings()  # 使用通义千问的 embeddings
    
    vectorstore = Qdrant.from_documents(
        documents=texts,
        embedding=embeddings,
        location=retrieval_config.get("location", ":memory:"),
        collection_name=retrieval_config.get("collection_name", "my_documents")
    )
    
    logger.info(f"向量存储已配置: {retrieval_config.get('collection_name', 'my_documents')}")
    return vectorstore 