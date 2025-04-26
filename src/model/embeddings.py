"""
嵌入模型相关实现
"""
import os
from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("embeddings")

class TongyiEmbeddings(Embeddings):
    """通义千问的 embeddings 实现"""
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.batch_size = 10  # 通义千问 API 的批量限制

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多个文本转换为向量"""
        try:
            all_embeddings = []
            # 分批处理文本
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                response = self.client.embeddings.create(
                    model="text-embedding-v3",
                    input=batch_texts,
                    encoding_format="float"
                )
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        except Exception as e:
            logger.error(f"通义千问 embeddings 调用失败: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """将单个文本转换为向量"""
        return self.embed_documents([text])[0] 