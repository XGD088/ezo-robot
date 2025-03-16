from dotenv import load_dotenv
import os
from langsmith import Client
import logging


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_langsmith_connection():
    """测试 LangSmith 连接"""
    try:
        # 加载环境变量
        load_dotenv()
        
        # 检查环境变量
        api_key = os.getenv("LANGCHAIN_API_KEY")
        project = os.getenv("LANGCHAIN_PROJECT")
        
        if not api_key:
            raise ValueError("LANGCHAIN_API_KEY 未设置")
        if not project:
            raise ValueError("LANGCHAIN_PROJECT 未设置")
            
        logger.info(f"使用项目: {project}")
        logger.info(f"API密钥前缀: {api_key[:6]}...")
        logger.info("正在连接 LangSmith...")
        
        # 创建 LangSmith 客户端
        client = Client()
        
        try:
            # 尝试列出项目
            logger.info("尝试列出项目...")
            projects = client.list_projects()
            logger.info(f"找到的项目: {[p.name for p in projects]}")
            
            # 确保项目存在
            if project not in [p.name for p in projects]:
                logger.info(f"项目 {project} 不存在，正在创建...")
                client.create_project(project_name=project)
            
        except Exception as e:
            # 如果是409错误（会话已存在），视为成功
            if "409" in str(e) and "Session already exists" in str(e):
                logger.info("LangSmith连接测试成功！（会话已存在）")
                logger.info(f"请访问 https://smith.langchain.com/ 查看您的项目")
                return True
            else:
                raise e
        
        logger.info("LangSmith连接测试成功！")
        logger.info(f"请访问 https://smith.langchain.com/ 查看您的项目")
        return True
        
    except Exception as e:
        logger.error(f"LangSmith连接测试失败: {str(e)}")
        logger.error("请确保：")
        logger.error("1. API密钥格式正确（应该以 lsv2_ 开头）")
        logger.error("2. 项目名称不包含特殊字符")
        logger.error("3. 环境变量已正确设置")
        logger.error(f"错误类型: {type(e).__name__}")
        logger.error(f"错误详情: {str(e)}")
        return False

if __name__ == "__main__":
    test_langsmith_connection() 