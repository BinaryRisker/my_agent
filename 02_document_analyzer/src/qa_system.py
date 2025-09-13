"""
问答系统模块
整合检索和生成功能的高级接口
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Third-party imports
from loguru import logger

# Project imports  
from retrieval_chain import RetrievalChainManager
from vector_store import VectorStoreManager


class QASystem:
    """问答系统主类"""
    
    def __init__(self, 
                 retrieval_chain_manager: RetrievalChainManager,
                 vector_store_manager: VectorStoreManager):
        """
        初始化问答系统
        
        Args:
            retrieval_chain_manager: 检索链管理器
            vector_store_manager: 向量存储管理器
        """
        self.retrieval_manager = retrieval_chain_manager
        self.vector_manager = vector_store_manager
        
        # 问答历史
        self.qa_history = []
        
        logger.info("问答系统初始化完成")
    
    def ask_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        询问问题（单次问答）
        
        Args:
            question: 问题
            **kwargs: 额外参数
            
        Returns:
            答案字典
        """
        try:
            logger.info(f"处理单次问答: {question[:50]}...")
            
            # 使用检索链管理器处理问题
            result = self.retrieval_manager.ask_question(question, **kwargs)
            
            # 记录到历史
            self.qa_history.append(result)
            
            # 限制历史记录长度
            if len(self.qa_history) > 100:
                self.qa_history = self.qa_history[-100:]
            
            logger.info("单次问答完成")
            return result
            
        except Exception as e:
            logger.error(f"单次问答失败: {e}")
            raise
    
    def conversational_ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        对话式问答（保持上下文）
        
        Args:
            question: 问题
            **kwargs: 额外参数
            
        Returns:
            答案字典
        """
        try:
            logger.info(f"处理对话问答: {question[:50]}...")
            
            # 使用对话检索链处理问题
            result = self.retrieval_manager.conversational_ask(question, **kwargs)
            
            # 记录到历史
            self.qa_history.append(result)
            
            # 限制历史记录长度
            if len(self.qa_history) > 100:
                self.qa_history = self.qa_history[-100:]
            
            logger.info("对话问答完成")
            return result
            
        except Exception as e:
            logger.error(f"对话问答失败: {e}")
            raise
    
    def batch_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        批量问答
        
        Args:
            questions: 问题列表
            
        Returns:
            答案列表
        """
        try:
            logger.info(f"批量处理 {len(questions)} 个问题")
            
            results = self.retrieval_manager.batch_ask(questions)
            
            # 添加到历史记录
            self.qa_history.extend(results)
            
            # 限制历史记录长度
            if len(self.qa_history) > 100:
                self.qa_history = self.qa_history[-100:]
            
            logger.info("批量问答完成")
            return results
            
        except Exception as e:
            logger.error(f"批量问答失败: {e}")
            raise
    
    def search_documents(self, 
                        query: str,
                        k: int = 4,
                        use_mmr: bool = False,
                        **kwargs) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 搜索查询
            k: 返回文档数量
            use_mmr: 是否使用MMR搜索
            **kwargs: 额外参数
            
        Returns:
            文档列表
        """
        try:
            logger.info(f"搜索文档: {query[:50]}... (k={k}, MMR={use_mmr})")
            
            if use_mmr:
                documents = self.vector_manager.max_marginal_relevance_search(
                    query, k=k, **kwargs
                )
            else:
                documents = self.vector_manager.similarity_search(
                    query, k=k, **kwargs
                )
            
            # 格式化返回结果
            results = []
            for i, doc in enumerate(documents):
                doc_info = {
                    "index": i,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, 'score', None)
                }
                results.append(doc_info)
            
            logger.info(f"文档搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"文档搜索失败: {e}")
            raise
    
    def get_qa_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取问答历史
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            历史记录列表
        """
        if limit:
            return self.qa_history[-limit:]
        return self.qa_history.copy()
    
    def clear_history(self):
        """清空问答历史"""
        self.qa_history.clear()
        self.retrieval_manager.clear_memory()
        logger.info("问答历史已清空")
    
    def export_history(self, file_path: str, format: str = "json"):
        """
        导出问答历史
        
        Args:
            file_path: 文件路径
            format: 导出格式 ("json", "txt")
        """
        try:
            if format.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.qa_history, f, ensure_ascii=False, indent=2)
            
            elif format.lower() == "txt":
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("# 问答历史记录\n\n")
                    for i, qa in enumerate(self.qa_history):
                        f.write(f"## 问答 {i+1}\n")
                        f.write(f"**时间**: {qa.get('timestamp', 'N/A')}\n")
                        f.write(f"**问题**: {qa.get('question', 'N/A')}\n")
                        f.write(f"**答案**: {qa.get('answer', 'N/A')}\n")
                        
                        if qa.get('source_documents'):
                            f.write("**参考来源**:\n")
                            for j, doc in enumerate(qa['source_documents']):
                                f.write(f"  {j+1}. {doc.get('content', '')[:100]}...\n")
                        
                        f.write("\n" + "-"*50 + "\n\n")
            
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"问答历史已导出到: {file_path}")
            
        except Exception as e:
            logger.error(f"导出问答历史失败: {e}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "total_questions": len(self.qa_history),
            "recent_questions": len([qa for qa in self.qa_history 
                                   if datetime.fromisoformat(qa.get('timestamp', '1970-01-01T00:00:00'))
                                   > datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)]),
            "vector_store_info": self.vector_manager.get_store_info(),
            "memory_info": self.retrieval_manager.get_memory_info(),
            "last_update": datetime.now().isoformat()
        }


if __name__ == "__main__":
    print("QA系统模块加载成功")
    print("主要功能:")
    print("- 单次问答")
    print("- 对话式问答") 
    print("- 批量问答")
    print("- 文档搜索")
    print("- 历史记录管理")