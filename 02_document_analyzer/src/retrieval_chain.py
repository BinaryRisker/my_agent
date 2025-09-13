"""
检索链模块
实现各种检索策略和链式处理
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

# LangChain imports
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document, BaseRetriever
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Third-party imports
from loguru import logger
import json

# Project imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config


class RetrievalChainManager:
    """检索链管理器"""
    
    def __init__(self, 
                 retriever: BaseRetriever,
                 llm: BaseLanguageModel,
                 chain_type: str = "stuff"):
        """
        初始化检索链管理器
        
        Args:
            retriever: 检索器实例
            llm: 语言模型
            chain_type: 链类型 ("stuff", "map_reduce", "refine", "map_rerank")
        """
        self.config = get_config()
        self.retriever = retriever
        self.llm = llm
        self.chain_type = chain_type
        
        # 初始化各种链
        self.qa_chain = None
        self.conversational_chain = None
        self.compression_retriever = None
        
        # 对话记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # 自定义提示模板
        self.qa_prompt_template = self._create_qa_prompt()
        
        logger.info(f"检索链管理器初始化完成 - 链类型: {chain_type}")
    
    def create_qa_chain(self, **kwargs) -> RetrievalQA:
        """
        创建问答链
        
        Args:
            **kwargs: 链的额外参数
            
        Returns:
            问答链实例
        """
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=self.chain_type,
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": self.qa_prompt_template,
                    **kwargs
                }
            )
            
            logger.info("问答链创建成功")
            return self.qa_chain
            
        except Exception as e:
            logger.error(f"创建问答链失败: {e}")
            raise
    
    def create_conversational_chain(self, **kwargs) -> ConversationalRetrievalChain:
        """
        创建对话检索链
        
        Args:
            **kwargs: 链的额外参数
            
        Returns:
            对话检索链实例
        """
        try:
            self.conversational_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=self.config.debug,
                **kwargs
            )
            
            logger.info("对话检索链创建成功")
            return self.conversational_chain
            
        except Exception as e:
            logger.error(f"创建对话检索链失败: {e}")
            raise
    
    def create_multi_query_retriever(self) -> MultiQueryRetriever:
        """
        创建多查询检索器
        
        Returns:
            多查询检索器实例
        """
        try:
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=self.retriever,
                llm=self.llm
            )
            
            logger.info("多查询检索器创建成功")
            return multi_query_retriever
            
        except Exception as e:
            logger.error(f"创建多查询检索器失败: {e}")
            raise
    
    def create_compression_retriever(self) -> ContextualCompressionRetriever:
        """
        创建压缩检索器
        
        Returns:
            压缩检索器实例
        """
        try:
            # 创建LLM链提取器
            compressor = LLMChainExtractor.from_llm(self.llm)
            
            # 创建压缩检索器
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.retriever
            )
            
            logger.info("压缩检索器创建成功")
            return self.compression_retriever
            
        except Exception as e:
            logger.error(f"创建压缩检索器失败: {e}")
            raise
    
    def ask_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        使用问答链回答问题
        
        Args:
            question: 问题
            **kwargs: 额外参数
            
        Returns:
            包含答案和来源的字典
        """
        if self.qa_chain is None:
            self.create_qa_chain()
        
        try:
            logger.info(f"执行问答查询: {question[:50]}...")
            
            # 执行查询
            result = self.qa_chain({"query": question}, **kwargs)
            
            # 处理结果
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # 处理来源文档
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:500],  # 限制长度
                        "metadata": doc.metadata,
                        "relevance_score": getattr(doc, 'score', None)
                    }
                    response["source_documents"].append(source_info)
            
            logger.info("问答查询完成")
            return response
            
        except Exception as e:
            logger.error(f"问答查询失败: {e}")
            raise
    
    def conversational_ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        使用对话链回答问题
        
        Args:
            question: 问题
            **kwargs: 额外参数
            
        Returns:
            包含答案和历史的字典
        """
        if self.conversational_chain is None:
            self.create_conversational_chain()
        
        try:
            logger.info(f"执行对话查询: {question[:50]}...")
            
            # 执行查询
            result = self.conversational_chain({"question": question}, **kwargs)
            
            # 处理结果
            response = {
                "question": question,
                "answer": result["answer"],
                "source_documents": [],
                "chat_history": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # 处理来源文档
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:500],
                        "metadata": doc.metadata,
                        "relevance_score": getattr(doc, 'score', None)
                    }
                    response["source_documents"].append(source_info)
            
            # 处理对话历史
            if "chat_history" in result:
                response["chat_history"] = result["chat_history"]
            
            logger.info("对话查询完成")
            return response
            
        except Exception as e:
            logger.error(f"对话查询失败: {e}")
            raise
    
    def batch_ask(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        批量问答
        
        Args:
            questions: 问题列表
            
        Returns:
            答案列表
        """
        results = []
        
        for i, question in enumerate(questions):
            try:
                logger.info(f"处理问题 {i+1}/{len(questions)}: {question[:30]}...")
                result = self.ask_question(question)
                results.append(result)
            except Exception as e:
                logger.error(f"问题 {i+1} 处理失败: {e}")
                error_result = {
                    "question": question,
                    "answer": f"处理失败: {str(e)}",
                    "source_documents": [],
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_result)
        
        logger.info(f"批量问答完成，处理了 {len(questions)} 个问题")
        return results
    
    def get_relevant_documents(self, 
                              query: str, 
                              use_compression: bool = False,
                              k: int = 4) -> List[Document]:
        """
        获取相关文档
        
        Args:
            query: 查询文本
            use_compression: 是否使用压缩检索
            k: 返回文档数量
            
        Returns:
            相关文档列表
        """
        try:
            if use_compression:
                if self.compression_retriever is None:
                    self.create_compression_retriever()
                retriever = self.compression_retriever
            else:
                retriever = self.retriever
            
            logger.info(f"获取相关文档: {query[:50]}... (压缩: {use_compression})")
            
            # 设置检索参数
            if hasattr(retriever, 'search_kwargs'):
                retriever.search_kwargs['k'] = k
            
            # 执行检索
            documents = retriever.get_relevant_documents(query)
            
            logger.info(f"检索到 {len(documents)} 个相关文档")
            return documents
            
        except Exception as e:
            logger.error(f"获取相关文档失败: {e}")
            raise
    
    def evaluate_retrieval(self, 
                          test_queries: List[str],
                          ground_truth: Optional[List[List[str]]] = None) -> Dict[str, Any]:
        """
        评估检索性能
        
        Args:
            test_queries: 测试查询列表
            ground_truth: 真实相关文档ID列表
            
        Returns:
            评估结果
        """
        results = {
            "total_queries": len(test_queries),
            "retrieval_results": [],
            "average_retrieval_time": 0,
            "evaluation_time": datetime.now().isoformat()
        }
        
        total_time = 0
        
        for i, query in enumerate(test_queries):
            start_time = datetime.now()
            
            try:
                # 执行检索
                documents = self.get_relevant_documents(query)
                
                end_time = datetime.now()
                retrieval_time = (end_time - start_time).total_seconds()
                total_time += retrieval_time
                
                # 记录结果
                query_result = {
                    "query": query,
                    "retrieved_count": len(documents),
                    "retrieval_time": retrieval_time,
                    "documents": [
                        {
                            "content_preview": doc.page_content[:100],
                            "metadata": doc.metadata
                        } for doc in documents[:3]  # 只记录前3个
                    ]
                }
                
                # 如果有真实标签，计算准确率
                if ground_truth and i < len(ground_truth):
                    true_docs = ground_truth[i]
                    retrieved_ids = [doc.metadata.get('doc_id', '') for doc in documents]
                    
                    # 计算精确率和召回率
                    if true_docs and retrieved_ids:
                        true_positive = len(set(true_docs) & set(retrieved_ids))
                        precision = true_positive / len(retrieved_ids) if retrieved_ids else 0
                        recall = true_positive / len(true_docs) if true_docs else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        query_result.update({
                            "precision": precision,
                            "recall": recall,
                            "f1": f1
                        })
                
                results["retrieval_results"].append(query_result)
                
            except Exception as e:
                logger.error(f"评估查询 {i+1} 失败: {e}")
                results["retrieval_results"].append({
                    "query": query,
                    "error": str(e),
                    "retrieval_time": 0
                })
        
        # 计算平均检索时间
        results["average_retrieval_time"] = total_time / len(test_queries) if test_queries else 0
        
        # 如果有真实标签，计算整体指标
        if ground_truth:
            valid_results = [r for r in results["retrieval_results"] if "precision" in r]
            if valid_results:
                results["overall_precision"] = sum(r["precision"] for r in valid_results) / len(valid_results)
                results["overall_recall"] = sum(r["recall"] for r in valid_results) / len(valid_results)
                results["overall_f1"] = sum(r["f1"] for r in valid_results) / len(valid_results)
        
        logger.info(f"检索评估完成，平均检索时间: {results['average_retrieval_time']:.3f}秒")
        return results
    
    def _create_qa_prompt(self) -> PromptTemplate:
        """创建问答提示模板"""
        template = """基于以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请诚实地说"根据提供的信息无法回答这个问题"。

上下文信息:
{context}

问题: {question}

请提供准确、有用的回答，并在可能的情况下引用相关的上下文信息："""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear()
        logger.info("对话记忆已清空")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取记忆信息"""
        return {
            "memory_type": type(self.memory).__name__,
            "messages_count": len(self.memory.chat_memory.messages),
            "memory_variables": self.memory.memory_variables
        }
    
    def export_conversation(self, file_path: str):
        """导出对话历史"""
        try:
            conversation_data = {
                "memory_info": self.get_memory_info(),
                "messages": [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content,
                        "timestamp": datetime.now().isoformat()
                    } for msg in self.memory.chat_memory.messages
                ],
                "export_time": datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"对话历史已导出到: {file_path}")
            
        except Exception as e:
            logger.error(f"导出对话历史失败: {e}")
            raise


if __name__ == "__main__":
    # 测试检索链管理器
    from langchain.schema import Document
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.llms import OpenAI
    
    # 创建测试数据
    test_docs = [
        Document(
            page_content="人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
            metadata={"source": "ai_intro.txt", "topic": "AI"}
        ),
        Document(
            page_content="机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习。",
            metadata={"source": "ml_basics.txt", "topic": "ML"}
        ),
        Document(
            page_content="深度学习是机器学习的一个子集，它使用神经网络来模拟人脑的工作方式。",
            metadata={"source": "dl_overview.txt", "topic": "DL"}
        )
    ]
    
    try:
        print("测试检索链管理器...")
        
        # 由于需要API密钥，这里只进行基本的初始化测试
        print("检索链管理器模块加载成功")
        print("主要功能包括：")
        print("- 问答链")
        print("- 对话检索链") 
        print("- 多查询检索器")
        print("- 压缩检索器")
        print("- 批量问答")
        print("- 检索评估")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()