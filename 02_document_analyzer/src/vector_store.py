"""
向量存储管理器
支持多种向量数据库和嵌入模型
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import uuid

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# Third-party imports
from loguru import logger
import numpy as np

# Project imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config


class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(self, 
                 store_type: str = "chroma",
                 embedding_model: Optional[Embeddings] = None,
                 persist_directory: Optional[str] = None):
        """
        初始化向量存储管理器
        
        Args:
            store_type: 向量存储类型 ("chroma", "faiss")
            embedding_model: 嵌入模型
            persist_directory: 持久化目录
        """
        self.config = get_config()
        self.store_type = store_type.lower()
        
        # 设置嵌入模型
        if embedding_model is None:
            self.embedding_model = OpenAIEmbeddings(
                openai_api_key=self.config.llm.api_key,
                model="text-embedding-ada-002"
            )
        else:
            self.embedding_model = embedding_model
        
        # 设置持久化目录
        if persist_directory is None:
            self.persist_directory = Path("embeddings") / self.store_type
        else:
            self.persist_directory = Path(persist_directory)
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 向量存储实例
        self.vector_store = None
        self.document_metadata = {}  # 存储文档元数据
        
        # 支持的存储类型
        self.supported_stores = {
            "chroma": self._create_chroma_store,
            "faiss": self._create_faiss_store
        }
        
        if self.store_type not in self.supported_stores:
            raise ValueError(f"不支持的向量存储类型: {store_type}")
        
        logger.info(f"向量存储管理器初始化完成 - 类型: {store_type}")
    
    def create_store(self, documents: List[Document], collection_name: str = "default") -> None:
        """
        创建向量存储
        
        Args:
            documents: 文档列表
            collection_name: 集合名称
        """
        if not documents:
            raise ValueError("文档列表不能为空")
        
        try:
            logger.info(f"开始创建向量存储，文档数量: {len(documents)}")
            
            # 根据存储类型创建向量存储
            create_func = self.supported_stores[self.store_type]
            self.vector_store = create_func(documents, collection_name)
            
            # 保存文档元数据
            self._save_document_metadata(documents, collection_name)
            
            logger.info(f"向量存储创建完成 - 集合: {collection_name}")
            
        except Exception as e:
            logger.error(f"创建向量存储失败: {e}")
            raise
    
    def load_store(self, collection_name: str = "default") -> bool:
        """
        加载现有的向量存储
        
        Args:
            collection_name: 集合名称
            
        Returns:
            是否加载成功
        """
        try:
            if self.store_type == "chroma":
                store_path = self.persist_directory / collection_name
                if not store_path.exists():
                    logger.warning(f"Chroma存储不存在: {store_path}")
                    return False
                
                self.vector_store = Chroma(
                    persist_directory=str(store_path),
                    embedding_function=self.embedding_model,
                    collection_name=collection_name
                )
                
            elif self.store_type == "faiss":
                index_path = self.persist_directory / f"{collection_name}.faiss"
                pkl_path = self.persist_directory / f"{collection_name}.pkl"
                
                if not (index_path.exists() and pkl_path.exists()):
                    logger.warning(f"FAISS存储不存在: {index_path}")
                    return False
                
                self.vector_store = FAISS.load_local(
                    str(self.persist_directory),
                    self.embedding_model,
                    index_name=collection_name
                )
            
            # 加载文档元数据
            self._load_document_metadata(collection_name)
            
            logger.info(f"向量存储加载成功 - 集合: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到现有存储
        
        Args:
            documents: 要添加的文档列表
            
        Returns:
            文档ID列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先创建或加载存储")
        
        try:
            logger.info(f"添加 {len(documents)} 个文档到向量存储")
            
            # 生成文档ID
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            
            # 为文档添加ID元数据
            for doc, doc_id in zip(documents, doc_ids):
                doc.metadata['doc_id'] = doc_id
                doc.metadata['added_time'] = datetime.now().isoformat()
            
            # 添加到向量存储
            if self.store_type == "chroma":
                self.vector_store.add_documents(documents, ids=doc_ids)
            else:  # FAISS
                self.vector_store.add_documents(documents)
            
            # 保存向量存储
            self.save_store()
            
            logger.info(f"成功添加 {len(documents)} 个文档")
            return doc_ids
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 4,
                         filter: Optional[Dict[str, Any]] = None,
                         fetch_k: int = 20) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 过滤条件
            fetch_k: 获取候选数量（仅FAISS）
            
        Returns:
            相似文档列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        try:
            logger.info(f"执行相似度搜索: '{query[:50]}...' (k={k})")
            
            if self.store_type == "chroma":
                if filter:
                    results = self.vector_store.similarity_search(query, k=k, filter=filter)
                else:
                    results = self.vector_store.similarity_search(query, k=k)
            else:  # FAISS
                results = self.vector_store.similarity_search(query, k=k, fetch_k=fetch_k)
            
            logger.info(f"搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            raise
    
    def similarity_search_with_score(self,
                                   query: str,
                                   k: int = 4,
                                   filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        带评分的相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 过滤条件
            
        Returns:
            (文档, 相似度分数) 列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        try:
            logger.info(f"执行带评分的相似度搜索: '{query[:50]}...' (k={k})")
            
            if self.store_type == "chroma":
                if filter:
                    results = self.vector_store.similarity_search_with_score(query, k=k, filter=filter)
                else:
                    results = self.vector_store.similarity_search_with_score(query, k=k)
            else:  # FAISS
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(f"搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"带评分的相似度搜索失败: {e}")
            raise
    
    def max_marginal_relevance_search(self,
                                     query: str,
                                     k: int = 4,
                                     fetch_k: int = 20,
                                     lambda_mult: float = 0.5) -> List[Document]:
        """
        最大边际相关性搜索（减少重复结果）
        
        Args:
            query: 查询文本
            k: 返回结果数量
            fetch_k: 获取候选数量
            lambda_mult: 多样性参数 (0-1)
            
        Returns:
            文档列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        try:
            logger.info(f"执行MMR搜索: '{query[:50]}...' (k={k}, fetch_k={fetch_k})")
            
            results = self.vector_store.max_marginal_relevance_search(
                query, 
                k=k, 
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            
            logger.info(f"MMR搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"MMR搜索失败: {e}")
            raise
    
    def save_store(self, collection_name: str = "default") -> None:
        """
        保存向量存储
        
        Args:
            collection_name: 集合名称
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        try:
            if self.store_type == "chroma":
                self.vector_store.persist()
            elif self.store_type == "faiss":
                self.vector_store.save_local(
                    str(self.persist_directory),
                    index_name=collection_name
                )
            
            logger.info(f"向量存储已保存 - 集合: {collection_name}")
            
        except Exception as e:
            logger.error(f"保存向量存储失败: {e}")
            raise
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        删除文档
        
        Args:
            doc_ids: 文档ID列表
            
        Returns:
            是否删除成功
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        try:
            if self.store_type == "chroma":
                self.vector_store.delete(ids=doc_ids)
                logger.info(f"从Chroma删除 {len(doc_ids)} 个文档")
                return True
            else:
                logger.warning("FAISS不支持删除文档操作")
                return False
                
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def get_store_info(self) -> Dict[str, Any]:
        """
        获取存储信息
        
        Returns:
            存储信息字典
        """
        info = {
            'store_type': self.store_type,
            'persist_directory': str(self.persist_directory),
            'embedding_model': type(self.embedding_model).__name__,
            'is_initialized': self.vector_store is not None,
            'collections': self._list_collections()
        }
        
        if self.vector_store is not None:
            try:
                if self.store_type == "chroma":
                    info['document_count'] = self.vector_store._collection.count()
                else:  # FAISS
                    info['document_count'] = self.vector_store.index.ntotal
            except:
                info['document_count'] = "未知"
        
        return info
    
    def _create_chroma_store(self, documents: List[Document], collection_name: str) -> Chroma:
        """创建Chroma向量存储"""
        store_path = self.persist_directory / collection_name
        store_path.mkdir(parents=True, exist_ok=True)
        
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=str(store_path),
            collection_name=collection_name
        )
    
    def _create_faiss_store(self, documents: List[Document], collection_name: str) -> FAISS:
        """创建FAISS向量存储"""
        return FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
    
    def _save_document_metadata(self, documents: List[Document], collection_name: str) -> None:
        """保存文档元数据"""
        metadata_file = self.persist_directory / f"{collection_name}_metadata.pkl"
        
        metadata = {
            'collection_name': collection_name,
            'document_count': len(documents),
            'created_time': datetime.now().isoformat(),
            'documents_info': []
        }
        
        for i, doc in enumerate(documents):
            doc_info = {
                'index': i,
                'content_length': len(doc.page_content),
                'metadata': doc.metadata.copy()
            }
            metadata['documents_info'].append(doc_info)
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.debug(f"文档元数据已保存: {metadata_file}")
    
    def _load_document_metadata(self, collection_name: str) -> None:
        """加载文档元数据"""
        metadata_file = self.persist_directory / f"{collection_name}_metadata.pkl"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    self.document_metadata = pickle.load(f)
                logger.debug(f"文档元数据已加载: {metadata_file}")
            except Exception as e:
                logger.warning(f"加载文档元数据失败: {e}")
                self.document_metadata = {}
    
    def _list_collections(self) -> List[str]:
        """列出所有集合"""
        collections = []
        
        if self.store_type == "chroma":
            for item in self.persist_directory.iterdir():
                if item.is_dir():
                    collections.append(item.name)
        else:  # FAISS
            for item in self.persist_directory.glob("*.faiss"):
                collections.append(item.stem)
        
        return collections
    
    def clear_store(self, collection_name: str = "default") -> bool:
        """
        清空向量存储
        
        Args:
            collection_name: 集合名称
            
        Returns:
            是否成功
        """
        try:
            if self.store_type == "chroma":
                store_path = self.persist_directory / collection_name
                if store_path.exists():
                    import shutil
                    shutil.rmtree(store_path)
                    logger.info(f"已清空Chroma存储: {collection_name}")
            else:  # FAISS
                index_path = self.persist_directory / f"{collection_name}.faiss"
                pkl_path = self.persist_directory / f"{collection_name}.pkl"
                
                if index_path.exists():
                    index_path.unlink()
                if pkl_path.exists():
                    pkl_path.unlink()
                logger.info(f"已清空FAISS存储: {collection_name}")
            
            # 清空元数据
            metadata_file = self.persist_directory / f"{collection_name}_metadata.pkl"
            if metadata_file.exists():
                metadata_file.unlink()
            
            self.vector_store = None
            self.document_metadata = {}
            
            return True
            
        except Exception as e:
            logger.error(f"清空向量存储失败: {e}")
            return False


if __name__ == "__main__":
    # 测试向量存储管理器
    from langchain.schema import Document
    
    # 创建测试文档
    test_docs = [
        Document(
            page_content="这是关于人工智能的文档。人工智能是计算机科学的一个分支。",
            metadata={"source": "ai_doc.txt", "type": "技术"}
        ),
        Document(
            page_content="机器学习是人工智能的一个重要子领域。它通过算法让计算机学习。",
            metadata={"source": "ml_doc.txt", "type": "技术"}
        ),
        Document(
            page_content="深度学习使用神经网络来解决复杂问题。它在图像识别方面表现出色。",
            metadata={"source": "dl_doc.txt", "type": "技术"}
        )
    ]
    
    try:
        # 测试Chroma存储
        print("测试Chroma向量存储...")
        chroma_manager = VectorStoreManager(store_type="chroma")
        
        # 创建存储
        chroma_manager.create_store(test_docs, "test_collection")
        
        # 搜索测试
        results = chroma_manager.similarity_search("人工智能", k=2)
        print(f"搜索结果数量: {len(results)}")
        for i, doc in enumerate(results):
            print(f"结果 {i+1}: {doc.page_content[:50]}...")
        
        # 获取存储信息
        info = chroma_manager.get_store_info()
        print("存储信息:", info)
        
        # 清理测试数据
        chroma_manager.clear_store("test_collection")
        print("测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()