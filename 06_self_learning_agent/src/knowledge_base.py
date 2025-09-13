"""
自学习Agent - 知识库管理模块
实现知识的存储、检索、更新和管理功能，支持多种知识表示形式
"""

import json
import sqlite3
import hashlib
import time
import pickle
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss  # 用于高效向量相似度搜索

from loguru import logger


class KnowledgeType(Enum):
    """知识类型枚举"""
    FACTUAL = "factual"           # 事实性知识
    PROCEDURAL = "procedural"     # 过程性知识
    EXPERIENTIAL = "experiential" # 经验性知识
    RULE = "rule"                 # 规则知识
    PATTERN = "pattern"           # 模式知识
    DOMAIN = "domain"             # 领域知识


class KnowledgeSource(Enum):
    """知识来源枚举"""
    USER_INPUT = "user_input"       # 用户输入
    SYSTEM_LEARNING = "system_learning"  # 系统学习
    EXTERNAL_DATA = "external_data"      # 外部数据
    AGENT_EXPERIENCE = "agent_experience"  # Agent经验
    MODEL_TRAINING = "model_training"    # 模型训练


@dataclass
class KnowledgeItem:
    """知识项数据结构"""
    id: str
    title: str
    content: str
    knowledge_type: KnowledgeType
    source: KnowledgeSource
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    usage_count: int = 0
    last_used: Optional[str] = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class KnowledgeQuery:
    """知识查询结构"""
    query_text: str
    knowledge_types: List[KnowledgeType] = None
    tags: List[str] = None
    source_filter: List[KnowledgeSource] = None
    confidence_threshold: float = 0.0
    max_results: int = 10
    similarity_threshold: float = 0.3


@dataclass
class KnowledgeSearchResult:
    """知识搜索结果"""
    items: List[KnowledgeItem]
    similarities: List[float]
    total_count: int
    query_time: float


class KnowledgeDatabase:
    """知识数据库管理器"""
    
    def __init__(self, db_path: str):
        """
        初始化知识数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"知识数据库初始化完成: {self.db_path}")
    
    def _init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建知识表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    knowledge_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    tags TEXT,
                    metadata TEXT,
                    embedding BLOB,
                    confidence REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_items(knowledge_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON knowledge_items(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON knowledge_items(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON knowledge_items(confidence)")
            
            conn.commit()
    
    def add_knowledge(self, knowledge_item: KnowledgeItem) -> bool:
        """
        添加知识项
        
        Args:
            knowledge_item: 知识项
            
        Returns:
            添加结果
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 序列化复杂字段
                tags_json = json.dumps(knowledge_item.tags, ensure_ascii=False)
                metadata_json = json.dumps(knowledge_item.metadata, ensure_ascii=False)
                embedding_blob = pickle.dumps(knowledge_item.embedding) if knowledge_item.embedding is not None else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO knowledge_items 
                    (id, title, content, knowledge_type, source, tags, metadata, 
                     embedding, confidence, usage_count, last_used, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    knowledge_item.id,
                    knowledge_item.title,
                    knowledge_item.content,
                    knowledge_item.knowledge_type.value,
                    knowledge_item.source.value,
                    tags_json,
                    metadata_json,
                    embedding_blob,
                    knowledge_item.confidence,
                    knowledge_item.usage_count,
                    knowledge_item.last_used,
                    knowledge_item.created_at,
                    knowledge_item.updated_at
                ))
                
                conn.commit()
                logger.info(f"知识项添加成功: {knowledge_item.id}")
                return True
                
        except Exception as e:
            logger.error(f"添加知识项失败: {e}")
            return False
    
    def get_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """
        获取知识项
        
        Args:
            knowledge_id: 知识项ID
            
        Returns:
            知识项或None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM knowledge_items WHERE id = ?", (knowledge_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_knowledge_item(row)
                return None
                
        except Exception as e:
            logger.error(f"获取知识项失败: {e}")
            return None
    
    def search_knowledge(self, filters: Dict[str, Any] = None) -> List[KnowledgeItem]:
        """
        搜索知识项
        
        Args:
            filters: 搜索过滤条件
            
        Returns:
            知识项列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM knowledge_items"
                params = []
                conditions = []
                
                if filters:
                    if 'knowledge_type' in filters:
                        conditions.append("knowledge_type = ?")
                        params.append(filters['knowledge_type'])
                    
                    if 'source' in filters:
                        conditions.append("source = ?")
                        params.append(filters['source'])
                    
                    if 'min_confidence' in filters:
                        conditions.append("confidence >= ?")
                        params.append(filters['min_confidence'])
                    
                    if 'created_after' in filters:
                        conditions.append("created_at >= ?")
                        params.append(filters['created_after'])
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY updated_at DESC"
                
                if 'limit' in filters:
                    query += " LIMIT ?"
                    params.append(filters['limit'])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_knowledge_item(row) for row in rows]
                
        except Exception as e:
            logger.error(f"搜索知识项失败: {e}")
            return []
    
    def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新知识项
        
        Args:
            knowledge_id: 知识项ID
            updates: 更新字段
            
        Returns:
            更新结果
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 构建更新查询
                set_clauses = []
                params = []
                
                for field, value in updates.items():
                    if field in ['tags', 'metadata']:
                        set_clauses.append(f"{field} = ?")
                        params.append(json.dumps(value, ensure_ascii=False))
                    elif field == 'embedding':
                        set_clauses.append(f"{field} = ?")
                        params.append(pickle.dumps(value) if value is not None else None)
                    else:
                        set_clauses.append(f"{field} = ?")
                        params.append(value)
                
                # 添加更新时间
                set_clauses.append("updated_at = ?")
                params.append(time.strftime("%Y-%m-%d %H:%M:%S"))
                
                params.append(knowledge_id)
                
                query = f"UPDATE knowledge_items SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, params)
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"知识项更新成功: {knowledge_id}")
                    return True
                else:
                    logger.warning(f"知识项不存在: {knowledge_id}")
                    return False
                
        except Exception as e:
            logger.error(f"更新知识项失败: {e}")
            return False
    
    def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        删除知识项
        
        Args:
            knowledge_id: 知识项ID
            
        Returns:
            删除结果
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM knowledge_items WHERE id = ?", (knowledge_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"知识项删除成功: {knowledge_id}")
                    return True
                else:
                    logger.warning(f"知识项不存在: {knowledge_id}")
                    return False
                
        except Exception as e:
            logger.error(f"删除知识项失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 总数量
                cursor.execute("SELECT COUNT(*) FROM knowledge_items")
                total_count = cursor.fetchone()[0]
                
                # 按类型统计
                cursor.execute("""
                    SELECT knowledge_type, COUNT(*) 
                    FROM knowledge_items 
                    GROUP BY knowledge_type
                """)
                type_stats = dict(cursor.fetchall())
                
                # 按来源统计
                cursor.execute("""
                    SELECT source, COUNT(*) 
                    FROM knowledge_items 
                    GROUP BY source
                """)
                source_stats = dict(cursor.fetchall())
                
                # 平均置信度
                cursor.execute("SELECT AVG(confidence) FROM knowledge_items")
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                # 使用统计
                cursor.execute("SELECT SUM(usage_count) FROM knowledge_items")
                total_usage = cursor.fetchone()[0] or 0
                
                return {
                    "total_count": total_count,
                    "type_distribution": type_stats,
                    "source_distribution": source_stats,
                    "average_confidence": round(avg_confidence, 3),
                    "total_usage": total_usage
                }
                
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def _row_to_knowledge_item(self, row) -> KnowledgeItem:
        """将数据库行转换为知识项对象"""
        embedding = pickle.loads(row[7]) if row[7] else None
        
        return KnowledgeItem(
            id=row[0],
            title=row[1],
            content=row[2],
            knowledge_type=KnowledgeType(row[3]),
            source=KnowledgeSource(row[4]),
            tags=json.loads(row[5]) if row[5] else [],
            metadata=json.loads(row[6]) if row[6] else {},
            embedding=embedding,
            confidence=row[8],
            usage_count=row[9],
            last_used=row[10],
            created_at=row[11],
            updated_at=row[12]
        )


class KnowledgeEmbeddingEngine:
    """知识嵌入引擎"""
    
    def __init__(self, embedding_dim: int = 512):
        """
        初始化嵌入引擎
        
        Args:
            embedding_dim: 嵌入维度
        """
        self.embedding_dim = embedding_dim
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False
        
        # FAISS索引用于高效相似度搜索
        self.faiss_index = None
        self.knowledge_ids = []
        
        logger.info(f"知识嵌入引擎初始化完成，维度: {embedding_dim}")
    
    def fit_vectorizer(self, texts: List[str]) -> bool:
        """
        训练TF-IDF向量化器
        
        Args:
            texts: 文本列表
            
        Returns:
            训练结果
        """
        try:
            if not texts:
                logger.warning("没有文本数据用于训练向量化器")
                return False
            
            self.tfidf_vectorizer.fit(texts)
            self.is_fitted = True
            
            logger.info(f"TF-IDF向量化器训练完成，特征数: {len(self.tfidf_vectorizer.vocabulary_)}")
            return True
            
        except Exception as e:
            logger.error(f"向量化器训练失败: {e}")
            return False
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        生成文本嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        try:
            if not self.is_fitted:
                logger.warning("向量化器尚未训练")
                return None
            
            # 使用TF-IDF生成向量
            tfidf_vector = self.tfidf_vectorizer.transform([text]).toarray()[0]
            
            # 如果需要调整维度
            if len(tfidf_vector) < self.embedding_dim:
                # 填充零
                embedding = np.zeros(self.embedding_dim)
                embedding[:len(tfidf_vector)] = tfidf_vector
            elif len(tfidf_vector) > self.embedding_dim:
                # 截断
                embedding = tfidf_vector[:self.embedding_dim]
            else:
                embedding = tfidf_vector
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return None
    
    def build_faiss_index(self, embeddings: List[np.ndarray], knowledge_ids: List[str]):
        """
        构建FAISS索引
        
        Args:
            embeddings: 嵌入向量列表
            knowledge_ids: 对应的知识项ID列表
        """
        try:
            if not embeddings:
                logger.warning("没有嵌入向量用于构建索引")
                return
            
            # 转换为numpy数组
            embedding_matrix = np.array(embeddings).astype(np.float32)
            
            # 创建FAISS索引
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度
            self.faiss_index.add(embedding_matrix)
            
            self.knowledge_ids = knowledge_ids.copy()
            
            logger.info(f"FAISS索引构建完成，向量数: {len(embeddings)}")
            
        except Exception as e:
            logger.error(f"构建FAISS索引失败: {e}")
    
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 10, 
                      threshold: float = 0.3) -> Tuple[List[str], List[float]]:
        """
        搜索相似的知识项
        
        Args:
            query_embedding: 查询嵌入向量
            top_k: 返回top-k结果
            threshold: 相似度阈值
            
        Returns:
            知识项ID列表和相似度分数列表
        """
        try:
            if self.faiss_index is None:
                logger.warning("FAISS索引未构建")
                return [], []
            
            # 搜索最相似的向量
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self.faiss_index.search(query_vector, top_k)
            
            # 过滤低于阈值的结果
            valid_results = []
            valid_scores = []
            
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1 and score >= threshold:  # -1表示无效索引
                    valid_results.append(self.knowledge_ids[idx])
                    valid_scores.append(float(score))
            
            return valid_results, valid_scores
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return [], []


class KnowledgeManager:
    """知识管理器 - 统一的知识管理接口"""
    
    def __init__(self, db_path: str = "knowledge.db"):
        """
        初始化知识管理器
        
        Args:
            db_path: 数据库路径
        """
        self.database = KnowledgeDatabase(db_path)
        self.embedding_engine = KnowledgeEmbeddingEngine()
        
        # 初始化时建立嵌入引擎
        self._initialize_embedding_engine()
        
        logger.info("知识管理器初始化完成")
    
    def _initialize_embedding_engine(self):
        """初始化嵌入引擎"""
        try:
            # 获取所有知识项的内容用于训练向量化器
            knowledge_items = self.database.search_knowledge({'limit': 10000})
            
            if knowledge_items:
                texts = [item.content for item in knowledge_items]
                
                # 训练向量化器
                self.embedding_engine.fit_vectorizer(texts)
                
                # 生成所有知识项的嵌入向量
                embeddings = []
                ids = []
                
                for item in knowledge_items:
                    if item.embedding is None:
                        embedding = self.embedding_engine.generate_embedding(item.content)
                        if embedding is not None:
                            # 更新数据库中的嵌入向量
                            self.database.update_knowledge(item.id, {'embedding': embedding})
                            item.embedding = embedding
                    
                    if item.embedding is not None:
                        embeddings.append(item.embedding)
                        ids.append(item.id)
                
                # 构建FAISS索引
                if embeddings:
                    self.embedding_engine.build_faiss_index(embeddings, ids)
                
                logger.info(f"嵌入引擎初始化完成，处理知识项数: {len(embeddings)}")
            else:
                logger.info("暂无知识项，跳过嵌入引擎初始化")
                
        except Exception as e:
            logger.error(f"初始化嵌入引擎失败: {e}")
    
    def add_knowledge(self, title: str, content: str, knowledge_type: KnowledgeType,
                     source: KnowledgeSource, tags: List[str] = None,
                     metadata: Dict[str, Any] = None, confidence: float = 1.0) -> str:
        """
        添加知识项
        
        Args:
            title: 知识标题
            content: 知识内容
            knowledge_type: 知识类型
            source: 知识来源
            tags: 标签列表
            metadata: 元数据
            confidence: 置信度
            
        Returns:
            知识项ID
        """
        try:
            # 生成知识项ID
            knowledge_id = hashlib.md5(
                (title + content + str(time.time())).encode('utf-8')
            ).hexdigest()
            
            # 生成嵌入向量
            embedding = self.embedding_engine.generate_embedding(content)
            
            # 创建知识项
            knowledge_item = KnowledgeItem(
                id=knowledge_id,
                title=title,
                content=content,
                knowledge_type=knowledge_type,
                source=source,
                tags=tags or [],
                metadata=metadata or {},
                embedding=embedding,
                confidence=confidence
            )
            
            # 添加到数据库
            success = self.database.add_knowledge(knowledge_item)
            
            if success:
                # 重建FAISS索引（简化实现，实际可以增量更新）
                self._rebuild_faiss_index()
                logger.info(f"知识项添加成功: {knowledge_id}")
                return knowledge_id
            else:
                raise Exception("数据库添加失败")
                
        except Exception as e:
            logger.error(f"添加知识项失败: {e}")
            return ""
    
    def search_knowledge(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        """
        搜索知识项
        
        Args:
            query: 查询条件
            
        Returns:
            搜索结果
        """
        start_time = time.time()
        
        try:
            # 生成查询嵌入向量
            query_embedding = self.embedding_engine.generate_embedding(query.query_text)
            
            if query_embedding is not None:
                # 使用向量相似度搜索
                similar_ids, similarities = self.embedding_engine.search_similar(
                    query_embedding, 
                    query.max_results * 2,  # 获取更多候选，后续过滤
                    query.similarity_threshold
                )
                
                # 获取知识项详情
                knowledge_items = []
                final_similarities = []
                
                for knowledge_id, similarity in zip(similar_ids, similarities):
                    item = self.database.get_knowledge(knowledge_id)
                    if item and self._match_filters(item, query):
                        knowledge_items.append(item)
                        final_similarities.append(similarity)
                        
                        # 更新使用统计
                        self.database.update_knowledge(knowledge_id, {
                            'usage_count': item.usage_count + 1,
                            'last_used': time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        if len(knowledge_items) >= query.max_results:
                            break
            
            else:
                # 回退到传统搜索
                filters = {}
                if query.knowledge_types:
                    # 简化处理，只取第一个类型
                    filters['knowledge_type'] = query.knowledge_types[0].value
                
                filters['limit'] = query.max_results
                knowledge_items = self.database.search_knowledge(filters)
                final_similarities = [1.0] * len(knowledge_items)  # 默认相似度
            
            query_time = time.time() - start_time
            
            return KnowledgeSearchResult(
                items=knowledge_items,
                similarities=final_similarities,
                total_count=len(knowledge_items),
                query_time=query_time
            )
            
        except Exception as e:
            logger.error(f"搜索知识项失败: {e}")
            return KnowledgeSearchResult(
                items=[],
                similarities=[],
                total_count=0,
                query_time=time.time() - start_time
            )
    
    def _match_filters(self, item: KnowledgeItem, query: KnowledgeQuery) -> bool:
        """检查知识项是否匹配查询条件"""
        # 检查知识类型
        if query.knowledge_types and item.knowledge_type not in query.knowledge_types:
            return False
        
        # 检查来源
        if query.source_filter and item.source not in query.source_filter:
            return False
        
        # 检查置信度
        if item.confidence < query.confidence_threshold:
            return False
        
        # 检查标签
        if query.tags:
            if not any(tag in item.tags for tag in query.tags):
                return False
        
        return True
    
    def _rebuild_faiss_index(self):
        """重建FAISS索引"""
        try:
            knowledge_items = self.database.search_knowledge({'limit': 10000})
            embeddings = []
            ids = []
            
            for item in knowledge_items:
                if item.embedding is not None:
                    embeddings.append(item.embedding)
                    ids.append(item.id)
            
            if embeddings:
                self.embedding_engine.build_faiss_index(embeddings, ids)
                logger.info("FAISS索引重建完成")
            
        except Exception as e:
            logger.error(f"重建FAISS索引失败: {e}")
    
    def get_knowledge_by_id(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """根据ID获取知识项"""
        return self.database.get_knowledge(knowledge_id)
    
    def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """更新知识项"""
        return self.database.update_knowledge(knowledge_id, updates)
    
    def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除知识项"""
        success = self.database.delete_knowledge(knowledge_id)
        if success:
            self._rebuild_faiss_index()
        return success
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.database.get_statistics()
    
    def export_knowledge(self, output_path: str, filters: Dict[str, Any] = None) -> bool:
        """
        导出知识库
        
        Args:
            output_path: 输出文件路径
            filters: 过滤条件
            
        Returns:
            导出结果
        """
        try:
            knowledge_items = self.database.search_knowledge(filters)
            
            export_data = []
            for item in knowledge_items:
                item_dict = asdict(item)
                # 移除不能序列化的嵌入向量
                item_dict.pop('embedding', None)
                export_data.append(item_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"知识库导出完成: {output_path}, 数量: {len(export_data)}")
            return True
            
        except Exception as e:
            logger.error(f"导出知识库失败: {e}")
            return False


if __name__ == "__main__":
    # 测试代码
    manager = KnowledgeManager("test_knowledge.db")
    
    # 添加测试知识
    knowledge_id1 = manager.add_knowledge(
        title="Python基础",
        content="Python是一种高级编程语言，语法简洁，易于学习",
        knowledge_type=KnowledgeType.FACTUAL,
        source=KnowledgeSource.USER_INPUT,
        tags=["编程", "Python", "基础"],
        confidence=0.9
    )
    
    knowledge_id2 = manager.add_knowledge(
        title="机器学习",
        content="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习",
        knowledge_type=KnowledgeType.DOMAIN,
        source=KnowledgeSource.EXTERNAL_DATA,
        tags=["AI", "机器学习", "算法"],
        confidence=0.95
    )
    
    # 搜索知识
    query = KnowledgeQuery(
        query_text="Python编程语言",
        max_results=5,
        similarity_threshold=0.1
    )
    
    results = manager.search_knowledge(query)
    print(f"搜索结果数量: {results.total_count}")
    print(f"查询时间: {results.query_time:.3f}秒")
    
    for item, similarity in zip(results.items, results.similarities):
        print(f"- {item.title} (相似度: {similarity:.3f})")
        print(f"  内容: {item.content[:50]}...")
        print(f"  标签: {item.tags}")
    
    # 获取统计信息
    stats = manager.get_statistics()
    print(f"\n知识库统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 导出知识库
    success = manager.export_knowledge("exported_knowledge.json")
    print(f"导出结果: {success}")