"""
自学习Agent - 学习引擎模块
实现经验学习、模式识别、自我优化和适应性改进功能
"""

import json
import time
import hashlib
import pickle
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict, deque

from loguru import logger


class LearningType(Enum):
    """学习类型枚举"""
    SUPERVISED = "supervised"         # 监督学习
    UNSUPERVISED = "unsupervised"    # 无监督学习
    REINFORCEMENT = "reinforcement"   # 强化学习
    SELF_SUPERVISED = "self_supervised"  # 自监督学习
    TRANSFER = "transfer"            # 迁移学习


class ExperienceType(Enum):
    """经验类型枚举"""
    SUCCESS = "success"      # 成功经验
    FAILURE = "failure"      # 失败经验
    PATTERN = "pattern"      # 模式经验
    OPTIMIZATION = "optimization"  # 优化经验
    ADAPTATION = "adaptation"     # 适应经验


@dataclass
class Experience:
    """经验数据结构"""
    id: str
    experience_type: ExperienceType
    context: Dict[str, Any]
    action: Dict[str, Any]
    result: Dict[str, Any]
    outcome: str  # "success", "failure", "partial"
    confidence: float
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    used_count: int = 0
    last_used: Optional[str] = None


@dataclass
class LearningModel:
    """学习模型数据结构"""
    id: str
    name: str
    model_type: str
    learning_type: LearningType
    model_data: Any  # 序列化的模型数据
    performance_metrics: Dict[str, float]
    training_data_size: int
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    version: int = 1


@dataclass
class LearningTask:
    """学习任务数据结构"""
    id: str
    task_type: str
    learning_type: LearningType
    data_sources: List[str]
    target_metrics: Dict[str, float]
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    model_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    completed_at: Optional[str] = None


class ExperienceMemory:
    """经验记忆系统"""
    
    def __init__(self, max_size: int = 10000):
        """
        初始化经验记忆
        
        Args:
            max_size: 最大经验数量
        """
        self.max_size = max_size
        self.experiences: Dict[str, Experience] = {}
        self.experience_queue = deque(maxlen=max_size)
        self.type_indices: Dict[ExperienceType, List[str]] = defaultdict(list)
        
        logger.info(f"经验记忆系统初始化完成，最大容量: {max_size}")
    
    def add_experience(self, experience: Experience) -> bool:
        """
        添加经验
        
        Args:
            experience: 经验对象
            
        Returns:
            添加结果
        """
        try:
            # 如果经验已存在，更新它
            if experience.id in self.experiences:
                old_exp = self.experiences[experience.id]
                # 移除旧的类型索引
                if old_exp.experience_type in self.type_indices:
                    if experience.id in self.type_indices[old_exp.experience_type]:
                        self.type_indices[old_exp.experience_type].remove(experience.id)
            
            # 添加新经验
            self.experiences[experience.id] = experience
            
            # 如果队列满了，移除最旧的经验
            if len(self.experience_queue) >= self.max_size:
                old_id = self.experience_queue[0]
                if old_id in self.experiences:
                    old_exp = self.experiences.pop(old_id)
                    # 清理类型索引
                    if old_exp.experience_type in self.type_indices:
                        if old_id in self.type_indices[old_exp.experience_type]:
                            self.type_indices[old_exp.experience_type].remove(old_id)
            
            # 添加到队列和索引
            self.experience_queue.append(experience.id)
            self.type_indices[experience.experience_type].append(experience.id)
            
            logger.debug(f"经验添加成功: {experience.id}")
            return True
            
        except Exception as e:
            logger.error(f"添加经验失败: {e}")
            return False
    
    def get_experience(self, experience_id: str) -> Optional[Experience]:
        """获取经验"""
        experience = self.experiences.get(experience_id)
        if experience:
            experience.used_count += 1
            experience.last_used = time.strftime("%Y-%m-%d %H:%M:%S")
        return experience
    
    def search_experiences(self, 
                          experience_type: Optional[ExperienceType] = None,
                          outcome: Optional[str] = None,
                          min_confidence: float = 0.0,
                          limit: int = 100) -> List[Experience]:
        """
        搜索经验
        
        Args:
            experience_type: 经验类型过滤
            outcome: 结果过滤
            min_confidence: 最小置信度
            limit: 结果限制
            
        Returns:
            经验列表
        """
        results = []
        
        # 确定搜索范围
        if experience_type:
            search_ids = self.type_indices.get(experience_type, [])
        else:
            search_ids = list(self.experiences.keys())
        
        # 应用过滤条件
        for exp_id in search_ids:
            if len(results) >= limit:
                break
                
            experience = self.experiences.get(exp_id)
            if not experience:
                continue
            
            # 过滤条件检查
            if outcome and experience.outcome != outcome:
                continue
            
            if experience.confidence < min_confidence:
                continue
            
            results.append(experience)
        
        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取经验统计"""
        total_count = len(self.experiences)
        
        # 按类型统计
        type_stats = {}
        for exp_type, ids in self.type_indices.items():
            type_stats[exp_type.value] = len(ids)
        
        # 按结果统计
        outcome_stats = defaultdict(int)
        confidence_sum = 0.0
        
        for exp in self.experiences.values():
            outcome_stats[exp.outcome] += 1
            confidence_sum += exp.confidence
        
        avg_confidence = confidence_sum / total_count if total_count > 0 else 0.0
        
        return {
            "total_count": total_count,
            "type_distribution": type_stats,
            "outcome_distribution": dict(outcome_stats),
            "average_confidence": round(avg_confidence, 3),
            "memory_usage": f"{len(self.experience_queue)}/{self.max_size}"
        }


class PatternRecognition:
    """模式识别器"""
    
    def __init__(self):
        """初始化模式识别器"""
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.clustering_model = None
        
        logger.info("模式识别器初始化完成")
    
    def discover_patterns(self, experiences: List[Experience], 
                         min_support: float = 0.1) -> List[Dict[str, Any]]:
        """
        发现经验模式
        
        Args:
            experiences: 经验列表
            min_support: 最小支持度
            
        Returns:
            发现的模式列表
        """
        try:
            if not experiences:
                return []
            
            # 提取特征
            features = []
            experience_ids = []
            
            for exp in experiences:
                feature_vector = self._extract_features(exp)
                if feature_vector is not None:
                    features.append(feature_vector)
                    experience_ids.append(exp.id)
            
            if not features:
                return []
            
            # 聚类分析
            features_array = np.array(features)
            n_clusters = min(10, len(features) // 5 + 1)  # 动态确定聚类数
            
            if n_clusters < 2:
                return []
            
            self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = self.clustering_model.fit_predict(features_array)
            
            # 分析每个聚类
            patterns = []
            for cluster_id in range(n_clusters):
                cluster_experiences = [
                    experiences[i] for i, label in enumerate(cluster_labels) 
                    if label == cluster_id
                ]
                
                support = len(cluster_experiences) / len(experiences)
                if support >= min_support:
                    pattern = self._analyze_cluster(cluster_id, cluster_experiences)
                    if pattern:
                        pattern['support'] = support
                        patterns.append(pattern)
            
            logger.info(f"发现模式数量: {len(patterns)}")
            return patterns
            
        except Exception as e:
            logger.error(f"模式发现失败: {e}")
            return []
    
    def _extract_features(self, experience: Experience) -> Optional[List[float]]:
        """
        从经验中提取特征向量
        
        Args:
            experience: 经验对象
            
        Returns:
            特征向量
        """
        try:
            features = []
            
            # 经验类型编码
            type_mapping = {
                ExperienceType.SUCCESS: 1.0,
                ExperienceType.FAILURE: 0.0,
                ExperienceType.PATTERN: 0.5,
                ExperienceType.OPTIMIZATION: 0.7,
                ExperienceType.ADAPTATION: 0.3
            }
            features.append(type_mapping.get(experience.experience_type, 0.5))
            
            # 结果编码
            outcome_mapping = {"success": 1.0, "failure": 0.0, "partial": 0.5}
            features.append(outcome_mapping.get(experience.outcome, 0.5))
            
            # 置信度和奖励
            features.extend([experience.confidence, experience.reward])
            
            # 使用次数（归一化）
            features.append(min(experience.used_count / 10.0, 1.0))
            
            # 上下文特征（简化处理）
            context_size = len(str(experience.context))
            features.append(min(context_size / 1000.0, 1.0))
            
            return features
            
        except Exception as e:
            logger.warning(f"特征提取失败: {e}")
            return None
    
    def _analyze_cluster(self, cluster_id: int, 
                        cluster_experiences: List[Experience]) -> Optional[Dict[str, Any]]:
        """
        分析聚类模式
        
        Args:
            cluster_id: 聚类ID
            cluster_experiences: 聚类中的经验
            
        Returns:
            模式描述
        """
        try:
            if not cluster_experiences:
                return None
            
            # 统计模式特征
            success_rate = sum(1 for exp in cluster_experiences 
                             if exp.outcome == "success") / len(cluster_experiences)
            
            avg_confidence = np.mean([exp.confidence for exp in cluster_experiences])
            avg_reward = np.mean([exp.reward for exp in cluster_experiences])
            
            # 最常见的经验类型
            type_counts = defaultdict(int)
            for exp in cluster_experiences:
                type_counts[exp.experience_type.value] += 1
            
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
            
            # 生成模式ID
            pattern_id = hashlib.md5(
                f"cluster_{cluster_id}_{len(cluster_experiences)}_{time.time()}".encode()
            ).hexdigest()[:16]
            
            pattern = {
                "id": pattern_id,
                "cluster_id": cluster_id,
                "experience_count": len(cluster_experiences),
                "success_rate": round(success_rate, 3),
                "average_confidence": round(avg_confidence, 3),
                "average_reward": round(avg_reward, 3),
                "dominant_type": most_common_type,
                "type_distribution": dict(type_counts),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return pattern
            
        except Exception as e:
            logger.error(f"聚类分析失败: {e}")
            return None


class AdaptiveLearning:
    """自适应学习器"""
    
    def __init__(self):
        """初始化自适应学习器"""
        self.learning_models: Dict[str, LearningModel] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.adaptation_strategies: Dict[str, Callable] = {}
        
        logger.info("自适应学习器初始化完成")
    
    def create_learning_model(self, name: str, model_type: str, 
                            learning_type: LearningType,
                            training_data: List[Dict[str, Any]]) -> Optional[str]:
        """
        创建学习模型
        
        Args:
            name: 模型名称
            model_type: 模型类型
            learning_type: 学习类型
            training_data: 训练数据
            
        Returns:
            模型ID
        """
        try:
            model_id = hashlib.md5(
                f"{name}_{model_type}_{time.time()}".encode()
            ).hexdigest()[:16]
            
            # 根据学习类型创建模型
            if learning_type == LearningType.SUPERVISED:
                model, metrics = self._train_supervised_model(training_data)
            elif learning_type == LearningType.UNSUPERVISED:
                model, metrics = self._train_unsupervised_model(training_data)
            else:
                logger.warning(f"暂不支持的学习类型: {learning_type}")
                return None
            
            if model is None:
                return None
            
            # 创建学习模型对象
            learning_model = LearningModel(
                id=model_id,
                name=name,
                model_type=model_type,
                learning_type=learning_type,
                model_data=pickle.dumps(model),
                performance_metrics=metrics,
                training_data_size=len(training_data)
            )
            
            self.learning_models[model_id] = learning_model
            logger.info(f"学习模型创建成功: {model_id} ({name})")
            
            return model_id
            
        except Exception as e:
            logger.error(f"创建学习模型失败: {e}")
            return None
    
    def _train_supervised_model(self, training_data: List[Dict[str, Any]]) -> Tuple[Any, Dict[str, float]]:
        """训练监督学习模型"""
        try:
            # 提取特征和标签
            X = []
            y = []
            
            for data in training_data:
                if 'features' in data and 'label' in data:
                    X.append(data['features'])
                    y.append(data['label'])
            
            if not X or not y:
                logger.error("训练数据格式不正确")
                return None, {}
            
            X = np.array(X)
            y = np.array(y)
            
            # 分割数据
            if len(X) > 10:  # 只有足够数据时才分割
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            # 训练随机森林模型
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            metrics = {
                "accuracy": round(accuracy, 4),
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"监督学习训练失败: {e}")
            return None, {}
    
    def _train_unsupervised_model(self, training_data: List[Dict[str, Any]]) -> Tuple[Any, Dict[str, float]]:
        """训练无监督学习模型"""
        try:
            # 提取特征
            X = []
            for data in training_data:
                if 'features' in data:
                    X.append(data['features'])
            
            if not X:
                logger.error("训练数据格式不正确")
                return None, {}
            
            X = np.array(X)
            
            # 训练K-means聚类模型
            n_clusters = min(10, len(X) // 3 + 1)
            if n_clusters < 2:
                n_clusters = 2
            
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X)
            
            # 评估模型（使用轮廓系数等）
            from sklearn.metrics import silhouette_score
            if len(set(labels)) > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = 0.0
            
            metrics = {
                "silhouette_score": round(silhouette, 4),
                "n_clusters": n_clusters,
                "training_samples": len(X)
            }
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"无监督学习训练失败: {e}")
            return None, {}
    
    def predict(self, model_id: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用模型进行预测
        
        Args:
            model_id: 模型ID
            input_data: 输入数据
            
        Returns:
            预测结果
        """
        try:
            if model_id not in self.learning_models:
                logger.error(f"模型不存在: {model_id}")
                return None
            
            learning_model = self.learning_models[model_id]
            model = pickle.loads(learning_model.model_data)
            
            # 提取特征
            if 'features' not in input_data:
                logger.error("输入数据缺少features字段")
                return None
            
            features = np.array([input_data['features']])
            
            # 进行预测
            if learning_model.learning_type == LearningType.SUPERVISED:
                prediction = model.predict(features)[0]
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features)[0].tolist()
                else:
                    probabilities = None
                
                result = {
                    "prediction": prediction,
                    "probabilities": probabilities,
                    "model_id": model_id,
                    "confidence": max(probabilities) if probabilities else 0.8
                }
            
            elif learning_model.learning_type == LearningType.UNSUPERVISED:
                cluster_label = model.predict(features)[0]
                distances = model.transform(features)[0]
                
                result = {
                    "cluster": int(cluster_label),
                    "distances": distances.tolist(),
                    "model_id": model_id,
                    "confidence": 1.0 - min(distances) / max(distances) if max(distances) > 0 else 0.5
                }
            
            else:
                logger.warning(f"不支持的学习类型预测: {learning_model.learning_type}")
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return None
    
    def update_performance(self, model_id: str, performance_score: float):
        """更新模型性能"""
        if model_id in self.learning_models:
            self.performance_history[model_id].append(performance_score)
            # 保持最近100个性能记录
            if len(self.performance_history[model_id]) > 100:
                self.performance_history[model_id] = self.performance_history[model_id][-100:]
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        if model_id not in self.learning_models:
            return None
        
        model = self.learning_models[model_id]
        performance_history = self.performance_history.get(model_id, [])
        
        return {
            "id": model.id,
            "name": model.name,
            "model_type": model.model_type,
            "learning_type": model.learning_type.value,
            "performance_metrics": model.performance_metrics,
            "training_data_size": model.training_data_size,
            "performance_history": performance_history[-10:],  # 最近10次性能
            "average_performance": np.mean(performance_history) if performance_history else 0.0,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "version": model.version
        }


class LearningEngine:
    """学习引擎 - 统一的学习管理接口"""
    
    def __init__(self, memory_size: int = 10000):
        """
        初始化学习引擎
        
        Args:
            memory_size: 经验记忆容量
        """
        self.experience_memory = ExperienceMemory(memory_size)
        self.pattern_recognition = PatternRecognition()
        self.adaptive_learning = AdaptiveLearning()
        self.learning_tasks: Dict[str, LearningTask] = {}
        
        logger.info("学习引擎初始化完成")
    
    def record_experience(self, experience_type: ExperienceType,
                         context: Dict[str, Any],
                         action: Dict[str, Any],
                         result: Dict[str, Any],
                         outcome: str,
                         confidence: float = 0.8,
                         reward: float = 0.0,
                         metadata: Dict[str, Any] = None) -> str:
        """
        记录经验
        
        Args:
            experience_type: 经验类型
            context: 上下文信息
            action: 执行的动作
            result: 结果
            outcome: 结果评价
            confidence: 置信度
            reward: 奖励值
            metadata: 元数据
            
        Returns:
            经验ID
        """
        try:
            experience_id = hashlib.md5(
                f"{experience_type.value}_{time.time()}_{hash(str(context))}".encode()
            ).hexdigest()
            
            experience = Experience(
                id=experience_id,
                experience_type=experience_type,
                context=context,
                action=action,
                result=result,
                outcome=outcome,
                confidence=confidence,
                reward=reward,
                metadata=metadata or {}
            )
            
            success = self.experience_memory.add_experience(experience)
            if success:
                logger.info(f"经验记录成功: {experience_id}")
                return experience_id
            else:
                logger.error("经验记录失败")
                return ""
                
        except Exception as e:
            logger.error(f"记录经验失败: {e}")
            return ""
    
    def learn_from_experiences(self, min_experiences: int = 10) -> Dict[str, Any]:
        """
        从经验中学习
        
        Args:
            min_experiences: 最小经验数量
            
        Returns:
            学习结果
        """
        try:
            # 获取所有经验
            all_experiences = self.experience_memory.search_experiences(limit=1000)
            
            if len(all_experiences) < min_experiences:
                logger.warning(f"经验数量不足，需要至少{min_experiences}个，当前{len(all_experiences)}个")
                return {"success": False, "message": "经验数量不足"}
            
            results = {}
            
            # 模式发现
            patterns = self.pattern_recognition.discover_patterns(all_experiences)
            results["patterns_discovered"] = len(patterns)
            results["patterns"] = patterns
            
            # 创建学习数据
            training_data = []
            for exp in all_experiences:
                features = self._experience_to_features(exp)
                if features:
                    training_data.append({
                        "features": features,
                        "label": 1 if exp.outcome == "success" else 0
                    })
            
            # 训练自适应模型
            if training_data:
                model_id = self.adaptive_learning.create_learning_model(
                    name=f"experience_model_{int(time.time())}",
                    model_type="random_forest",
                    learning_type=LearningType.SUPERVISED,
                    training_data=training_data
                )
                
                if model_id:
                    results["model_created"] = True
                    results["model_id"] = model_id
                    
                    # 评估模型性能
                    model_info = self.adaptive_learning.get_model_info(model_id)
                    if model_info:
                        results["model_performance"] = model_info["performance_metrics"]
                else:
                    results["model_created"] = False
            
            results["success"] = True
            results["experiences_processed"] = len(all_experiences)
            results["training_data_size"] = len(training_data)
            
            logger.info(f"从经验学习完成: {json.dumps(results, indent=2, ensure_ascii=False)}")
            return results
            
        except Exception as e:
            logger.error(f"从经验学习失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _experience_to_features(self, experience: Experience) -> Optional[List[float]]:
        """将经验转换为特征向量"""
        try:
            features = []
            
            # 经验类型特征
            type_features = {
                ExperienceType.SUCCESS: [1, 0, 0, 0, 0],
                ExperienceType.FAILURE: [0, 1, 0, 0, 0],
                ExperienceType.PATTERN: [0, 0, 1, 0, 0],
                ExperienceType.OPTIMIZATION: [0, 0, 0, 1, 0],
                ExperienceType.ADAPTATION: [0, 0, 0, 0, 1]
            }
            features.extend(type_features.get(experience.experience_type, [0, 0, 0, 0, 0]))
            
            # 数值特征
            features.extend([
                experience.confidence,
                experience.reward,
                experience.used_count / 10.0,  # 归一化
            ])
            
            # 上下文复杂度
            context_complexity = len(str(experience.context)) / 1000.0  # 归一化
            features.append(min(context_complexity, 1.0))
            
            return features
            
        except Exception as e:
            logger.warning(f"特征提取失败: {e}")
            return None
    
    def predict_outcome(self, model_id: str, context: Dict[str, Any], 
                       action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        预测行动结果
        
        Args:
            model_id: 模型ID
            context: 上下文
            action: 行动
            
        Returns:
            预测结果
        """
        try:
            # 构建输入特征
            features = []
            
            # 简化的特征提取（实际应该更复杂）
            features.extend([
                len(str(context)) / 1000.0,
                len(str(action)) / 1000.0,
                hash(str(context)) % 100 / 100.0,  # 上下文哈希特征
                hash(str(action)) % 100 / 100.0    # 动作哈希特征
            ])
            
            # 填充到标准特征长度
            while len(features) < 9:  # 与_experience_to_features保持一致
                features.append(0.0)
            
            input_data = {"features": features}
            
            # 使用模型预测
            prediction = self.adaptive_learning.predict(model_id, input_data)
            
            if prediction:
                logger.info(f"结果预测完成: {prediction}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"预测结果失败: {e}")
            return None
    
    def optimize_strategy(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化策略建议
        
        Args:
            task_context: 任务上下文
            
        Returns:
            优化建议
        """
        try:
            # 搜索相关经验
            relevant_experiences = self.experience_memory.search_experiences(
                experience_type=ExperienceType.SUCCESS,
                min_confidence=0.5,
                limit=50
            )
            
            if not relevant_experiences:
                return {"success": False, "message": "没有找到相关的成功经验"}
            
            # 分析成功模式
            success_patterns = []
            for exp in relevant_experiences:
                # 提取成功要素
                pattern = {
                    "context_size": len(str(exp.context)),
                    "action_complexity": len(str(exp.action)),
                    "confidence": exp.confidence,
                    "reward": exp.reward,
                    "context_keys": list(exp.context.keys()) if isinstance(exp.context, dict) else [],
                    "action_keys": list(exp.action.keys()) if isinstance(exp.action, dict) else []
                }
                success_patterns.append(pattern)
            
            # 生成优化建议
            recommendations = []
            
            # 平均置信度建议
            avg_confidence = np.mean([p["confidence"] for p in success_patterns])
            if avg_confidence > 0.8:
                recommendations.append({
                    "type": "confidence_threshold",
                    "suggestion": f"保持高置信度操作 (>{avg_confidence:.2f})",
                    "priority": "high"
                })
            
            # 上下文复杂度建议
            avg_context_size = np.mean([p["context_size"] for p in success_patterns])
            recommendations.append({
                "type": "context_optimization",
                "suggestion": f"优化上下文信息到{avg_context_size:.0f}字符左右",
                "priority": "medium"
            })
            
            # 常见成功要素
            all_context_keys = []
            all_action_keys = []
            for p in success_patterns:
                all_context_keys.extend(p["context_keys"])
                all_action_keys.extend(p["action_keys"])
            
            # 统计最常见的键
            from collections import Counter
            common_context_keys = Counter(all_context_keys).most_common(5)
            common_action_keys = Counter(all_action_keys).most_common(5)
            
            if common_context_keys:
                recommendations.append({
                    "type": "key_elements",
                    "suggestion": f"关注关键上下文要素: {[k for k, v in common_context_keys]}",
                    "priority": "high"
                })
            
            if common_action_keys:
                recommendations.append({
                    "type": "action_elements", 
                    "suggestion": f"重点考虑关键行动要素: {[k for k, v in common_action_keys]}",
                    "priority": "high"
                })
            
            result = {
                "success": True,
                "analyzed_experiences": len(relevant_experiences),
                "recommendations": recommendations,
                "success_rate": len(relevant_experiences) / max(1, len(self.experience_memory.experiences)),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"策略优化完成: {len(recommendations)}个建议")
            return result
            
        except Exception as e:
            logger.error(f"策略优化失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        try:
            # 经验统计
            exp_stats = self.experience_memory.get_statistics()
            
            # 模型统计
            model_count = len(self.adaptive_learning.learning_models)
            model_types = defaultdict(int)
            
            for model in self.adaptive_learning.learning_models.values():
                model_types[model.learning_type.value] += 1
            
            # 学习任务统计
            task_stats = defaultdict(int)
            for task in self.learning_tasks.values():
                task_stats[task.status] += 1
            
            return {
                "experience_memory": exp_stats,
                "learning_models": {
                    "total_count": model_count,
                    "type_distribution": dict(model_types)
                },
                "learning_tasks": {
                    "total_count": len(self.learning_tasks),
                    "status_distribution": dict(task_stats)
                },
                "system_status": "active" if exp_stats["total_count"] > 0 else "idle"
            }
            
        except Exception as e:
            logger.error(f"获取学习统计失败: {e}")
            return {}
    
    def export_learning_data(self, output_path: str) -> bool:
        """
        导出学习数据
        
        Args:
            output_path: 输出路径
            
        Returns:
            导出结果
        """
        try:
            export_data = {
                "experiences": [],
                "models": [],
                "statistics": self.get_learning_statistics(),
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 导出经验（不包含复杂对象）
            for exp in self.experience_memory.experiences.values():
                exp_dict = {
                    "id": exp.id,
                    "experience_type": exp.experience_type.value,
                    "context": exp.context,
                    "action": exp.action,
                    "result": exp.result,
                    "outcome": exp.outcome,
                    "confidence": exp.confidence,
                    "reward": exp.reward,
                    "metadata": exp.metadata,
                    "created_at": exp.created_at,
                    "used_count": exp.used_count,
                    "last_used": exp.last_used
                }
                export_data["experiences"].append(exp_dict)
            
            # 导出模型信息（不包含模型数据）
            for model in self.adaptive_learning.learning_models.values():
                model_dict = {
                    "id": model.id,
                    "name": model.name,
                    "model_type": model.model_type,
                    "learning_type": model.learning_type.value,
                    "performance_metrics": model.performance_metrics,
                    "training_data_size": model.training_data_size,
                    "created_at": model.created_at,
                    "updated_at": model.updated_at,
                    "version": model.version
                }
                export_data["models"].append(model_dict)
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"学习数据导出完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出学习数据失败: {e}")
            return False


if __name__ == "__main__":
    # 测试代码
    engine = LearningEngine(memory_size=1000)
    
    # 记录一些测试经验
    for i in range(20):
        success = i % 3 != 0  # 大部分成功
        outcome = "success" if success else "failure"
        reward = 1.0 if success else -0.5
        
        exp_id = engine.record_experience(
            experience_type=ExperienceType.SUCCESS if success else ExperienceType.FAILURE,
            context={"task_id": i, "complexity": i % 5, "user_input": f"test_{i}"},
            action={"method": f"method_{i % 3}", "parameters": {"param1": i}},
            result={"output": f"result_{i}", "time_taken": i * 0.1},
            outcome=outcome,
            confidence=0.8 + (i % 3) * 0.1,
            reward=reward,
            metadata={"experiment": "test_run"}
        )
        
        print(f"经验记录: {exp_id}")
    
    # 从经验中学习
    print("\n开始从经验中学习...")
    learning_results = engine.learn_from_experiences()
    print(f"学习结果: {json.dumps(learning_results, indent=2, ensure_ascii=False)}")
    
    # 策略优化
    print("\n获取策略优化建议...")
    optimization = engine.optimize_strategy({"current_task": "optimization_test"})
    print(f"优化建议: {json.dumps(optimization, indent=2, ensure_ascii=False)}")
    
    # 获取统计信息
    print("\n学习统计信息:")
    stats = engine.get_learning_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 导出学习数据
    success = engine.export_learning_data("learning_data_export.json")
    print(f"\n数据导出结果: {success}")