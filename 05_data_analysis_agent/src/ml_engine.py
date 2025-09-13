"""
机器学习引擎
支持分类、回归、聚类等常见机器学习任务
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import joblib
import json
from datetime import datetime

# Third-party imports
from loguru import logger
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 机器学习算法
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config


class MLEngine:
    """机器学习引擎主类"""
    
    def __init__(self):
        """初始化机器学习引擎"""
        self.config = get_config()
        self.models: Dict[str, Any] = {}
        self.model_history: Dict[str, List[Dict]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # 支持的算法
        self.classification_algorithms = {
            'logistic_regression': LogisticRegression,
            'decision_tree': DecisionTreeClassifier,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svm': SVC,
            'naive_bayes': GaussianNB,
            'knn': KNeighborsClassifier,
            'mlp': MLPClassifier
        }
        
        self.regression_algorithms = {
            'linear_regression': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'decision_tree': DecisionTreeRegressor,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'svm': SVR,
            'knn': KNeighborsRegressor,
            'mlp': MLPRegressor
        }
        
        self.clustering_algorithms = {
            'kmeans': KMeans,
            'dbscan': DBSCAN,
            'agglomerative': AgglomerativeClustering
        }
        
        logger.info("机器学习引擎初始化完成")
    
    def train_model(self,
                   model_name: str,
                   algorithm: str,
                   task_type: str,
                   X_train: pd.DataFrame,
                   y_train: Optional[pd.Series] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            model_name: 模型名称
            algorithm: 算法名称
            task_type: 任务类型 (classification, regression, clustering)
            X_train: 训练特征
            y_train: 训练目标（聚类任务可选）
            **kwargs: 算法参数
            
        Returns:
            训练结果
        """
        try:
            # 选择算法
            if task_type == "classification":
                if algorithm not in self.classification_algorithms:
                    raise ValueError(f"不支持的分类算法: {algorithm}")
                model_class = self.classification_algorithms[algorithm]
            elif task_type == "regression":
                if algorithm not in self.regression_algorithms:
                    raise ValueError(f"不支持的回归算法: {algorithm}")
                model_class = self.regression_algorithms[algorithm]
            elif task_type == "clustering":
                if algorithm not in self.clustering_algorithms:
                    raise ValueError(f"不支持的聚类算法: {algorithm}")
                model_class = self.clustering_algorithms[algorithm]
                if y_train is not None:
                    logger.warning("聚类任务不需要目标变量，将被忽略")
                    y_train = None
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")
            
            # 数据预处理
            X_processed = self._preprocess_features(X_train, model_name, fit=True)
            
            # 创建并训练模型
            model = model_class(**kwargs)
            
            start_time = datetime.now()
            
            if task_type in ["classification", "regression"]:
                if y_train is None:
                    raise ValueError(f"{task_type} 任务需要目标变量")
                model.fit(X_processed, y_train)
            else:  # clustering
                model.fit(X_processed)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # 存储模型
            self.models[model_name] = {
                'model': model,
                'algorithm': algorithm,
                'task_type': task_type,
                'feature_names': X_train.columns.tolist(),
                'training_time': training_time,
                'created_at': start_time.isoformat()
            }
            
            # 记录训练历史
            if model_name not in self.model_history:
                self.model_history[model_name] = []
            
            history_entry = {
                'action': 'train',
                'algorithm': algorithm,
                'task_type': task_type,
                'training_samples': len(X_train),
                'features': len(X_train.columns),
                'parameters': kwargs,
                'training_time': training_time,
                'timestamp': start_time.isoformat()
            }
            
            self.model_history[model_name].append(history_entry)
            
            result = {
                'model_name': model_name,
                'algorithm': algorithm,
                'task_type': task_type,
                'training_samples': len(X_train),
                'features': len(X_train.columns),
                'training_time': training_time,
                'success': True
            }
            
            # 如果是聚类任务，添加聚类结果
            if task_type == "clustering":
                labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_processed)
                result['cluster_labels'] = labels.tolist()
                result['n_clusters'] = len(np.unique(labels))
                
                # 计算轮廓系数（如果不是噪声点过多）
                if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels):
                    try:
                        silhouette_avg = silhouette_score(X_processed, labels)
                        result['silhouette_score'] = silhouette_avg
                    except:
                        pass
            
            logger.info(f"模型训练完成: {model_name}, 算法: {algorithm}, 时间: {training_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise
    
    def predict(self,
               model_name: str,
               X_test: pd.DataFrame,
               return_probabilities: bool = False) -> Dict[str, Any]:
        """
        模型预测
        
        Args:
            model_name: 模型名称
            X_test: 测试特征
            return_probabilities: 是否返回概率（仅分类任务）
            
        Returns:
            预测结果
        """
        if model_name not in self.models:
            raise ValueError(f"模型不存在: {model_name}")
        
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            task_type = model_info['task_type']
            
            # 数据预处理
            X_processed = self._preprocess_features(X_test, model_name, fit=False)
            
            # 预测
            start_time = datetime.now()
            predictions = model.predict(X_processed)
            end_time = datetime.now()
            
            prediction_time = (end_time - start_time).total_seconds()
            
            result = {
                'model_name': model_name,
                'predictions': predictions.tolist(),
                'prediction_time': prediction_time,
                'samples_predicted': len(X_test)
            }
            
            # 如果是分类任务且需要概率
            if task_type == "classification" and return_probabilities:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_processed)
                    result['probabilities'] = probabilities.tolist()
                    if hasattr(model, 'classes_'):
                        result['class_names'] = model.classes_.tolist()
            
            logger.info(f"模型预测完成: {model_name}, 样本数: {len(X_test)}")
            return result
            
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            raise
    
    def evaluate_model(self,
                      model_name: str,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        模型评估
        
        Args:
            model_name: 模型名称
            X_test: 测试特征
            y_test: 测试目标
            metrics: 评估指标列表
            
        Returns:
            评估结果
        """
        if model_name not in self.models:
            raise ValueError(f"模型不存在: {model_name}")
        
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            task_type = model_info['task_type']
            
            if task_type == "clustering":
                raise ValueError("聚类模型无法使用有监督评估")
            
            # 预测
            prediction_result = self.predict(model_name, X_test, return_probabilities=True)
            predictions = np.array(prediction_result['predictions'])
            
            # 根据任务类型选择默认指标
            if metrics is None:
                if task_type == "classification":
                    metrics = ['accuracy', 'precision', 'recall', 'f1']
                elif task_type == "regression":
                    metrics = ['mse', 'mae', 'r2']
            
            # 计算评估指标
            evaluation_results = {}
            
            if task_type == "classification":
                # 分类指标
                if 'accuracy' in metrics:
                    evaluation_results['accuracy'] = accuracy_score(y_test, predictions)
                
                # 对于多分类，使用平均方式计算
                avg_method = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
                
                if 'precision' in metrics:
                    evaluation_results['precision'] = precision_score(
                        y_test, predictions, average=avg_method, zero_division=0
                    )
                
                if 'recall' in metrics:
                    evaluation_results['recall'] = recall_score(
                        y_test, predictions, average=avg_method, zero_division=0
                    )
                
                if 'f1' in metrics:
                    evaluation_results['f1'] = f1_score(
                        y_test, predictions, average=avg_method, zero_division=0
                    )
            
            elif task_type == "regression":
                # 回归指标
                if 'mse' in metrics:
                    evaluation_results['mse'] = mean_squared_error(y_test, predictions)
                
                if 'mae' in metrics:
                    evaluation_results['mae'] = mean_absolute_error(y_test, predictions)
                
                if 'r2' in metrics:
                    evaluation_results['r2'] = r2_score(y_test, predictions)
                
                if 'rmse' in metrics:
                    evaluation_results['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
            
            result = {
                'model_name': model_name,
                'task_type': task_type,
                'metrics': evaluation_results,
                'test_samples': len(X_test),
                'evaluation_time': prediction_result['prediction_time']
            }
            
            # 记录评估历史
            history_entry = {
                'action': 'evaluate',
                'metrics': evaluation_results,
                'test_samples': len(X_test),
                'timestamp': datetime.now().isoformat()
            }
            self.model_history[model_name].append(history_entry)
            
            logger.info(f"模型评估完成: {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            raise
    
    def cross_validate(self,
                      algorithm: str,
                      task_type: str,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv: int = 5,
                      scoring: str = None,
                      **kwargs) -> Dict[str, Any]:
        """
        交叉验证
        
        Args:
            algorithm: 算法名称
            task_type: 任务类型
            X: 特征数据
            y: 目标变量
            cv: 交叉验证折数
            scoring: 评分方法
            **kwargs: 算法参数
            
        Returns:
            交叉验证结果
        """
        try:
            # 选择算法
            if task_type == "classification":
                if algorithm not in self.classification_algorithms:
                    raise ValueError(f"不支持的分类算法: {algorithm}")
                model_class = self.classification_algorithms[algorithm]
                if scoring is None:
                    scoring = 'accuracy'
            elif task_type == "regression":
                if algorithm not in self.regression_algorithms:
                    raise ValueError(f"不支持的回归算法: {algorithm}")
                model_class = self.regression_algorithms[algorithm]
                if scoring is None:
                    scoring = 'neg_mean_squared_error'
            else:
                raise ValueError("交叉验证不支持聚类任务")
            
            # 创建模型
            model = model_class(**kwargs)
            
            # 数据预处理
            temp_scaler = StandardScaler()
            X_processed = temp_scaler.fit_transform(X)
            
            # 执行交叉验证
            start_time = datetime.now()
            cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring=scoring)
            end_time = datetime.now()
            
            cv_time = (end_time - start_time).total_seconds()
            
            result = {
                'algorithm': algorithm,
                'task_type': task_type,
                'cv_folds': cv,
                'scoring': scoring,
                'cv_scores': cv_scores.tolist(),
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'cv_time': cv_time,
                'samples': len(X),
                'features': len(X.columns)
            }
            
            logger.info(f"交叉验证完成: {algorithm}, 平均得分: {cv_scores.mean():.4f}")
            return result
            
        except Exception as e:
            logger.error(f"交叉验证失败: {e}")
            raise
    
    def hyperparameter_tuning(self,
                             model_name: str,
                             algorithm: str,
                             task_type: str,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             param_grid: Dict[str, List],
                             cv: int = 5,
                             scoring: str = None) -> Dict[str, Any]:
        """
        超参数调优
        
        Args:
            model_name: 模型名称
            algorithm: 算法名称
            task_type: 任务类型
            X_train: 训练特征
            y_train: 训练目标
            param_grid: 参数网格
            cv: 交叉验证折数
            scoring: 评分方法
            
        Returns:
            调优结果
        """
        try:
            # 选择算法和默认评分方法
            if task_type == "classification":
                if algorithm not in self.classification_algorithms:
                    raise ValueError(f"不支持的分类算法: {algorithm}")
                model_class = self.classification_algorithms[algorithm]
                if scoring is None:
                    scoring = 'accuracy'
            elif task_type == "regression":
                if algorithm not in self.regression_algorithms:
                    raise ValueError(f"不支持的回归算法: {algorithm}")
                model_class = self.regression_algorithms[algorithm]
                if scoring is None:
                    scoring = 'neg_mean_squared_error'
            else:
                raise ValueError("超参数调优不支持聚类任务")
            
            # 创建基础模型
            model = model_class()
            
            # 数据预处理
            X_processed = self._preprocess_features(X_train, f"{model_name}_tuning", fit=True)
            
            # 网格搜索
            start_time = datetime.now()
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
            )
            grid_search.fit(X_processed, y_train)
            end_time = datetime.now()
            
            tuning_time = (end_time - start_time).total_seconds()
            
            # 使用最佳参数训练最终模型
            best_model = model_class(**grid_search.best_params_)
            best_model.fit(X_processed, y_train)
            
            # 存储最佳模型
            self.models[model_name] = {
                'model': best_model,
                'algorithm': algorithm,
                'task_type': task_type,
                'feature_names': X_train.columns.tolist(),
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'tuning_time': tuning_time,
                'created_at': start_time.isoformat()
            }
            
            result = {
                'model_name': model_name,
                'algorithm': algorithm,
                'task_type': task_type,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': {
                    'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                },
                'tuning_time': tuning_time,
                'total_fits': len(grid_search.cv_results_['params'])
            }
            
            # 记录调优历史
            if model_name not in self.model_history:
                self.model_history[model_name] = []
            
            history_entry = {
                'action': 'hyperparameter_tuning',
                'algorithm': algorithm,
                'param_grid': param_grid,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'tuning_time': tuning_time,
                'timestamp': start_time.isoformat()
            }
            self.model_history[model_name].append(history_entry)
            
            logger.info(f"超参数调优完成: {model_name}, 最佳得分: {grid_search.best_score_:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"超参数调优失败: {e}")
            raise
    
    def save_model(self, model_name: str, file_path: Union[str, Path]) -> bool:
        """
        保存模型
        
        Args:
            model_name: 模型名称
            file_path: 保存路径
            
        Returns:
            是否成功
        """
        if model_name not in self.models:
            raise ValueError(f"模型不存在: {model_name}")
        
        try:
            file_path = Path(file_path)
            
            # 准备保存数据
            save_data = {
                'model_info': self.models[model_name],
                'model_history': self.model_history.get(model_name, []),
                'scaler': self.scalers.get(model_name)
            }
            
            # 保存模型
            joblib.dump(save_data, file_path)
            
            logger.info(f"模型保存成功: {model_name} -> {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return False
    
    def load_model(self, model_name: str, file_path: Union[str, Path]) -> bool:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            file_path: 模型文件路径
            
        Returns:
            是否成功
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {file_path}")
            
            # 加载模型数据
            save_data = joblib.load(file_path)
            
            # 恢复模型信息
            self.models[model_name] = save_data['model_info']
            self.model_history[model_name] = save_data.get('model_history', [])
            
            if save_data.get('scaler'):
                self.scalers[model_name] = save_data['scaler']
            
            logger.info(f"模型加载成功: {model_name} <- {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息
        """
        if model_name not in self.models:
            raise ValueError(f"模型不存在: {model_name}")
        
        model_info = self.models[model_name].copy()
        
        # 移除不能序列化的模型对象
        model_info.pop('model', None)
        
        # 添加历史信息
        model_info['history'] = self.model_history.get(model_name, [])
        
        return model_info
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        列出所有模型
        
        Returns:
            模型列表
        """
        models_list = []
        
        for name, info in self.models.items():
            model_summary = {
                'name': name,
                'algorithm': info['algorithm'],
                'task_type': info['task_type'],
                'features': len(info['feature_names']),
                'created_at': info['created_at']
            }
            
            # 添加性能指标（如果有）
            if 'best_score' in info:
                model_summary['best_score'] = info['best_score']
            
            models_list.append(model_summary)
        
        return models_list
    
    def remove_model(self, model_name: str) -> bool:
        """
        删除模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否成功
        """
        if model_name in self.models:
            del self.models[model_name]
            
            if model_name in self.model_history:
                del self.model_history[model_name]
            
            if model_name in self.scalers:
                del self.scalers[model_name]
            
            logger.info(f"模型已删除: {model_name}")
            return True
        
        return False
    
    def _preprocess_features(self, 
                           X: pd.DataFrame, 
                           model_name: str, 
                           fit: bool = False) -> np.ndarray:
        """
        特征预处理
        
        Args:
            X: 特征数据
            model_name: 模型名称
            fit: 是否拟合预处理器
            
        Returns:
            处理后的特征
        """
        # 处理非数值列
        X_processed = X.copy()
        
        # 标签编码分类特征
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # 标准化数值特征
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            self.scalers[model_name] = scaler
        else:
            if model_name in self.scalers:
                scaler = self.scalers[model_name]
                X_scaled = scaler.transform(X_processed)
            else:
                logger.warning(f"未找到模型 {model_name} 的预处理器，跳过标准化")
                X_scaled = X_processed.values
        
        return X_scaled
    
    def get_feature_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        获取特征重要性
        
        Args:
            model_name: 模型名称
            
        Returns:
            特征重要性字典
        """
        if model_name not in self.models:
            raise ValueError(f"模型不存在: {model_name}")
        
        model_info = self.models[model_name]
        model = model_info['model']
        feature_names = model_info['feature_names']
        
        # 检查模型是否支持特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))
        elif hasattr(model, 'coef_'):
            # 对于线性模型，使用系数的绝对值
            if len(model.coef_.shape) == 1:
                importances = np.abs(model.coef_)
            else:
                importances = np.abs(model.coef_).mean(axis=0)
            return dict(zip(feature_names, importances))
        else:
            logger.warning(f"模型 {model_name} 不支持特征重要性分析")
            return None
    
    def compare_models(self, 
                      model_names: List[str], 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, Any]:
        """
        比较多个模型
        
        Args:
            model_names: 模型名称列表
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            比较结果
        """
        comparison_results = {}
        
        for model_name in model_names:
            if model_name not in self.models:
                logger.warning(f"模型不存在，跳过: {model_name}")
                continue
            
            try:
                # 评估模型
                eval_result = self.evaluate_model(model_name, X_test, y_test)
                comparison_results[model_name] = eval_result
                
            except Exception as e:
                logger.error(f"模型 {model_name} 评估失败: {e}")
                comparison_results[model_name] = {"error": str(e)}
        
        return {
            'models_compared': len(comparison_results),
            'test_samples': len(X_test),
            'results': comparison_results,
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # 测试机器学习引擎
    from sklearn.datasets import make_classification, make_regression
    
    ml_engine = MLEngine()
    
    try:
        # 创建分类测试数据
        X_cls, y_cls = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        X_cls = pd.DataFrame(X_cls, columns=[f'feature_{i}' for i in range(10)])
        y_cls = pd.Series(y_cls)
        
        X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
        
        # 训练分类模型
        train_result = ml_engine.train_model(
            "test_classifier", "random_forest", "classification",
            X_train, y_train, n_estimators=100, random_state=42
        )
        print(f"分类模型训练结果: {train_result}")
        
        # 模型评估
        eval_result = ml_engine.evaluate_model("test_classifier", X_test, y_test)
        print(f"分类模型评估结果: {eval_result}")
        
        # 特征重要性
        importance = ml_engine.get_feature_importance("test_classifier")
        print(f"特征重要性: {importance}")
        
        print("机器学习引擎测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")