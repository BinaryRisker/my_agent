"""
数据处理器
支持各种数据格式的加载、清洗、转换和分析
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import sqlite3
from io import StringIO
import warnings

# Third-party imports
from loguru import logger
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config

warnings.filterwarnings('ignore')


class DataProcessor:
    """数据处理器主类"""
    
    def __init__(self):
        """初始化数据处理器"""
        self.config = get_config()
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.preprocessing_history: Dict[str, List[Dict]] = {}
        
        logger.info("数据处理器初始化完成")
    
    def load_data(self, 
                  source: Union[str, Path, Dict],
                  data_type: str = "auto",
                  name: Optional[str] = None,
                  **kwargs) -> str:
        """
        加载数据
        
        Args:
            source: 数据源路径或数据字典
            data_type: 数据类型 (csv, json, excel, sql, auto)
            name: 数据集名称
            **kwargs: 加载参数
            
        Returns:
            数据集名称
        """
        try:
            if isinstance(source, dict):
                # 直接从字典加载
                df = pd.DataFrame(source)
                dataset_name = name or "dict_data"
                
            elif isinstance(source, (str, Path)):
                source = Path(source)
                
                # 自动检测数据类型
                if data_type == "auto":
                    data_type = self._detect_data_type(source)
                
                dataset_name = name or source.stem
                
                # 根据类型加载数据
                if data_type == "csv":
                    df = pd.read_csv(source, **kwargs)
                elif data_type == "json":
                    df = pd.read_json(source, **kwargs)
                elif data_type == "excel":
                    df = pd.read_excel(source, **kwargs)
                elif data_type == "sql":
                    # 需要连接字符串
                    conn_str = kwargs.get("connection_string", "")
                    query = kwargs.get("query", f"SELECT * FROM {source.stem}")
                    conn = sqlite3.connect(conn_str)
                    df = pd.read_sql_query(query, conn)
                    conn.close()
                else:
                    raise ValueError(f"不支持的数据类型: {data_type}")
            
            else:
                raise ValueError("不支持的数据源类型")
            
            # 存储数据集
            self.datasets[dataset_name] = df
            self.preprocessing_history[dataset_name] = []
            
            logger.info(f"数据加载成功: {dataset_name}, 形状: {df.shape}")
            return dataset_name
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def get_data_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            数据集信息字典
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.datasets[dataset_name]
        
        # 基本信息
        info = {
            "name": dataset_name,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage().sum(),
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentage": (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        # 数值列统计
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            info["numeric_stats"] = df[numeric_columns].describe().to_dict()
        
        # 分类列统计
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            info["categorical_stats"] = {}
            for col in categorical_columns:
                info["categorical_stats"][col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": df[col].value_counts().head().to_dict()
                }
        
        return info
    
    def clean_data(self, 
                   dataset_name: str,
                   operations: List[Dict[str, Any]],
                   create_new: bool = False) -> Optional[str]:
        """
        清洗数据
        
        Args:
            dataset_name: 数据集名称
            operations: 清洗操作列表
            create_new: 是否创建新数据集
            
        Returns:
            数据集名称（如果创建新数据集）
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.datasets[dataset_name].copy()
        target_name = f"{dataset_name}_cleaned" if create_new else dataset_name
        
        try:
            for operation in operations:
                op_type = operation["type"]
                params = operation.get("params", {})
                
                if op_type == "drop_duplicates":
                    df = df.drop_duplicates(**params)
                
                elif op_type == "drop_nulls":
                    columns = params.get("columns")
                    df = df.dropna(subset=columns, **{k: v for k, v in params.items() if k != "columns"})
                
                elif op_type == "fill_nulls":
                    method = params.get("method", "mean")
                    columns = params.get("columns", df.columns.tolist())
                    
                    for col in columns:
                        if col in df.columns:
                            if method == "mean" and df[col].dtype in ['int64', 'float64']:
                                df[col].fillna(df[col].mean(), inplace=True)
                            elif method == "median" and df[col].dtype in ['int64', 'float64']:
                                df[col].fillna(df[col].median(), inplace=True)
                            elif method == "mode":
                                df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else df[col].mean(), inplace=True)
                            elif method == "forward":
                                df[col].fillna(method='ffill', inplace=True)
                            elif method == "backward":
                                df[col].fillna(method='bfill', inplace=True)
                            else:
                                df[col].fillna(params.get("value", 0), inplace=True)
                
                elif op_type == "remove_outliers":
                    columns = params.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
                    method = params.get("method", "iqr")
                    
                    for col in columns:
                        if col in df.columns and df[col].dtype in ['int64', 'float64']:
                            if method == "iqr":
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower = Q1 - 1.5 * IQR
                                upper = Q3 + 1.5 * IQR
                                df = df[(df[col] >= lower) & (df[col] <= upper)]
                            elif method == "zscore":
                                from scipy import stats
                                z_scores = np.abs(stats.zscore(df[col]))
                                df = df[z_scores < 3]
                
                elif op_type == "convert_types":
                    type_mapping = params.get("mapping", {})
                    for col, dtype in type_mapping.items():
                        if col in df.columns:
                            df[col] = df[col].astype(dtype)
                
                elif op_type == "rename_columns":
                    mapping = params.get("mapping", {})
                    df.rename(columns=mapping, inplace=True)
                
                elif op_type == "drop_columns":
                    columns = params.get("columns", [])
                    df.drop(columns=columns, inplace=True, errors='ignore')
                
                # 记录操作历史
                self.preprocessing_history[target_name].append({
                    "operation": op_type,
                    "params": params,
                    "timestamp": pd.Timestamp.now()
                })
            
            # 存储处理后的数据
            self.datasets[target_name] = df
            if target_name not in self.preprocessing_history:
                self.preprocessing_history[target_name] = []
            
            logger.info(f"数据清洗完成: {target_name}, 新形状: {df.shape}")
            
            return target_name if create_new else None
            
        except Exception as e:
            logger.error(f"数据清洗失败: {e}")
            raise
    
    def transform_data(self, 
                      dataset_name: str,
                      transformations: List[Dict[str, Any]],
                      create_new: bool = False) -> Optional[str]:
        """
        数据转换
        
        Args:
            dataset_name: 数据集名称
            transformations: 转换操作列表
            create_new: 是否创建新数据集
            
        Returns:
            数据集名称（如果创建新数据集）
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.datasets[dataset_name].copy()
        target_name = f"{dataset_name}_transformed" if create_new else dataset_name
        
        try:
            for transformation in transformations:
                trans_type = transformation["type"]
                params = transformation.get("params", {})
                
                if trans_type == "normalize":
                    columns = params.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
                    scaler = StandardScaler()
                    df[columns] = scaler.fit_transform(df[columns])
                
                elif trans_type == "min_max_scale":
                    columns = params.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
                    scaler = MinMaxScaler()
                    df[columns] = scaler.fit_transform(df[columns])
                
                elif trans_type == "encode_categorical":
                    columns = params.get("columns", df.select_dtypes(include=['object']).columns.tolist())
                    method = params.get("method", "label")
                    
                    for col in columns:
                        if col in df.columns:
                            if method == "label":
                                le = LabelEncoder()
                                df[col] = le.fit_transform(df[col].astype(str))
                            elif method == "onehot":
                                dummies = pd.get_dummies(df[col], prefix=col)
                                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                
                elif trans_type == "create_features":
                    feature_definitions = params.get("features", [])
                    for feature_def in feature_definitions:
                        feature_name = feature_def["name"]
                        expression = feature_def["expression"]
                        # 安全的表达式计算（实际项目中需要更安全的方式）
                        df[feature_name] = df.eval(expression)
                
                elif trans_type == "bin_numeric":
                    column = params.get("column")
                    bins = params.get("bins", 5)
                    labels = params.get("labels")
                    
                    if column in df.columns:
                        df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=labels)
                
                elif trans_type == "log_transform":
                    columns = params.get("columns", [])
                    for col in columns:
                        if col in df.columns and df[col].dtype in ['int64', 'float64']:
                            df[f"{col}_log"] = np.log1p(df[col])
                
                # 记录转换历史
                self.preprocessing_history[target_name].append({
                    "transformation": trans_type,
                    "params": params,
                    "timestamp": pd.Timestamp.now()
                })
            
            # 存储转换后的数据
            self.datasets[target_name] = df
            if target_name not in self.preprocessing_history:
                self.preprocessing_history[target_name] = []
            
            logger.info(f"数据转换完成: {target_name}, 新形状: {df.shape}")
            
            return target_name if create_new else None
            
        except Exception as e:
            logger.error(f"数据转换失败: {e}")
            raise
    
    def analyze_data(self, 
                    dataset_name: str,
                    analysis_type: str = "descriptive",
                    **kwargs) -> Dict[str, Any]:
        """
        数据分析
        
        Args:
            dataset_name: 数据集名称
            analysis_type: 分析类型 (descriptive, correlation, distribution)
            **kwargs: 分析参数
            
        Returns:
            分析结果
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.datasets[dataset_name]
        
        try:
            if analysis_type == "descriptive":
                return self._descriptive_analysis(df, **kwargs)
            elif analysis_type == "correlation":
                return self._correlation_analysis(df, **kwargs)
            elif analysis_type == "distribution":
                return self._distribution_analysis(df, **kwargs)
            elif analysis_type == "missing_values":
                return self._missing_values_analysis(df, **kwargs)
            elif analysis_type == "outliers":
                return self._outliers_analysis(df, **kwargs)
            else:
                raise ValueError(f"不支持的分析类型: {analysis_type}")
                
        except Exception as e:
            logger.error(f"数据分析失败: {e}")
            raise
    
    def create_visualization(self, 
                           dataset_name: str,
                           viz_type: str,
                           **kwargs) -> str:
        """
        创建可视化
        
        Args:
            dataset_name: 数据集名称
            viz_type: 可视化类型
            **kwargs: 可视化参数
            
        Returns:
            可视化HTML字符串
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.datasets[dataset_name]
        
        try:
            if viz_type == "histogram":
                return self._create_histogram(df, **kwargs)
            elif viz_type == "scatter":
                return self._create_scatter(df, **kwargs)
            elif viz_type == "box":
                return self._create_boxplot(df, **kwargs)
            elif viz_type == "correlation_heatmap":
                return self._create_correlation_heatmap(df, **kwargs)
            elif viz_type == "line":
                return self._create_line_plot(df, **kwargs)
            elif viz_type == "bar":
                return self._create_bar_plot(df, **kwargs)
            else:
                raise ValueError(f"不支持的可视化类型: {viz_type}")
                
        except Exception as e:
            logger.error(f"创建可视化失败: {e}")
            raise
    
    def export_data(self, 
                    dataset_name: str,
                    output_path: Union[str, Path],
                    format_type: str = "csv",
                    **kwargs) -> bool:
        """
        导出数据
        
        Args:
            dataset_name: 数据集名称
            output_path: 输出路径
            format_type: 导出格式
            **kwargs: 导出参数
            
        Returns:
            是否成功
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.datasets[dataset_name]
        output_path = Path(output_path)
        
        try:
            if format_type == "csv":
                df.to_csv(output_path, **kwargs)
            elif format_type == "json":
                df.to_json(output_path, **kwargs)
            elif format_type == "excel":
                df.to_excel(output_path, **kwargs)
            elif format_type == "parquet":
                df.to_parquet(output_path, **kwargs)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")
            
            logger.info(f"数据导出成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"数据导出失败: {e}")
            return False
    
    def split_data(self, 
                   dataset_name: str,
                   target_column: str,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[str, str, str, str]:
        """
        分割数据集为训练集和测试集
        
        Args:
            dataset_name: 数据集名称
            target_column: 目标列名
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            (X_train_name, X_test_name, y_train_name, y_test_name)
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.datasets[dataset_name]
        
        if target_column not in df.columns:
            raise ValueError(f"目标列不存在: {target_column}")
        
        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # 存储分割后的数据集
            X_train_name = f"{dataset_name}_X_train"
            X_test_name = f"{dataset_name}_X_test"
            y_train_name = f"{dataset_name}_y_train"
            y_test_name = f"{dataset_name}_y_test"
            
            self.datasets[X_train_name] = X_train
            self.datasets[X_test_name] = X_test
            self.datasets[y_train_name] = y_train.to_frame()
            self.datasets[y_test_name] = y_test.to_frame()
            
            logger.info(f"数据分割完成: 训练集{X_train.shape}, 测试集{X_test.shape}")
            
            return X_train_name, X_test_name, y_train_name, y_test_name
            
        except Exception as e:
            logger.error(f"数据分割失败: {e}")
            raise
    
    def _detect_data_type(self, file_path: Path) -> str:
        """自动检测数据类型"""
        extension = file_path.suffix.lower()
        
        type_mapping = {
            '.csv': 'csv',
            '.json': 'json',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.db': 'sql',
            '.sqlite': 'sql'
        }
        
        return type_mapping.get(extension, 'csv')
    
    def _descriptive_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """描述性分析"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        result = {
            "dataset_shape": df.shape,
            "numeric_summary": df[numeric_columns].describe().to_dict() if len(numeric_columns) > 0 else {},
            "categorical_summary": {},
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict()
        }
        
        # 分类变量摘要
        for col in categorical_columns:
            result["categorical_summary"][col] = {
                "unique_count": df[col].nunique(),
                "top_values": df[col].value_counts().head(10).to_dict()
            }
        
        return result
    
    def _correlation_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """相关性分析"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return {"error": "需要至少2个数值列进行相关性分析"}
        
        correlation_matrix = df[numeric_columns].corr()
        
        # 找出强相关关系
        strong_correlations = []
        threshold = kwargs.get("threshold", 0.7)
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    strong_correlations.append({
                        "variable1": correlation_matrix.columns[i],
                        "variable2": correlation_matrix.columns[j],
                        "correlation": corr_value
                    })
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "strong_correlations": strong_correlations
        }
    
    def _distribution_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """分布分析"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        result = {"distributions": {}}
        
        for col in numeric_columns:
            if col in df.columns:
                data = df[col].dropna()
                result["distributions"][col] = {
                    "mean": data.mean(),
                    "median": data.median(),
                    "std": data.std(),
                    "skewness": data.skew(),
                    "kurtosis": data.kurtosis(),
                    "quantiles": {
                        "25%": data.quantile(0.25),
                        "50%": data.quantile(0.50),
                        "75%": data.quantile(0.75)
                    }
                }
        
        return result
    
    def _missing_values_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """缺失值分析"""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        return {
            "missing_counts": missing_data.to_dict(),
            "missing_percentages": missing_percentage.to_dict(),
            "columns_with_missing": missing_data[missing_data > 0].to_dict(),
            "total_missing": df.isnull().sum().sum(),
            "complete_rows": len(df) - df.isnull().any(axis=1).sum()
        }
    
    def _outliers_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """异常值分析"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_columns:
            if col in df.columns:
                data = df[col].dropna()
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                outliers_info[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": (len(outliers) / len(data)) * 100,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_values": outliers.tolist()[:50]  # 限制返回数量
                }
        
        return outliers_info
    
    def _create_histogram(self, df: pd.DataFrame, **kwargs) -> str:
        """创建直方图"""
        column = kwargs.get("column")
        if not column or column not in df.columns:
            raise ValueError("需要指定有效的列名")
        
        fig = px.histogram(df, x=column, title=f"{column} 分布直方图")
        return fig.to_html()
    
    def _create_scatter(self, df: pd.DataFrame, **kwargs) -> str:
        """创建散点图"""
        x_col = kwargs.get("x_column")
        y_col = kwargs.get("y_column")
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            raise ValueError("需要指定有效的x和y列名")
        
        color_col = kwargs.get("color_column")
        
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col,
            title=f"{x_col} vs {y_col} 散点图"
        )
        return fig.to_html()
    
    def _create_boxplot(self, df: pd.DataFrame, **kwargs) -> str:
        """创建箱线图"""
        column = kwargs.get("column")
        if not column or column not in df.columns:
            raise ValueError("需要指定有效的列名")
        
        fig = px.box(df, y=column, title=f"{column} 箱线图")
        return fig.to_html()
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, **kwargs) -> str:
        """创建相关性热力图"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            raise ValueError("需要至少2个数值列")
        
        corr_matrix = df[numeric_columns].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="相关性热力图"
        )
        return fig.to_html()
    
    def _create_line_plot(self, df: pd.DataFrame, **kwargs) -> str:
        """创建线图"""
        x_col = kwargs.get("x_column")
        y_col = kwargs.get("y_column")
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            raise ValueError("需要指定有效的x和y列名")
        
        fig = px.line(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col} 线图")
        return fig.to_html()
    
    def _create_bar_plot(self, df: pd.DataFrame, **kwargs) -> str:
        """创建柱状图"""
        x_col = kwargs.get("x_column")
        y_col = kwargs.get("y_column")
        
        if not x_col or x_col not in df.columns:
            raise ValueError("需要指定有效的x列名")
        
        if not y_col:
            # 如果没有指定y列，使用计数
            value_counts = df[x_col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"{x_col} 计数柱状图"
            )
        else:
            if y_col not in df.columns:
                raise ValueError("需要指定有效的y列名")
            fig = px.bar(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col} 柱状图")
        
        return fig.to_html()
    
    def get_datasets_list(self) -> List[str]:
        """获取数据集列表"""
        return list(self.datasets.keys())
    
    def get_dataset_sample(self, dataset_name: str, n: int = 5) -> Dict[str, Any]:
        """获取数据集样本"""
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.datasets[dataset_name]
        return {
            "head": df.head(n).to_dict(),
            "tail": df.tail(n).to_dict(),
            "sample": df.sample(min(n, len(df))).to_dict()
        }
    
    def remove_dataset(self, dataset_name: str) -> bool:
        """删除数据集"""
        if dataset_name in self.datasets:
            del self.datasets[dataset_name]
            if dataset_name in self.preprocessing_history:
                del self.preprocessing_history[dataset_name]
            logger.info(f"数据集已删除: {dataset_name}")
            return True
        return False


if __name__ == "__main__":
    # 测试数据处理器
    processor = DataProcessor()
    
    # 创建测试数据
    test_data = {
        'A': [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
        'B': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'C': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        'D': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
    }
    
    try:
        # 加载数据
        dataset_name = processor.load_data(test_data, name="test_data")
        print(f"数据集已加载: {dataset_name}")
        
        # 获取数据信息
        info = processor.get_data_info(dataset_name)
        print(f"数据集信息: {info}")
        
        # 数据清洗
        cleaning_ops = [
            {"type": "fill_nulls", "params": {"method": "mean", "columns": ["A"]}}
        ]
        processor.clean_data(dataset_name, cleaning_ops)
        
        # 描述性分析
        analysis = processor.analyze_data(dataset_name, "descriptive")
        print(f"描述性分析结果: {analysis}")
        
        print("数据处理器测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")