"""
数据分析Agent主程序
整合数据处理、可视化和机器学习功能，提供全面的数据科学解决方案
"""

import sys
import argparse
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

# Third-party imports
from loguru import logger
import gradio as gr

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config

from data_processor import DataProcessor
from ml_engine import MLEngine


class DataAnalysisAgent:
    """数据分析Agent主类"""
    
    def __init__(self):
        """初始化数据分析Agent"""
        self.config = get_config()
        
        # 初始化核心组件
        self.data_processor = DataProcessor()
        self.ml_engine = MLEngine()
        
        # 会话状态
        self.session_state = {
            'loaded_datasets': [],
            'trained_models': [],
            'analysis_history': [],
            'visualization_history': []
        }
        
        logger.info("数据分析Agent初始化完成")
    
    def load_data_from_source(self,
                            source: Union[str, Dict],
                            data_type: str = "auto",
                            name: Optional[str] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        从数据源加载数据
        
        Args:
            source: 数据源（文件路径或数据字典）
            data_type: 数据类型
            name: 数据集名称
            **kwargs: 加载参数
            
        Returns:
            加载结果
        """
        try:
            dataset_name = self.data_processor.load_data(source, data_type, name, **kwargs)
            
            # 获取数据信息
            data_info = self.data_processor.get_data_info(dataset_name)
            
            # 更新会话状态
            if dataset_name not in self.session_state['loaded_datasets']:
                self.session_state['loaded_datasets'].append(dataset_name)
            
            result = {
                'dataset_name': dataset_name,
                'success': True,
                'data_info': data_info,
                'message': f'数据集 {dataset_name} 加载成功'
            }
            
            logger.info(f"数据加载完成: {dataset_name}")
            return result
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'数据加载失败: {e}'
            }
    
    def perform_data_analysis(self,
                            dataset_name: str,
                            analysis_type: str,
                            **kwargs) -> Dict[str, Any]:
        """
        执行数据分析
        
        Args:
            dataset_name: 数据集名称
            analysis_type: 分析类型
            **kwargs: 分析参数
            
        Returns:
            分析结果
        """
        try:
            analysis_result = self.data_processor.analyze_data(
                dataset_name, analysis_type, **kwargs
            )
            
            # 记录分析历史
            self.session_state['analysis_history'].append({
                'dataset_name': dataset_name,
                'analysis_type': analysis_type,
                'parameters': kwargs,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            result = {
                'dataset_name': dataset_name,
                'analysis_type': analysis_type,
                'success': True,
                'result': analysis_result,
                'message': f'{analysis_type} 分析完成'
            }
            
            logger.info(f"数据分析完成: {dataset_name} - {analysis_type}")
            return result
            
        except Exception as e:
            logger.error(f"数据分析失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'数据分析失败: {e}'
            }
    
    def create_data_visualization(self,
                                dataset_name: str,
                                viz_type: str,
                                **kwargs) -> Dict[str, Any]:
        """
        创建数据可视化
        
        Args:
            dataset_name: 数据集名称
            viz_type: 可视化类型
            **kwargs: 可视化参数
            
        Returns:
            可视化结果
        """
        try:
            viz_html = self.data_processor.create_visualization(
                dataset_name, viz_type, **kwargs
            )
            
            # 记录可视化历史
            self.session_state['visualization_history'].append({
                'dataset_name': dataset_name,
                'viz_type': viz_type,
                'parameters': kwargs,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            result = {
                'dataset_name': dataset_name,
                'viz_type': viz_type,
                'success': True,
                'html': viz_html,
                'message': f'{viz_type} 可视化创建完成'
            }
            
            logger.info(f"可视化创建完成: {dataset_name} - {viz_type}")
            return result
            
        except Exception as e:
            logger.error(f"可视化创建失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'可视化创建失败: {e}'
            }
    
    def clean_and_transform_data(self,
                               dataset_name: str,
                               operations: List[Dict[str, Any]],
                               transformations: List[Dict[str, Any]] = None,
                               create_new: bool = False) -> Dict[str, Any]:
        """
        数据清洗和转换
        
        Args:
            dataset_name: 数据集名称
            operations: 清洗操作列表
            transformations: 转换操作列表
            create_new: 是否创建新数据集
            
        Returns:
            处理结果
        """
        try:
            result_name = dataset_name
            
            # 数据清洗
            if operations:
                clean_result = self.data_processor.clean_data(
                    dataset_name, operations, create_new
                )
                if clean_result:
                    result_name = clean_result
                    if result_name not in self.session_state['loaded_datasets']:
                        self.session_state['loaded_datasets'].append(result_name)
            
            # 数据转换
            if transformations:
                transform_result = self.data_processor.transform_data(
                    result_name, transformations, create_new and not operations
                )
                if transform_result:
                    result_name = transform_result
                    if result_name not in self.session_state['loaded_datasets']:
                        self.session_state['loaded_datasets'].append(result_name)
            
            # 获取处理后的数据信息
            data_info = self.data_processor.get_data_info(result_name)
            
            result = {
                'original_dataset': dataset_name,
                'result_dataset': result_name,
                'success': True,
                'data_info': data_info,
                'message': f'数据处理完成: {result_name}'
            }
            
            logger.info(f"数据处理完成: {dataset_name} -> {result_name}")
            return result
            
        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'数据处理失败: {e}'
            }
    
    def train_ml_model(self,
                      model_name: str,
                      algorithm: str,
                      task_type: str,
                      dataset_name: str,
                      target_column: Optional[str] = None,
                      test_size: float = 0.2,
                      **kwargs) -> Dict[str, Any]:
        """
        训练机器学习模型
        
        Args:
            model_name: 模型名称
            algorithm: 算法名称
            task_type: 任务类型
            dataset_name: 数据集名称
            target_column: 目标列名
            test_size: 测试集比例
            **kwargs: 算法参数
            
        Returns:
            训练结果
        """
        try:
            # 获取数据集
            if dataset_name not in self.data_processor.datasets:
                raise ValueError(f"数据集不存在: {dataset_name}")
            
            df = self.data_processor.datasets[dataset_name]
            
            if task_type in ["classification", "regression"]:
                if not target_column:
                    raise ValueError(f"{task_type} 任务需要指定目标列")
                
                if target_column not in df.columns:
                    raise ValueError(f"目标列不存在: {target_column}")
                
                # 分割数据
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # 训练模型
                train_result = self.ml_engine.train_model(
                    model_name, algorithm, task_type, X_train, y_train, **kwargs
                )
                
                # 评估模型
                eval_result = self.ml_engine.evaluate_model(model_name, X_test, y_test)
                
                result = {
                    'model_name': model_name,
                    'algorithm': algorithm,
                    'task_type': task_type,
                    'dataset_name': dataset_name,
                    'success': True,
                    'train_result': train_result,
                    'eval_result': eval_result,
                    'message': f'模型 {model_name} 训练完成'
                }
                
            else:  # clustering
                X = df
                
                train_result = self.ml_engine.train_model(
                    model_name, algorithm, task_type, X, **kwargs
                )
                
                result = {
                    'model_name': model_name,
                    'algorithm': algorithm,
                    'task_type': task_type,
                    'dataset_name': dataset_name,
                    'success': True,
                    'train_result': train_result,
                    'message': f'聚类模型 {model_name} 训练完成'
                }
            
            # 更新会话状态
            if model_name not in self.session_state['trained_models']:
                self.session_state['trained_models'].append(model_name)
            
            logger.info(f"模型训练完成: {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'模型训练失败: {e}'
            }
    
    def predict_with_model(self,
                          model_name: str,
                          dataset_name: str,
                          return_probabilities: bool = False) -> Dict[str, Any]:
        """
        使用模型进行预测
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            return_probabilities: 是否返回概率
            
        Returns:
            预测结果
        """
        try:
            # 获取数据集
            if dataset_name not in self.data_processor.datasets:
                raise ValueError(f"数据集不存在: {dataset_name}")
            
            df = self.data_processor.datasets[dataset_name]
            
            # 执行预测
            prediction_result = self.ml_engine.predict(
                model_name, df, return_probabilities
            )
            
            result = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'success': True,
                'prediction_result': prediction_result,
                'message': f'预测完成: {model_name}'
            }
            
            logger.info(f"预测完成: {model_name} on {dataset_name}")
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'预测失败: {e}'
            }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        datasets_info = []
        
        for dataset_name in self.session_state['loaded_datasets']:
            try:
                info = self.data_processor.get_data_info(dataset_name)
                datasets_info.append(info)
            except Exception as e:
                logger.warning(f"获取数据集信息失败: {dataset_name}, {e}")
        
        return {
            'loaded_datasets': len(self.session_state['loaded_datasets']),
            'datasets_info': datasets_info,
            'trained_models': len(self.session_state['trained_models']),
            'models_info': self.ml_engine.list_models(),
            'analysis_count': len(self.session_state['analysis_history']),
            'visualization_count': len(self.session_state['visualization_history'])
        }
    
    def export_analysis_results(self,
                              output_path: str,
                              include_data: bool = True,
                              include_models: bool = True) -> Dict[str, Any]:
        """
        导出分析结果
        
        Args:
            output_path: 输出路径
            include_data: 是否包含数据
            include_models: 是否包含模型
            
        Returns:
            导出结果
        """
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            exported_files = []
            
            # 导出数据集
            if include_data:
                for dataset_name in self.session_state['loaded_datasets']:
                    try:
                        data_path = output_path / f"{dataset_name}.csv"
                        success = self.data_processor.export_data(
                            dataset_name, data_path, "csv"
                        )
                        if success:
                            exported_files.append(str(data_path))
                    except Exception as e:
                        logger.warning(f"导出数据集失败: {dataset_name}, {e}")
            
            # 导出模型
            if include_models:
                for model_name in self.session_state['trained_models']:
                    try:
                        model_path = output_path / f"{model_name}.pkl"
                        success = self.ml_engine.save_model(model_name, model_path)
                        if success:
                            exported_files.append(str(model_path))
                    except Exception as e:
                        logger.warning(f"导出模型失败: {model_name}, {e}")
            
            # 导出会话摘要
            summary_path = output_path / "analysis_summary.json"
            summary = self.get_data_summary()
            summary['analysis_history'] = self.session_state['analysis_history']
            summary['visualization_history'] = self.session_state['visualization_history']
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            exported_files.append(str(summary_path))
            
            result = {
                'success': True,
                'output_path': str(output_path),
                'exported_files': exported_files,
                'message': f'分析结果已导出到: {output_path}'
            }
            
            logger.info(f"分析结果导出完成: {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"导出分析结果失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'导出失败: {e}'
            }


def create_gradio_interface(agent: DataAnalysisAgent):
    """创建Gradio Web界面"""
    
    def load_data_interface(file_path: str, data_type: str, dataset_name: str):
        """数据加载界面"""
        try:
            if not file_path:
                return "请选择数据文件", "", ""
            
            result = agent.load_data_from_source(
                file_path,
                data_type if data_type != "auto" else "auto",
                dataset_name if dataset_name.strip() else None
            )
            
            if result['success']:
                data_info = json.dumps(result['data_info'], indent=2, ensure_ascii=False)
                sample = agent.data_processor.get_dataset_sample(result['dataset_name'])
                sample_str = json.dumps(sample, indent=2, ensure_ascii=False)
                
                return result['message'], data_info, sample_str
            else:
                return result['message'], "", ""
                
        except Exception as e:
            return f"加载失败: {e}", "", ""
    
    def analyze_data_interface(dataset_name: str, analysis_type: str):
        """数据分析界面"""
        try:
            if not dataset_name:
                return "请选择数据集"
            
            result = agent.perform_data_analysis(dataset_name, analysis_type)
            
            if result['success']:
                return json.dumps(result['result'], indent=2, ensure_ascii=False)
            else:
                return result['message']
                
        except Exception as e:
            return f"分析失败: {e}"
    
    def visualize_data_interface(dataset_name: str, viz_type: str, 
                               column: str, x_column: str, y_column: str):
        """数据可视化界面"""
        try:
            if not dataset_name:
                return "请选择数据集", ""
            
            # 构建可视化参数
            kwargs = {}
            if column:
                kwargs['column'] = column
            if x_column:
                kwargs['x_column'] = x_column
            if y_column:
                kwargs['y_column'] = y_column
            
            result = agent.create_data_visualization(dataset_name, viz_type, **kwargs)
            
            if result['success']:
                return result['message'], result['html']
            else:
                return result['message'], ""
                
        except Exception as e:
            return f"可视化失败: {e}", ""
    
    def clean_data_interface(dataset_name: str, operations_json: str):
        """数据清洗界面"""
        try:
            if not dataset_name:
                return "请选择数据集", ""
            
            if not operations_json.strip():
                return "请输入清洗操作", ""
            
            operations = json.loads(operations_json)
            if not isinstance(operations, list):
                operations = [operations]
            
            result = agent.clean_and_transform_data(dataset_name, operations, create_new=True)
            
            if result['success']:
                info = json.dumps(result['data_info'], indent=2, ensure_ascii=False)
                return result['message'], info
            else:
                return result['message'], ""
                
        except Exception as e:
            return f"清洗失败: {e}", ""
    
    def train_model_interface(model_name: str, algorithm: str, task_type: str,
                            dataset_name: str, target_column: str, params_json: str):
        """模型训练界面"""
        try:
            if not all([model_name, algorithm, task_type, dataset_name]):
                return "请填写所有必需字段"
            
            if task_type in ["classification", "regression"] and not target_column:
                return "监督学习任务需要指定目标列"
            
            # 解析参数
            kwargs = {}
            if params_json.strip():
                kwargs = json.loads(params_json)
            
            result = agent.train_ml_model(
                model_name, algorithm, task_type, dataset_name,
                target_column if target_column else None,
                **kwargs
            )
            
            if result['success']:
                return json.dumps(result, indent=2, ensure_ascii=False, default=str)
            else:
                return result['message']
                
        except Exception as e:
            return f"训练失败: {e}"
    
    def get_summary_interface():
        """获取摘要界面"""
        summary = agent.get_data_summary()
        return json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    
    # 创建Gradio界面
    with gr.Blocks(title="数据分析Agent", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 📊 数据分析Agent")
        gr.Markdown("全面的数据科学解决方案：数据处理、分析、可视化和机器学习")
        
        with gr.Tabs():
            # 数据加载标签页
            with gr.TabItem("📁 数据加载"):
                with gr.Row():
                    with gr.Column():
                        data_file = gr.File(label="选择数据文件", file_types=[".csv", ".json", ".xlsx"])
                        data_type = gr.Dropdown(
                            label="数据类型",
                            choices=["auto", "csv", "json", "excel"],
                            value="auto"
                        )
                        dataset_name_input = gr.Textbox(
                            label="数据集名称（可选）",
                            placeholder="留空则自动生成"
                        )
                        load_btn = gr.Button("📂 加载数据", variant="primary")
                    
                    with gr.Column():
                        load_result = gr.Textbox(
                            label="加载结果",
                            lines=3,
                            interactive=False
                        )
                
                with gr.Row():
                    with gr.Column():
                        data_info = gr.Textbox(
                            label="数据集信息",
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.Column():
                        data_sample = gr.Textbox(
                            label="数据样本",
                            lines=15,
                            interactive=False
                        )
                
                load_btn.click(
                    load_data_interface,
                    inputs=[data_file, data_type, dataset_name_input],
                    outputs=[load_result, data_info, data_sample]
                )
            
            # 数据分析标签页
            with gr.TabItem("🔍 数据分析"):
                with gr.Row():
                    with gr.Column():
                        analyze_dataset = gr.Dropdown(
                            label="选择数据集",
                            choices=[],
                            interactive=True
                        )
                        analyze_type = gr.Dropdown(
                            label="分析类型",
                            choices=["descriptive", "correlation", "distribution", 
                                   "missing_values", "outliers"],
                            value="descriptive"
                        )
                        analyze_btn = gr.Button("🔍 开始分析", variant="primary")
                    
                    with gr.Column():
                        analyze_result = gr.Textbox(
                            label="分析结果",
                            lines=25,
                            interactive=False
                        )
                
                analyze_btn.click(
                    analyze_data_interface,
                    inputs=[analyze_dataset, analyze_type],
                    outputs=[analyze_result]
                )
            
            # 数据可视化标签页
            with gr.TabItem("📈 数据可视化"):
                with gr.Row():
                    with gr.Column():
                        viz_dataset = gr.Dropdown(
                            label="选择数据集",
                            choices=[],
                            interactive=True
                        )
                        viz_type = gr.Dropdown(
                            label="可视化类型",
                            choices=["histogram", "scatter", "box", "correlation_heatmap", 
                                   "line", "bar"],
                            value="histogram"
                        )
                        viz_column = gr.Textbox(
                            label="列名（用于直方图、箱线图）",
                            placeholder="输入列名"
                        )
                        viz_x_column = gr.Textbox(
                            label="X轴列名（用于散点图、线图等）",
                            placeholder="输入X轴列名"
                        )
                        viz_y_column = gr.Textbox(
                            label="Y轴列名（用于散点图、线图等）",
                            placeholder="输入Y轴列名"
                        )
                        viz_btn = gr.Button("📊 创建可视化", variant="primary")
                    
                    with gr.Column():
                        viz_result = gr.Textbox(
                            label="可视化结果",
                            lines=3,
                            interactive=False
                        )
                        viz_display = gr.HTML(label="可视化图表")
                
                viz_btn.click(
                    visualize_data_interface,
                    inputs=[viz_dataset, viz_type, viz_column, viz_x_column, viz_y_column],
                    outputs=[viz_result, viz_display]
                )
            
            # 数据清洗标签页
            with gr.TabItem("🧹 数据清洗"):
                with gr.Row():
                    with gr.Column():
                        clean_dataset = gr.Dropdown(
                            label="选择数据集",
                            choices=[],
                            interactive=True
                        )
                        clean_operations = gr.Textbox(
                            label="清洗操作（JSON格式）",
                            placeholder='[{"type": "fill_nulls", "params": {"method": "mean"}}]',
                            lines=10
                        )
                        clean_btn = gr.Button("🧹 开始清洗", variant="primary")
                    
                    with gr.Column():
                        clean_result = gr.Textbox(
                            label="清洗结果",
                            lines=3,
                            interactive=False
                        )
                        clean_info = gr.Textbox(
                            label="清洗后数据信息",
                            lines=15,
                            interactive=False
                        )
                
                clean_btn.click(
                    clean_data_interface,
                    inputs=[clean_dataset, clean_operations],
                    outputs=[clean_result, clean_info]
                )
            
            # 机器学习标签页
            with gr.TabItem("🤖 机器学习"):
                with gr.Row():
                    with gr.Column():
                        ml_model_name = gr.Textbox(
                            label="模型名称",
                            placeholder="输入模型名称"
                        )
                        ml_algorithm = gr.Dropdown(
                            label="算法",
                            choices=["logistic_regression", "random_forest", "svm", 
                                   "linear_regression", "ridge", "kmeans"],
                            value="random_forest"
                        )
                        ml_task_type = gr.Dropdown(
                            label="任务类型",
                            choices=["classification", "regression", "clustering"],
                            value="classification"
                        )
                        ml_dataset = gr.Dropdown(
                            label="选择数据集",
                            choices=[],
                            interactive=True
                        )
                        ml_target = gr.Textbox(
                            label="目标列（监督学习）",
                            placeholder="输入目标列名"
                        )
                        ml_params = gr.Textbox(
                            label="算法参数（JSON格式）",
                            placeholder='{"n_estimators": 100}',
                            lines=5
                        )
                        ml_train_btn = gr.Button("🚀 训练模型", variant="primary")
                    
                    with gr.Column():
                        ml_result = gr.Textbox(
                            label="训练结果",
                            lines=25,
                            interactive=False
                        )
                
                ml_train_btn.click(
                    train_model_interface,
                    inputs=[ml_model_name, ml_algorithm, ml_task_type, 
                           ml_dataset, ml_target, ml_params],
                    outputs=[ml_result]
                )
            
            # 项目摘要标签页
            with gr.TabItem("📋 项目摘要"):
                with gr.Row():
                    with gr.Column():
                        refresh_summary_btn = gr.Button("🔄 刷新摘要", variant="primary")
                    
                    with gr.Column():
                        summary_display = gr.Textbox(
                            label="项目摘要",
                            lines=25,
                            interactive=False
                        )
                
                refresh_summary_btn.click(
                    get_summary_interface,
                    outputs=[summary_display]
                )
        
        # 定期更新数据集选择框
        def update_dataset_choices():
            choices = agent.session_state['loaded_datasets']
            return (
                gr.Dropdown(choices=choices),
                gr.Dropdown(choices=choices),
                gr.Dropdown(choices=choices),
                gr.Dropdown(choices=choices)
            )
        
        # 可以添加定时更新逻辑
        interface.load(
            update_dataset_choices,
            outputs=[analyze_dataset, viz_dataset, clean_dataset, ml_dataset]
        )
    
    return interface


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据分析Agent")
    parser.add_argument("--mode", choices=['web', 'cli'], default='web',
                       help="运行模式")
    parser.add_argument("--host", default="127.0.0.1", help="服务器主机")
    parser.add_argument("--port", type=int, default=7866, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        # CLI模式（简化实现）
        agent = DataAnalysisAgent()
        print("数据分析Agent CLI模式")
        print("可用命令: load, analyze, visualize, train, summary")
        
        while True:
            try:
                command = input(">>> ").strip().lower()
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'summary':
                    summary = agent.get_data_summary()
                    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
                else:
                    print("命令不支持，请使用 web 模式获得完整功能")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {e}")
    
    else:
        # Web界面模式
        agent = DataAnalysisAgent()
        
        # 创建Web界面
        interface = create_gradio_interface(agent)
        
        # 启动服务器
        logger.info(f"启动数据分析Agent Web界面: http://{args.host}:{args.port}")
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True
        )


if __name__ == "__main__":
    main()