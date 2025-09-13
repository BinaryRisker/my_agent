"""
自学习Agent主程序
整合知识库管理和学习引擎，提供完整的自学习和自我优化功能
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Third-party imports
from loguru import logger
import gradio as gr

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config

from knowledge_base import (
    KnowledgeManager, KnowledgeType, KnowledgeSource, 
    KnowledgeQuery, KnowledgeSearchResult
)
from learning_engine import (
    LearningEngine, ExperienceType, LearningType,
    Experience
)


class SelfLearningAgent:
    """自学习Agent主类"""
    
    def __init__(self, 
                 knowledge_db_path: str = "knowledge.db",
                 memory_size: int = 10000):
        """
        初始化自学习Agent
        
        Args:
            knowledge_db_path: 知识库数据库路径
            memory_size: 经验记忆容量
        """
        self.config = get_config()
        
        # 初始化核心组件
        self.knowledge_manager = KnowledgeManager(knowledge_db_path)
        self.learning_engine = LearningEngine(memory_size)
        
        # Agent状态
        self.agent_state = {
            'active_tasks': [],
            'learning_sessions': [],
            'optimization_history': [],
            'performance_metrics': {
                'knowledge_items': 0,
                'experiences': 0,
                'learned_patterns': 0,
                'success_rate': 0.0
            }
        }
        
        logger.info("自学习Agent初始化完成")
    
    def add_knowledge_from_interaction(self, 
                                     title: str,
                                     content: str,
                                     knowledge_type: str = "experiential",
                                     source: str = "user_interaction",
                                     tags: List[str] = None,
                                     confidence: float = 0.8) -> Dict[str, Any]:
        """
        从交互中添加知识
        
        Args:
            title: 知识标题
            content: 知识内容
            knowledge_type: 知识类型
            source: 知识来源
            tags: 标签
            confidence: 置信度
            
        Returns:
            添加结果
        """
        try:
            # 转换枚举类型
            k_type = KnowledgeType(knowledge_type)
            k_source = KnowledgeSource(source)
            
            # 添加知识到知识库
            knowledge_id = self.knowledge_manager.add_knowledge(
                title=title,
                content=content,
                knowledge_type=k_type,
                source=k_source,
                tags=tags or [],
                confidence=confidence
            )
            
            if knowledge_id:
                # 记录学习经验
                exp_id = self.learning_engine.record_experience(
                    experience_type=ExperienceType.SUCCESS,
                    context={
                        'task': 'knowledge_addition',
                        'title': title,
                        'type': knowledge_type,
                        'source': source
                    },
                    action={
                        'method': 'add_knowledge',
                        'confidence': confidence,
                        'tags_count': len(tags) if tags else 0
                    },
                    result={
                        'knowledge_id': knowledge_id,
                        'status': 'added'
                    },
                    outcome='success',
                    confidence=confidence,
                    reward=1.0
                )
                
                # 更新性能指标
                self._update_performance_metrics()
                
                return {
                    'success': True,
                    'knowledge_id': knowledge_id,
                    'experience_id': exp_id,
                    'message': f'知识项添加成功: {title}'
                }
            else:
                return {
                    'success': False,
                    'message': '知识项添加失败'
                }
                
        except Exception as e:
            logger.error(f"添加知识失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'添加知识失败: {e}'
            }
    
    def query_knowledge_and_learn(self, 
                                 query_text: str,
                                 max_results: int = 5,
                                 similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        查询知识并从查询过程中学习
        
        Args:
            query_text: 查询文本
            max_results: 最大结果数
            similarity_threshold: 相似度阈值
            
        Returns:
            查询结果和学习信息
        """
        try:
            # 构建查询
            query = KnowledgeQuery(
                query_text=query_text,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
            
            # 执行查询
            search_result = self.knowledge_manager.search_knowledge(query)
            
            # 分析查询效果
            query_effectiveness = len(search_result.items) / max_results
            success = len(search_result.items) > 0
            
            # 记录查询经验
            exp_id = self.learning_engine.record_experience(
                experience_type=ExperienceType.SUCCESS if success else ExperienceType.FAILURE,
                context={
                    'task': 'knowledge_query',
                    'query_text': query_text,
                    'max_results': max_results,
                    'similarity_threshold': similarity_threshold
                },
                action={
                    'method': 'search_knowledge',
                    'parameters': {
                        'query_length': len(query_text),
                        'threshold': similarity_threshold
                    }
                },
                result={
                    'items_found': len(search_result.items),
                    'query_time': search_result.query_time,
                    'effectiveness': query_effectiveness
                },
                outcome='success' if success else 'failure',
                confidence=min(query_effectiveness + 0.5, 1.0),
                reward=query_effectiveness
            )
            
            # 格式化结果
            knowledge_items = []
            for item, similarity in zip(search_result.items, search_result.similarities):
                knowledge_items.append({
                    'id': item.id,
                    'title': item.title,
                    'content': item.content,
                    'type': item.knowledge_type.value,
                    'source': item.source.value,
                    'tags': item.tags,
                    'confidence': item.confidence,
                    'similarity': similarity,
                    'usage_count': item.usage_count,
                    'created_at': item.created_at
                })
            
            result = {
                'success': success,
                'query_text': query_text,
                'items_found': len(search_result.items),
                'query_time': search_result.query_time,
                'effectiveness': query_effectiveness,
                'knowledge_items': knowledge_items,
                'experience_id': exp_id,
                'message': f'找到{len(search_result.items)}个相关知识项'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"查询知识失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'查询失败: {e}'
            }
    
    def perform_learning_session(self, 
                               session_name: str = None,
                               min_experiences: int = 10) -> Dict[str, Any]:
        """
        执行学习会话
        
        Args:
            session_name: 会话名称
            min_experiences: 最小经验数量
            
        Returns:
            学习结果
        """
        try:
            session_name = session_name or f"learning_session_{len(self.agent_state['learning_sessions']) + 1}"
            
            # 从经验中学习
            learning_result = self.learning_engine.learn_from_experiences(min_experiences)
            
            if learning_result.get('success', False):
                # 记录学习会话
                learning_session = {
                    'name': session_name,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'experiences_processed': learning_result.get('experiences_processed', 0),
                    'patterns_discovered': learning_result.get('patterns_discovered', 0),
                    'model_created': learning_result.get('model_created', False),
                    'model_id': learning_result.get('model_id', ''),
                    'performance': learning_result.get('model_performance', {})
                }
                
                self.agent_state['learning_sessions'].append(learning_session)
                
                # 更新性能指标
                self._update_performance_metrics()
                
                result = {
                    'success': True,
                    'session_name': session_name,
                    'learning_result': learning_result,
                    'session_info': learning_session,
                    'message': f'学习会话 {session_name} 完成'
                }
                
                # 如果创建了新模型，记录优化经验
                if learning_result.get('model_created', False):
                    self.learning_engine.record_experience(
                        experience_type=ExperienceType.OPTIMIZATION,
                        context={
                            'task': 'model_training',
                            'session_name': session_name,
                            'data_size': learning_result.get('training_data_size', 0)
                        },
                        action={
                            'method': 'create_learning_model',
                            'model_type': 'random_forest',
                            'learning_type': 'supervised'
                        },
                        result={
                            'model_id': learning_result.get('model_id', ''),
                            'performance': learning_result.get('model_performance', {})
                        },
                        outcome='success',
                        confidence=0.9,
                        reward=2.0
                    )
                
                return result
            else:
                return {
                    'success': False,
                    'session_name': session_name,
                    'learning_result': learning_result,
                    'message': learning_result.get('message', '学习失败')
                }
                
        except Exception as e:
            logger.error(f"学习会话失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'学习会话失败: {e}'
            }
    
    def optimize_agent_performance(self) -> Dict[str, Any]:
        """
        优化Agent性能
        
        Returns:
            优化结果
        """
        try:
            # 获取优化建议
            optimization_result = self.learning_engine.optimize_strategy({
                'current_performance': self.agent_state['performance_metrics'],
                'active_tasks': len(self.agent_state['active_tasks']),
                'learning_sessions': len(self.agent_state['learning_sessions'])
            })
            
            if optimization_result.get('success', False):
                # 记录优化历史
                optimization_record = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'analyzed_experiences': optimization_result.get('analyzed_experiences', 0),
                    'recommendations': optimization_result.get('recommendations', []),
                    'success_rate': optimization_result.get('success_rate', 0.0)
                }
                
                self.agent_state['optimization_history'].append(optimization_record)
                
                # 记录优化经验
                self.learning_engine.record_experience(
                    experience_type=ExperienceType.OPTIMIZATION,
                    context={
                        'task': 'performance_optimization',
                        'current_metrics': self.agent_state['performance_metrics']
                    },
                    action={
                        'method': 'optimize_strategy',
                        'analyzed_count': optimization_result.get('analyzed_experiences', 0)
                    },
                    result={
                        'recommendations_count': len(optimization_result.get('recommendations', [])),
                        'success_rate': optimization_result.get('success_rate', 0.0)
                    },
                    outcome='success',
                    confidence=0.8,
                    reward=1.5
                )
                
                result = {
                    'success': True,
                    'optimization_result': optimization_result,
                    'optimization_record': optimization_record,
                    'message': f'生成了{len(optimization_result.get("recommendations", []))}个优化建议'
                }
                
                return result
            else:
                return {
                    'success': False,
                    'optimization_result': optimization_result,
                    'message': optimization_result.get('message', '优化失败')
                }
                
        except Exception as e:
            logger.error(f"性能优化失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'性能优化失败: {e}'
            }
    
    def predict_action_outcome(self, 
                              context: Dict[str, Any],
                              action: Dict[str, Any],
                              model_id: str = None) -> Dict[str, Any]:
        """
        预测行动结果
        
        Args:
            context: 上下文
            action: 行动
            model_id: 使用的模型ID（None则使用最新模型）
            
        Returns:
            预测结果
        """
        try:
            # 如果没有指定模型，查找最新的模型
            if not model_id:
                learning_stats = self.learning_engine.get_learning_statistics()
                if learning_stats.get('learning_models', {}).get('total_count', 0) == 0:
                    return {
                        'success': False,
                        'message': '没有可用的学习模型，请先执行学习会话'
                    }
                
                # 简化：使用第一个可用模型（实际应该选择最优模型）
                for model in self.learning_engine.adaptive_learning.learning_models.values():
                    model_id = model.id
                    break
            
            # 执行预测
            prediction = self.learning_engine.predict_outcome(model_id, context, action)
            
            if prediction:
                # 记录预测经验
                self.learning_engine.record_experience(
                    experience_type=ExperienceType.PATTERN,
                    context={
                        'task': 'action_prediction',
                        'model_id': model_id,
                        'context_complexity': len(str(context)),
                        'action_complexity': len(str(action))
                    },
                    action={
                        'method': 'predict_outcome',
                        'model_type': 'supervised'
                    },
                    result={
                        'prediction': prediction.get('prediction', 0),
                        'confidence': prediction.get('confidence', 0.0)
                    },
                    outcome='success',
                    confidence=prediction.get('confidence', 0.0),
                    reward=0.8
                )
                
                return {
                    'success': True,
                    'model_id': model_id,
                    'prediction': prediction,
                    'message': '预测完成'
                }
            else:
                return {
                    'success': False,
                    'model_id': model_id,
                    'message': '预测失败'
                }
                
        except Exception as e:
            logger.error(f"预测行动结果失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'预测失败: {e}'
            }
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        try:
            # 获取知识库统计
            knowledge_stats = self.knowledge_manager.get_statistics()
            
            # 获取学习引擎统计
            learning_stats = self.learning_engine.get_learning_statistics()
            
            # 计算成功率
            exp_stats = learning_stats.get('experience_memory', {})
            outcome_dist = exp_stats.get('outcome_distribution', {})
            total_outcomes = sum(outcome_dist.values())
            success_rate = outcome_dist.get('success', 0) / total_outcomes if total_outcomes > 0 else 0.0
            
            # 更新指标
            self.agent_state['performance_metrics'].update({
                'knowledge_items': knowledge_stats.get('total_count', 0),
                'experiences': exp_stats.get('total_count', 0),
                'learned_patterns': len(self.agent_state['learning_sessions']),
                'success_rate': round(success_rate, 3),
                'average_confidence': knowledge_stats.get('average_confidence', 0.0),
                'learning_models': learning_stats.get('learning_models', {}).get('total_count', 0)
            })
            
        except Exception as e:
            logger.warning(f"更新性能指标失败: {e}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        try:
            # 更新性能指标
            self._update_performance_metrics()
            
            # 获取详细统计
            knowledge_stats = self.knowledge_manager.get_statistics()
            learning_stats = self.learning_engine.get_learning_statistics()
            
            return {
                'agent_info': {
                    'type': 'self_learning_agent',
                    'version': '1.0.0',
                    'status': 'active'
                },
                'performance_metrics': self.agent_state['performance_metrics'],
                'knowledge_base': knowledge_stats,
                'learning_engine': learning_stats,
                'session_history': {
                    'total_learning_sessions': len(self.agent_state['learning_sessions']),
                    'recent_sessions': self.agent_state['learning_sessions'][-3:],  # 最近3次
                    'optimization_attempts': len(self.agent_state['optimization_history']),
                    'recent_optimizations': self.agent_state['optimization_history'][-2:]  # 最近2次
                },
                'current_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取Agent状态失败: {e}")
            return {
                'error': str(e),
                'message': f'获取状态失败: {e}'
            }
    
    def export_agent_data(self, output_dir: str) -> Dict[str, Any]:
        """
        导出Agent数据
        
        Args:
            output_dir: 输出目录
            
        Returns:
            导出结果
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            exported_files = []
            
            # 导出知识库
            knowledge_export_path = output_path / "knowledge_base.json"
            knowledge_success = self.knowledge_manager.export_knowledge(str(knowledge_export_path))
            if knowledge_success:
                exported_files.append(str(knowledge_export_path))
            
            # 导出学习数据
            learning_export_path = output_path / "learning_data.json"
            learning_success = self.learning_engine.export_learning_data(str(learning_export_path))
            if learning_success:
                exported_files.append(str(learning_export_path))
            
            # 导出Agent状态
            agent_status_path = output_path / "agent_status.json"
            agent_status = self.get_agent_status()
            with open(agent_status_path, 'w', encoding='utf-8') as f:
                json.dump(agent_status, f, indent=2, ensure_ascii=False, default=str)
            exported_files.append(str(agent_status_path))
            
            return {
                'success': True,
                'output_directory': str(output_path),
                'exported_files': exported_files,
                'knowledge_export': knowledge_success,
                'learning_export': learning_success,
                'message': f'Agent数据已导出到: {output_path}'
            }
            
        except Exception as e:
            logger.error(f"导出Agent数据失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'导出失败: {e}'
            }


def create_gradio_interface(agent: SelfLearningAgent):
    """创建Gradio Web界面"""
    
    def add_knowledge_interface(title: str, content: str, knowledge_type: str,
                              source: str, tags: str, confidence: float):
        """添加知识界面"""
        try:
            if not title or not content:
                return "请填写标题和内容"
            
            tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
            
            result = agent.add_knowledge_from_interaction(
                title=title,
                content=content,
                knowledge_type=knowledge_type,
                source=source,
                tags=tags_list,
                confidence=confidence
            )
            
            if result['success']:
                return f"✅ {result['message']}\n知识ID: {result['knowledge_id']}"
            else:
                return f"❌ {result['message']}"
                
        except Exception as e:
            return f"❌ 添加失败: {e}"
    
    def query_knowledge_interface(query_text: str, max_results: int, similarity_threshold: float):
        """查询知识界面"""
        try:
            if not query_text:
                return "请输入查询文本", ""
            
            result = agent.query_knowledge_and_learn(
                query_text=query_text,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
            
            if result['success']:
                message = f"✅ {result['message']}\n查询时间: {result['query_time']:.3f}秒"
                
                # 格式化知识项
                items_text = ""
                for item in result['knowledge_items']:
                    items_text += f"\n📝 **{item['title']}** (相似度: {item['similarity']:.3f})\n"
                    items_text += f"类型: {item['type']} | 来源: {item['source']}\n"
                    items_text += f"内容: {item['content'][:200]}...\n"
                    items_text += f"标签: {', '.join(item['tags'])}\n"
                    items_text += "---\n"
                
                return message, items_text
            else:
                return f"❌ {result['message']}", ""
                
        except Exception as e:
            return f"❌ 查询失败: {e}", ""
    
    def learning_session_interface(session_name: str, min_experiences: int):
        """学习会话界面"""
        try:
            result = agent.perform_learning_session(
                session_name=session_name if session_name else None,
                min_experiences=min_experiences
            )
            
            if result['success']:
                session_info = result['session_info']
                return f"""✅ {result['message']}
                
📊 会话统计:
- 处理经验数: {session_info['experiences_processed']}
- 发现模式数: {session_info['patterns_discovered']}
- 创建模型: {'是' if session_info['model_created'] else '否'}
- 模型ID: {session_info.get('model_id', 'N/A')}

📈 学习结果: {json.dumps(result['learning_result'], indent=2, ensure_ascii=False)}"""
            else:
                return f"❌ {result['message']}"
                
        except Exception as e:
            return f"❌ 学习会话失败: {e}"
    
    def optimization_interface():
        """优化界面"""
        try:
            result = agent.optimize_agent_performance()
            
            if result['success']:
                recommendations = result['optimization_result']['recommendations']
                rec_text = ""
                for rec in recommendations:
                    rec_text += f"🎯 **{rec['type'].upper()}** ({rec['priority']})\n"
                    rec_text += f"   {rec['suggestion']}\n\n"
                
                return f"""✅ {result['message']}
                
📊 分析统计:
- 分析经验数: {result['optimization_result']['analyzed_experiences']}
- 成功率: {result['optimization_result']['success_rate']:.3f}

💡 优化建议:
{rec_text}"""
            else:
                return f"❌ {result['message']}"
                
        except Exception as e:
            return f"❌ 优化失败: {e}"
    
    def prediction_interface(context_json: str, action_json: str):
        """预测界面"""
        try:
            if not context_json or not action_json:
                return "请输入上下文和行动的JSON格式数据"
            
            context = json.loads(context_json)
            action = json.loads(action_json)
            
            result = agent.predict_action_outcome(context, action)
            
            if result['success']:
                prediction = result['prediction']
                return f"""✅ {result['message']}

🎯 预测结果:
- 模型ID: {result['model_id']}
- 预测值: {prediction.get('prediction', 'N/A')}
- 置信度: {prediction.get('confidence', 0.0):.3f}

📊 详细信息: {json.dumps(prediction, indent=2, ensure_ascii=False)}"""
            else:
                return f"❌ {result['message']}"
                
        except json.JSONDecodeError as e:
            return f"❌ JSON格式错误: {e}"
        except Exception as e:
            return f"❌ 预测失败: {e}"
    
    def status_interface():
        """状态界面"""
        status = agent.get_agent_status()
        return json.dumps(status, indent=2, ensure_ascii=False, default=str)
    
    # 创建Gradio界面
    with gr.Blocks(title="自学习Agent", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🧠 自学习Agent")
        gr.Markdown("具备知识管理、经验学习和自我优化能力的智能Agent")
        
        with gr.Tabs():
            # 知识管理标签页
            with gr.TabItem("📚 知识管理"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 添加知识")
                        knowledge_title = gr.Textbox(label="知识标题", placeholder="输入知识标题")
                        knowledge_content = gr.Textbox(
                            label="知识内容", 
                            placeholder="输入详细的知识内容",
                            lines=5
                        )
                        knowledge_type = gr.Dropdown(
                            label="知识类型",
                            choices=["factual", "procedural", "experiential", "rule", "pattern", "domain"],
                            value="experiential"
                        )
                        knowledge_source = gr.Dropdown(
                            label="知识来源",
                            choices=["user_input", "system_learning", "external_data", "agent_experience"],
                            value="user_input"
                        )
                        knowledge_tags = gr.Textbox(label="标签", placeholder="用逗号分隔多个标签")
                        knowledge_confidence = gr.Slider(
                            label="置信度",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.8,
                            step=0.1
                        )
                        add_knowledge_btn = gr.Button("➕ 添加知识", variant="primary")
                    
                    with gr.Column():
                        add_knowledge_result = gr.Textbox(
                            label="添加结果",
                            lines=5,
                            interactive=False
                        )
                
                add_knowledge_btn.click(
                    add_knowledge_interface,
                    inputs=[knowledge_title, knowledge_content, knowledge_type, 
                           knowledge_source, knowledge_tags, knowledge_confidence],
                    outputs=[add_knowledge_result]
                )
                
                gr.Markdown("### 查询知识")
                with gr.Row():
                    with gr.Column():
                        query_text = gr.Textbox(label="查询文本", placeholder="输入要查询的内容")
                        with gr.Row():
                            max_results = gr.Number(label="最大结果数", value=5)
                            similarity_threshold = gr.Slider(
                                label="相似度阈值",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.3,
                                step=0.1
                            )
                        query_btn = gr.Button("🔍 查询知识", variant="primary")
                    
                    with gr.Column():
                        query_result = gr.Textbox(
                            label="查询结果",
                            lines=3,
                            interactive=False
                        )
                
                query_knowledge_items = gr.Textbox(
                    label="找到的知识项",
                    lines=15,
                    interactive=False
                )
                
                query_btn.click(
                    query_knowledge_interface,
                    inputs=[query_text, max_results, similarity_threshold],
                    outputs=[query_result, query_knowledge_items]
                )
            
            # 学习引擎标签页
            with gr.TabItem("🎓 学习引擎"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 执行学习会话")
                        session_name = gr.Textbox(
                            label="会话名称（可选）",
                            placeholder="留空则自动生成"
                        )
                        min_experiences = gr.Number(
                            label="最小经验数量",
                            value=10,
                            minimum=1
                        )
                        learning_btn = gr.Button("🚀 开始学习", variant="primary")
                    
                    with gr.Column():
                        learning_result = gr.Textbox(
                            label="学习结果",
                            lines=15,
                            interactive=False
                        )
                
                learning_btn.click(
                    learning_session_interface,
                    inputs=[session_name, min_experiences],
                    outputs=[learning_result]
                )
            
            # 自我优化标签页
            with gr.TabItem("⚡ 自我优化"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 性能优化")
                        gr.Markdown("分析历史经验，生成优化建议")
                        optimization_btn = gr.Button("🎯 执行优化", variant="primary")
                    
                    with gr.Column():
                        optimization_result = gr.Textbox(
                            label="优化结果",
                            lines=20,
                            interactive=False
                        )
                
                optimization_btn.click(
                    optimization_interface,
                    outputs=[optimization_result]
                )
            
            # 预测功能标签页
            with gr.TabItem("🔮 行动预测"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 预测行动结果")
                        context_input = gr.Textbox(
                            label="上下文（JSON格式）",
                            placeholder='{"task": "example", "complexity": 5}',
                            lines=5
                        )
                        action_input = gr.Textbox(
                            label="行动（JSON格式）",
                            placeholder='{"method": "process", "parameters": {"param1": "value"}}',
                            lines=5
                        )
                        predict_btn = gr.Button("🎯 预测结果", variant="primary")
                    
                    with gr.Column():
                        prediction_result = gr.Textbox(
                            label="预测结果",
                            lines=15,
                            interactive=False
                        )
                
                predict_btn.click(
                    prediction_interface,
                    inputs=[context_input, action_input],
                    outputs=[prediction_result]
                )
            
            # Agent状态标签页
            with gr.TabItem("📊 Agent状态"):
                with gr.Row():
                    with gr.Column():
                        refresh_status_btn = gr.Button("🔄 刷新状态", variant="primary")
                    
                    with gr.Column():
                        agent_status = gr.Textbox(
                            label="Agent状态",
                            lines=25,
                            interactive=False
                        )
                
                refresh_status_btn.click(
                    status_interface,
                    outputs=[agent_status]
                )
        
        # 初始加载状态
        interface.load(status_interface, outputs=[agent_status])
    
    return interface


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自学习Agent")
    parser.add_argument("--mode", choices=['web', 'cli'], default='web',
                       help="运行模式")
    parser.add_argument("--host", default="127.0.0.1", help="服务器主机")
    parser.add_argument("--port", type=int, default=7867, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    parser.add_argument("--db-path", default="knowledge.db", help="知识库数据库路径")
    parser.add_argument("--memory-size", type=int, default=10000, help="经验记忆容量")
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        # CLI模式（简化实现）
        agent = SelfLearningAgent(args.db_path, args.memory_size)
        print("自学习Agent CLI模式")
        print("可用命令: add, query, learn, optimize, predict, status")
        
        while True:
            try:
                command = input(">>> ").strip().lower()
                if command in ['quit', 'exit']:
                    break
                elif command == 'status':
                    status = agent.get_agent_status()
                    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))
                elif command == 'learn':
                    result = agent.perform_learning_session()
                    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
                elif command == 'optimize':
                    result = agent.optimize_agent_performance()
                    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
                else:
                    print("命令不支持，请使用 web 模式获得完整功能")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {e}")
    
    else:
        # Web界面模式
        agent = SelfLearningAgent(args.db_path, args.memory_size)
        
        # 创建Web界面
        interface = create_gradio_interface(agent)
        
        # 启动服务器
        logger.info(f"启动自学习Agent Web界面: http://{args.host}:{args.port}")
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True
        )


if __name__ == "__main__":
    main()