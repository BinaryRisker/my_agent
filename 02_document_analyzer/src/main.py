"""
文档分析Agent主程序
RAG系统的入口点
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# Project imports
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from retrieval_chain import RetrievalChainManager
from qa_system import QASystem

# Third-party imports
from loguru import logger
import gradio as gr
import json

# Common imports
from common.config import get_config


class DocumentAnalyzerAgent:
    """文档分析Agent主类"""
    
    def __init__(self):
        """初始化文档分析Agent"""
        self.config = get_config()
        self._setup_logging()
        
        # 初始化组件
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = None
        self.retrieval_chain_manager = None
        self.qa_system = None
        self.llm = None
        
        # 当前加载的文档
        self.current_documents = []
        self.current_collection = "default"
        
        logger.info("文档分析Agent初始化完成")
    
    def _setup_logging(self):
        """设置日志配置"""
        logger.remove()
        
        # 控制台输出
        logger.add(
            sys.stderr,
            level=self.config.logging.level,
            format=self.config.logging.format,
            colorize=True
        )
        
        # 文件输出
        log_path = Path(__file__).parent.parent / "logs" / "document_analyzer.log"
        log_path.parent.mkdir(exist_ok=True)
        
        logger.add(
            str(log_path),
            level=self.config.logging.level,
            format=self.config.logging.format,
            rotation=self.config.logging.rotation,
            retention=self.config.logging.retention,
            encoding="utf-8"
        )
        
        logger.info("日志系统初始化完成")
    
    def initialize_llm(self):
        """初始化语言模型"""
        try:
            if not self.config.llm.api_key or self.config.llm.api_key == "your_openai_api_key_here":
                raise ValueError("请在.env文件中配置有效的OPENAI_API_KEY")
            
            self.llm = ChatOpenAI(
                model_name=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                openai_api_key=self.config.llm.api_key,
                openai_api_base=self.config.llm.api_base,
                request_timeout=self.config.llm.timeout
            )
            
            logger.info(f"LLM初始化成功: {self.config.llm.model}")
            
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            raise
    
    def load_documents(self, 
                      file_paths: Optional[List[str]] = None,
                      directory_path: Optional[str] = None,
                      recursive: bool = True) -> int:
        """
        加载文档
        
        Args:
            file_paths: 文件路径列表
            directory_path: 目录路径
            recursive: 是否递归加载
            
        Returns:
            加载的文档数量
        """
        try:
            documents = []
            
            if file_paths:
                logger.info(f"从文件列表加载文档: {len(file_paths)} 个文件")
                documents = self.document_processor.load_documents(file_paths)
            
            elif directory_path:
                logger.info(f"从目录加载文档: {directory_path}")
                documents = self.document_processor.load_directory(
                    directory_path, 
                    recursive=recursive
                )
            
            else:
                raise ValueError("请提供文件路径列表或目录路径")
            
            if not documents:
                logger.warning("未加载到任何文档")
                return 0
            
            # 分割文档
            logger.info("开始分割文档...")
            self.current_documents = self.document_processor.split_documents(
                documents, 
                method='recursive',
                chunk_size=1000,
                chunk_overlap=200
            )
            
            logger.info(f"文档加载完成，共 {len(self.current_documents)} 个片段")
            return len(self.current_documents)
            
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            raise
    
    def create_vector_store(self, 
                           collection_name: str = "default",
                           store_type: str = "chroma") -> bool:
        """
        创建向量存储
        
        Args:
            collection_name: 集合名称
            store_type: 存储类型
            
        Returns:
            是否成功
        """
        try:
            if not self.current_documents:
                raise ValueError("没有加载的文档，请先加载文档")
            
            logger.info(f"创建向量存储: {store_type}, 集合: {collection_name}")
            
            # 初始化向量存储管理器
            self.vector_store_manager = VectorStoreManager(
                store_type=store_type,
                persist_directory=str(Path(__file__).parent.parent / "embeddings")
            )
            
            # 创建向量存储
            self.vector_store_manager.create_store(
                self.current_documents, 
                collection_name
            )
            
            self.current_collection = collection_name
            
            logger.info("向量存储创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建向量存储失败: {e}")
            return False
    
    def load_vector_store(self, 
                         collection_name: str = "default",
                         store_type: str = "chroma") -> bool:
        """
        加载现有向量存储
        
        Args:
            collection_name: 集合名称
            store_type: 存储类型
            
        Returns:
            是否成功
        """
        try:
            logger.info(f"加载向量存储: {store_type}, 集合: {collection_name}")
            
            # 初始化向量存储管理器
            self.vector_store_manager = VectorStoreManager(
                store_type=store_type,
                persist_directory=str(Path(__file__).parent.parent / "embeddings")
            )
            
            # 加载向量存储
            success = self.vector_store_manager.load_store(collection_name)
            
            if success:
                self.current_collection = collection_name
                logger.info("向量存储加载成功")
            else:
                logger.warning("向量存储加载失败")
            
            return success
            
        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")
            return False
    
    def initialize_qa_system(self):
        """初始化问答系统"""
        try:
            if self.llm is None:
                self.initialize_llm()
            
            if self.vector_store_manager is None:
                raise ValueError("向量存储未初始化")
            
            logger.info("初始化问答系统...")
            
            # 创建检索器
            retriever = self.vector_store_manager.vector_store.as_retriever(
                search_kwargs={"k": 4}
            )
            
            # 创建检索链管理器
            self.retrieval_chain_manager = RetrievalChainManager(
                retriever=retriever,
                llm=self.llm,
                chain_type="stuff"
            )
            
            # 创建问答系统
            self.qa_system = QASystem(
                retrieval_chain_manager=self.retrieval_chain_manager,
                vector_store_manager=self.vector_store_manager
            )
            
            logger.info("问答系统初始化成功")
            
        except Exception as e:
            logger.error(f"问答系统初始化失败: {e}")
            raise
    
    def ask_question(self, question: str, use_conversation: bool = False) -> dict:
        """
        询问问题
        
        Args:
            question: 问题
            use_conversation: 是否使用对话模式
            
        Returns:
            答案字典
        """
        try:
            if self.qa_system is None:
                self.initialize_qa_system()
            
            logger.info(f"处理问题: {question}")
            
            if use_conversation:
                result = self.qa_system.conversational_ask(question)
            else:
                result = self.qa_system.ask_question(question)
            
            logger.info("问题处理完成")
            return result
            
        except Exception as e:
            logger.error(f"问题处理失败: {e}")
            return {
                "question": question,
                "answer": f"处理失败: {str(e)}",
                "source_documents": [],
                "error": True
            }
    
    def get_system_info(self) -> dict:
        """获取系统信息"""
        info = {
            "loaded_documents": len(self.current_documents),
            "current_collection": self.current_collection,
            "llm_initialized": self.llm is not None,
            "vector_store_initialized": self.vector_store_manager is not None,
            "qa_system_initialized": self.qa_system is not None,
        }
        
        if self.vector_store_manager:
            info.update(self.vector_store_manager.get_store_info())
        
        return info


def create_gradio_interface(agent: DocumentAnalyzerAgent):
    """创建Gradio界面"""
    
    def load_documents_ui(files, directory, recursive):
        """文档加载界面函数"""
        try:
            file_paths = None
            if files:
                file_paths = [f.name for f in files]
            
            count = agent.load_documents(
                file_paths=file_paths,
                directory_path=directory if directory else None,
                recursive=recursive
            )
            
            return f"✅ 成功加载 {count} 个文档片段"
            
        except Exception as e:
            return f"❌ 加载失败: {str(e)}"
    
    def create_vector_store_ui(collection_name, store_type):
        """创建向量存储界面函数"""
        try:
            success = agent.create_vector_store(collection_name, store_type)
            return "✅ 向量存储创建成功" if success else "❌ 向量存储创建失败"
            
        except Exception as e:
            return f"❌ 创建失败: {str(e)}"
    
    def ask_question_ui(question, use_conversation):
        """问答界面函数"""
        try:
            result = agent.ask_question(question, use_conversation)
            
            # 格式化输出
            output = f"**问题:** {result['question']}\\n\\n"
            output += f"**答案:** {result['answer']}\\n\\n"
            
            if result.get('source_documents'):
                output += "**参考来源:**\\n"
                for i, doc in enumerate(result['source_documents'][:3]):
                    output += f"{i+1}. {doc['content'][:200]}...\\n"
                    if doc.get('metadata', {}).get('source_file'):
                        output += f"   来源: {doc['metadata']['source_file']}\\n"
                    output += "\\n"
            
            return output
            
        except Exception as e:
            return f"❌ 处理失败: {str(e)}"
    
    def get_system_info_ui():
        """系统信息界面函数"""
        try:
            info = agent.get_system_info()
            output = "## 系统状态\\n\\n"
            
            for key, value in info.items():
                output += f"- **{key}**: {value}\\n"
            
            return output
            
        except Exception as e:
            return f"❌ 获取信息失败: {str(e)}"
    
    # 创建Gradio界面
    with gr.Blocks(title="文档分析Agent - RAG系统") as interface:
        gr.Markdown("# 📄 文档分析Agent - RAG系统")
        gr.Markdown("*基于LangChain的检索增强生成系统*")
        
        with gr.Tabs():
            # 文档加载标签页
            with gr.TabItem("📁 文档加载"):
                gr.Markdown("### 加载文档到系统")
                
                with gr.Row():
                    with gr.Column():
                        files_input = gr.File(
                            label="选择文件",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md"]
                        )
                        directory_input = gr.Textbox(
                            label="或输入目录路径",
                            placeholder="例如: ./documents"
                        )
                        recursive_checkbox = gr.Checkbox(
                            label="递归加载子目录",
                            value=True
                        )
                        load_btn = gr.Button("🚀 加载文档", variant="primary")
                    
                    with gr.Column():
                        load_output = gr.Textbox(
                            label="加载结果",
                            lines=5
                        )
                
                load_btn.click(
                    load_documents_ui,
                    inputs=[files_input, directory_input, recursive_checkbox],
                    outputs=[load_output]
                )
            
            # 向量存储标签页
            with gr.TabItem("🗄️ 向量存储"):
                gr.Markdown("### 创建和管理向量存储")
                
                with gr.Row():
                    collection_input = gr.Textbox(
                        label="集合名称",
                        value="default",
                        placeholder="输入集合名称"
                    )
                    store_type_dropdown = gr.Dropdown(
                        choices=["chroma", "faiss"],
                        value="chroma",
                        label="存储类型"
                    )
                
                create_store_btn = gr.Button("🔨 创建向量存储", variant="primary")
                store_output = gr.Textbox(label="操作结果", lines=3)
                
                create_store_btn.click(
                    create_vector_store_ui,
                    inputs=[collection_input, store_type_dropdown],
                    outputs=[store_output]
                )
            
            # 问答标签页
            with gr.TabItem("💬 智能问答"):
                gr.Markdown("### 基于文档的智能问答")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(
                            label="输入问题",
                            lines=2,
                            placeholder="例如: 文档的主要内容是什么？"
                        )
                        conversation_checkbox = gr.Checkbox(
                            label="使用对话模式（保持上下文）",
                            value=False
                        )
                        ask_btn = gr.Button("🤔 提问", variant="primary")
                    
                    with gr.Column(scale=4):
                        answer_output = gr.Markdown(label="回答")
                
                ask_btn.click(
                    ask_question_ui,
                    inputs=[question_input, conversation_checkbox],
                    outputs=[answer_output]
                )
            
            # 系统状态标签页
            with gr.TabItem("📊 系统状态"):
                gr.Markdown("### 系统信息和状态")
                
                info_btn = gr.Button("🔄 刷新信息", variant="secondary")
                info_output = gr.Markdown()
                
                info_btn.click(
                    get_system_info_ui,
                    outputs=[info_output]
                )
                
                # 页面加载时自动获取信息
                interface.load(get_system_info_ui, outputs=[info_output])
    
    return interface


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文档分析Agent - RAG系统")
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="cli",
        help="运行模式"
    )
    parser.add_argument(
        "--load-docs",
        type=str,
        help="要加载的文档目录路径"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="向量存储集合名称"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web模式主机地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Web模式端口"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建Agent实例
        agent = DocumentAnalyzerAgent()
        
        if args.mode == "web":
            # Web模式
            logger.info("启动Web界面...")
            interface = create_gradio_interface(agent)
            interface.launch(
                server_name=args.host,
                server_port=args.port,
                share=False
            )
        
        else:
            # CLI模式
            logger.info("启动CLI模式...")
            print("🤖 文档分析Agent - RAG系统")
            print("=" * 50)
            
            # 如果指定了文档目录，自动加载
            if args.load_docs:
                print(f"📁 加载文档目录: {args.load_docs}")
                count = agent.load_documents(directory_path=args.load_docs)
                print(f"✅ 加载了 {count} 个文档片段")
                
                print("🔨 创建向量存储...")
                success = agent.create_vector_store(args.collection)
                if success:
                    print("✅ 向量存储创建成功")
                    
                    print("🤖 初始化问答系统...")
                    agent.initialize_qa_system()
                    print("✅ 系统初始化完成")
                    
                    # 进入问答循环
                    print("\\n💬 开始问答 (输入 'quit' 退出):")
                    while True:
                        try:
                            question = input("\\n问题: ").strip()
                            if question.lower() in ['quit', 'exit', '退出']:
                                break
                            
                            if not question:
                                continue
                            
                            result = agent.ask_question(question)
                            print(f"\\n回答: {result['answer']}")
                            
                            if result.get('source_documents'):
                                print("\\n📚 参考来源:")
                                for i, doc in enumerate(result['source_documents'][:2]):
                                    print(f"  {i+1}. {doc['content'][:150]}...")
                        
                        except KeyboardInterrupt:
                            break
                        except Exception as e:
                            print(f"❌ 处理出错: {e}")
                
                else:
                    print("❌ 向量存储创建失败")
            
            else:
                print("请使用 --load-docs 参数指定文档目录")
                print("例如: python main.py --load-docs ./documents")
    
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()