"""
文档处理器
支持多种文档格式的解析和处理
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# LangChain imports
from langchain.document_loaders import (
    PyPDFLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain.schema import Document

# Third-party imports
from loguru import logger
import magic

# Project imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config


class DocumentProcessor:
    """文档处理器类"""
    
    def __init__(self):
        """初始化文档处理器"""
        self.config = get_config()
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.md': UnstructuredMarkdownLoader,
            '.markdown': UnstructuredMarkdownLoader,
            '.txt': TextLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
        }
        
        # 文本分割器配置
        self.text_splitters = {
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\\n\\n", "\\n", " ", ""]
            ),
            'token': TokenTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            'markdown': MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
            )
        }
        
        logger.info("文档处理器初始化完成")
    
    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        加载单个文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            文档列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 获取文件扩展名
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {extension}")
        
        try:
            # 选择合适的加载器
            loader_class = self.supported_extensions[extension]
            loader = loader_class(str(file_path))
            
            # 加载文档
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source_file': str(file_path),
                    'file_name': file_path.name,
                    'file_extension': extension,
                    'file_size': file_path.stat().st_size,
                    'load_time': datetime.now().isoformat(),
                    'file_hash': self._calculate_file_hash(file_path)
                })
            
            logger.info(f"成功加载文档: {file_path} ({len(documents)} 个片段)")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败: {file_path}, 错误: {e}")
            raise
    
    def load_documents(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        批量加载文档
        
        Args:
            file_paths: 文档路径列表
            
        Returns:
            所有文档的列表
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"跳过文档 {file_path}: {e}")
                continue
        
        logger.info(f"批量加载完成，总计 {len(all_documents)} 个文档片段")
        return all_documents
    
    def load_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> List[Document]:
        """
        加载目录中的所有支持的文档
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归加载子目录
            
        Returns:
            所有文档的列表
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"目录不存在或不是目录: {directory_path}")
        
        # 查找所有支持的文件
        file_paths = []
        
        if recursive:
            for extension in self.supported_extensions.keys():
                file_paths.extend(directory_path.rglob(f"*{extension}"))
        else:
            for extension in self.supported_extensions.keys():
                file_paths.extend(directory_path.glob(f"*{extension}"))
        
        logger.info(f"在目录 {directory_path} 中找到 {len(file_paths)} 个支持的文件")
        
        return self.load_documents(file_paths)
    
    def split_documents(self, 
                       documents: List[Document], 
                       method: str = 'recursive',
                       **kwargs) -> List[Document]:
        """
        分割文档
        
        Args:
            documents: 文档列表
            method: 分割方法 ('recursive', 'token', 'markdown')
            **kwargs: 分割器参数
            
        Returns:
            分割后的文档片段列表
        """
        if method not in self.text_splitters:
            raise ValueError(f"不支持的分割方法: {method}")
        
        splitter = self.text_splitters[method]
        
        # 更新分割器参数
        if kwargs:
            if method == 'recursive':
                if 'chunk_size' in kwargs:
                    splitter._chunk_size = kwargs['chunk_size']
                if 'chunk_overlap' in kwargs:
                    splitter._chunk_overlap = kwargs['chunk_overlap']
            elif method == 'token':
                if 'chunk_size' in kwargs:
                    splitter._chunk_size = kwargs['chunk_size']
                if 'chunk_overlap' in kwargs:
                    splitter._chunk_overlap = kwargs['chunk_overlap']
        
        try:
            # 分割文档
            split_docs = splitter.split_documents(documents)
            
            # 为每个片段添加额外的元数据
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    'chunk_index': i,
                    'split_method': method,
                    'split_time': datetime.now().isoformat(),
                    'chunk_length': len(doc.page_content)
                })
            
            logger.info(f"文档分割完成: {len(documents)} -> {len(split_docs)} 个片段")
            return split_docs
            
        except Exception as e:
            logger.error(f"文档分割失败: {e}")
            raise
    
    def extract_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """
        提取文档集合的元数据统计
        
        Args:
            documents: 文档列表
            
        Returns:
            元数据统计信息
        """
        if not documents:
            return {}
        
        metadata = {
            'total_documents': len(documents),
            'total_characters': sum(len(doc.page_content) for doc in documents),
            'total_words': sum(len(doc.page_content.split()) for doc in documents),
            'file_types': {},
            'sources': set(),
            'processing_time': datetime.now().isoformat()
        }
        
        # 统计文件类型
        for doc in documents:
            if 'file_extension' in doc.metadata:
                ext = doc.metadata['file_extension']
                metadata['file_types'][ext] = metadata['file_types'].get(ext, 0) + 1
            
            if 'source_file' in doc.metadata:
                metadata['sources'].add(doc.metadata['source_file'])
        
        metadata['unique_sources'] = len(metadata['sources'])
        metadata['sources'] = list(metadata['sources'])
        
        return metadata
    
    def filter_documents(self, 
                        documents: List[Document],
                        min_length: Optional[int] = None,
                        max_length: Optional[int] = None,
                        keywords: Optional[List[str]] = None,
                        file_types: Optional[List[str]] = None) -> List[Document]:
        """
        过滤文档
        
        Args:
            documents: 文档列表
            min_length: 最小长度
            max_length: 最大长度
            keywords: 必须包含的关键词
            file_types: 允许的文件类型
            
        Returns:
            过滤后的文档列表
        """
        filtered_docs = documents.copy()
        
        # 长度过滤
        if min_length is not None:
            filtered_docs = [doc for doc in filtered_docs if len(doc.page_content) >= min_length]
        
        if max_length is not None:
            filtered_docs = [doc for doc in filtered_docs if len(doc.page_content) <= max_length]
        
        # 关键词过滤
        if keywords:
            filtered_docs = [
                doc for doc in filtered_docs 
                if any(keyword.lower() in doc.page_content.lower() for keyword in keywords)
            ]
        
        # 文件类型过滤
        if file_types:
            filtered_docs = [
                doc for doc in filtered_docs 
                if doc.metadata.get('file_extension', '').lower() in [ft.lower() for ft in file_types]
            ]
        
        logger.info(f"文档过滤完成: {len(documents)} -> {len(filtered_docs)} 个文档")
        return filtered_docs
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        计算文件哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件的MD5哈希值
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"计算文件哈希失败: {file_path}, {e}")
            return "unknown"
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式列表"""
        return list(self.supported_extensions.keys())
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        验证文件是否可以处理
        
        Args:
            file_path: 文件路径
            
        Returns:
            验证结果
        """
        file_path = Path(file_path)
        result = {
            'valid': False,
            'file_exists': False,
            'supported_format': False,
            'file_size': 0,
            'error': None
        }
        
        try:
            # 检查文件是否存在
            if not file_path.exists():
                result['error'] = "文件不存在"
                return result
            
            result['file_exists'] = True
            result['file_size'] = file_path.stat().st_size
            
            # 检查文件格式
            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                result['error'] = f"不支持的文件格式: {extension}"
                return result
            
            result['supported_format'] = True
            
            # 检查文件大小限制 (例如: 100MB)
            max_size = 100 * 1024 * 1024  # 100MB
            if result['file_size'] > max_size:
                result['error'] = f"文件过大: {result['file_size']} bytes"
                return result
            
            result['valid'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result


if __name__ == "__main__":
    # 测试文档处理器
    processor = DocumentProcessor()
    
    # 打印支持的格式
    print("支持的文件格式:", processor.get_supported_formats())
    
    # 创建测试文档目录
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    # 创建测试文档
    test_file = test_dir / "test.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("这是一个测试文档。\\n\\n它包含了多个段落。\\n\\n用于测试文档处理功能。")
    
    try:
        # 测试加载文档
        documents = processor.load_document(test_file)
        print(f"加载了 {len(documents)} 个文档片段")
        
        # 测试分割文档
        split_docs = processor.split_documents(documents, method='recursive', chunk_size=50, chunk_overlap=10)
        print(f"分割后得到 {len(split_docs)} 个片段")
        
        # 测试提取元数据
        metadata = processor.extract_metadata(split_docs)
        print("文档元数据:", metadata)
        
    except Exception as e:
        print(f"测试失败: {e}")
    
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()
        if test_dir.exists():
            test_dir.rmdir()
    
    print("文档处理器测试完成")