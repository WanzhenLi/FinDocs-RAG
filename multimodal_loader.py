"""
Multi-Format Document Loader for FinDoc Demo
Handles loading various document types including PDFs, Word docs, Excel files, and text files
"""

import os
from typing import List, Dict, Any, Union
from pathlib import Path
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiFormatDocumentLoader:
    """Handles loading various document types"""
    
    def __init__(self):
        """Initialize the multi-format document loader with supported file types"""
        self.loaders = {
            "pdf": PyPDFLoader,
            "docx": Docx2txtLoader,
            "doc": Docx2txtLoader,
            "csv": CSVLoader,
            "xlsx": UnstructuredExcelLoader,
            "xls": UnstructuredExcelLoader,
            "txt": TextLoader,
            "md": TextLoader,
            "py": TextLoader,
            "js": TextLoader,
            "html": TextLoader,
            "xml": TextLoader,
        }
        
        # Text-based formats
        self.text_formats = {"txt", "md", "py", "js", "html", "xml", "json", "yaml", "yml"}
    
    def get_file_extension(self, file_path: Union[str, Path]) -> str:
        """Extract file extension from file path"""
        return Path(file_path).suffix[1:].lower()
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if the file format is supported"""
        extension = self.get_file_extension(file_path)
        return extension in self.loaders
    
    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a document based on its file extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List[Document]: Loaded document chunks
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        extension = self.get_file_extension(file_path)
        
        # Check if format is supported
        if not self.is_supported_format(file_path):
            raise ValueError(f"Unsupported file type: {extension}")
        
        logger.info(f"Loading document: {file_path} (format: {extension})")
        
        try:
            # Get the appropriate loader
            loader_class = self.loaders[extension]
            
            # Special handling for different file types
            if extension in ["csv"]:
                # For CSV files, we might want to specify encoding
                loader = loader_class(str(file_path), encoding="utf-8")
            else:
                # Standard loading for other formats
                loader = loader_class(str(file_path))
            
            # Load the document
            documents = loader.load()
            
            # Add metadata about the file
            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "file_type": extension,
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size if file_path.exists() else 0,
                })
            
            logger.info(f"Successfully loaded {len(documents)} document chunks from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise Exception(f"Failed to load document {file_path}: {str(e)}")
    
    def load_multiple_documents(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        Load multiple documents from a list of file paths
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            List[Document]: Combined list of loaded document chunks
        """
        all_documents = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {str(e)}")
                failed_files.append(str(file_path))
        
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
        
        logger.info(f"Successfully loaded {len(all_documents)} total document chunks from {len(file_paths) - len(failed_files)} files")
        return all_documents
    
    def load_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List[Document]: Combined list of loaded document chunks
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        all_files = []
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and self.is_supported_format(file_path):
                all_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} supported files in {directory_path}")
        
        # Load all found files
        return self.load_multiple_documents(all_files)
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions"""
        return list(self.loaders.keys())
    
    def get_document_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic information about a document without loading it
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": "File not found"}
        
        extension = self.get_file_extension(file_path)
        
        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_extension": extension,
            "file_size": file_path.stat().st_size,
            "is_supported": self.is_supported_format(file_path),
            "loader_type": self.loaders.get(extension, "Unsupported").__name__ if extension in self.loaders else "Unsupported"
        }


# Convenience function for quick document loading
def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Convenience function to load a single document
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List[Document]: Loaded document chunks
    """
    loader = MultiFormatDocumentLoader()
    return loader.load_document(file_path)


# Backward compatibility alias
MultiModalDocumentLoader = MultiFormatDocumentLoader


# Example usage and testing
if __name__ == "__main__":
    # Create loader instance
    loader = MultiFormatDocumentLoader()
    
    # Print supported extensions
    print("Supported file extensions:")
    print(loader.get_supported_extensions())
    
    # Example usage (uncomment to test with actual files)
    # documents = loader.load_document("sample.pdf")
    # print(f"Loaded {len(documents)} documents")
