"""
Multi-Format Document Loader for Streamlit App Integration
Handles loading various document types including PDFs, Word docs, Excel files, and text files
"""

import tempfile
import os
from typing import List
from pathlib import Path
import logging

from langchain_core.documents import Document
from multimodal_loader import MultiFormatDocumentLoader as BaseMultiFormatLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamlitMultiFormatDocumentLoader:
    """Multi-format document loader with Streamlit integration"""
    
    def __init__(self):
        """Initialize with the base multi-format loader"""
        self.base_loader = BaseMultiFormatLoader()
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document from file path using the multi-format loader"""
        return self.base_loader.load_document(file_path)
    
    def load_uploaded_file(self, uploaded_file) -> List[Document]:
        """
        Loads a document from a Streamlit uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List[Document]: Document chunks from the uploaded file
        """
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Check if file type is supported
        if not self.base_loader.is_supported_format(f"dummy.{file_extension}"):
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Create temporary file to work with loaders that need file paths
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f".{file_extension}",
            prefix=f"uploaded_{uploaded_file.name.split('.')[0]}_"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            logger.info(f"Processing uploaded file: {uploaded_file.name} (size: {len(uploaded_file.getvalue())} bytes)")
            
            # Load document using the loader
            documents = self.base_loader.load_document(tmp_file_path)
            
            # Update metadata with original filename and upload info
            for doc in documents:
                doc.metadata.update({
                    "original_filename": uploaded_file.name,
                    "upload_size": len(uploaded_file.getvalue()),
                    "upload_type": uploaded_file.type if hasattr(uploaded_file, 'type') else 'unknown',
                    "processed_via": "streamlit_upload"
                })
            
            logger.info(f"Successfully processed {uploaded_file.name}: {len(documents)} chunks extracted")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing uploaded file {uploaded_file.name}: {str(e)}")
            raise Exception(f"Failed to process uploaded file {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                logger.warning(f"Could not delete temporary file: {tmp_file_path}")
    
    def load_multiple_uploaded_files(self, uploaded_files) -> List[Document]:
        """
        Loads multiple documents from Streamlit uploaded files
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            List[Document]: Combined document chunks from all files
        """
        all_documents = []
        failed_files = []
        
        for uploaded_file in uploaded_files:
            try:
                documents = self.load_uploaded_file(uploaded_file)
                all_documents.extend(documents)
                logger.info(f"Successfully loaded {uploaded_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {uploaded_file.name}: {str(e)}")
                failed_files.append(uploaded_file.name)
        
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
        
        logger.info(f"Total: {len(all_documents)} document chunks from {len(uploaded_files) - len(failed_files)} successful uploads")
        return all_documents
    
    def get_supported_extensions(self) -> List[str]:
        """Returns a list of supported file extensions"""
        return self.base_loader.get_supported_extensions()
    
    def get_supported_extensions_display(self) -> str:
        """Returns a formatted string of supported extensions for display"""
        extensions = self.get_supported_extensions()
        return ", ".join([f".{ext}" for ext in sorted(extensions)])
    
    def is_supported_file(self, filename: str) -> bool:
        """Checks if a filename has a supported extension"""
        return self.base_loader.is_supported_format(filename)
    
    def get_upload_info(self, uploaded_file) -> dict:
        """
        Gets information about an uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            dict: File information including name, size, and type
        """
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        return {
            "filename": uploaded_file.name,
            "size": len(uploaded_file.getvalue()),
            "extension": file_extension,
            "is_supported": self.is_supported_file(uploaded_file.name),
            "type": uploaded_file.type if hasattr(uploaded_file, 'type') else 'unknown'
        }


# Convenience functions for backward compatibility
def load_document(file_path: str) -> List[Document]:
    """
    Loads a single document from a file path
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List[Document]: Document chunks from the file
    """
    loader = StreamlitMultiFormatDocumentLoader()
    return loader.load_document(file_path)


def load_uploaded_file(uploaded_file) -> List[Document]:
    """
    Loads a document from a Streamlit uploaded file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        List[Document]: Document chunks from the uploaded file
    """
    loader = StreamlitMultiFormatDocumentLoader()
    return loader.load_uploaded_file(uploaded_file)


# Create default loader instance for easy import
MultiModalDocumentLoader = StreamlitMultiFormatDocumentLoader
