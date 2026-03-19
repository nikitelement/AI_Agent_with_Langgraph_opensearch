"""
File processor for handling document uploads.
Supports PDF, DOCX, and TXT files.
"""

import os
import io
from typing import List, Optional
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument

from src.mcp_server.chroma_client import Document
from src.utils import get_logger

logger = get_logger(__name__)


class FileProcessor:
    """
    Process various file formats and convert to Document objects.
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}
    
    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if file extension is supported"""
        return Path(file_path).suffix.lower() in FileProcessor.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def process_file(file_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Process a file and return list of documents.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata to attach to documents
        
        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}
        
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        logger.info(f"Processing file: {file_path.name} ({extension})")
        
        if extension == '.pdf':
            return FileProcessor._process_pdf(file_path, metadata)
        elif extension == '.docx':
            return FileProcessor._process_docx(file_path, metadata)
        elif extension in {'.txt', '.md'}:
            return FileProcessor._process_text(file_path, metadata)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    @staticmethod
    def _process_pdf(file_path: Path, metadata: dict) -> List[Document]:
        """Process PDF file"""
        documents = []
        
        try:
            with open(file_path, 'rb') as f:
                pdf = PdfReader(f)
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        doc = Document(
                            content=text,
                            metadata={
                                **metadata,
                                "source": str(file_path.name),
                                "page": page_num + 1,
                                "total_pages": len(pdf.pages)
                            }
                        )
                        documents.append(doc)
                
                logger.info(f"Extracted {len(documents)} pages from PDF: {file_path.name}")
                
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path.name}: {e}")
            raise
        
        return documents
    
    @staticmethod
    def _process_docx(file_path: Path, metadata: dict) -> List[Document]:
        """Process DOCX file"""
        documents = []
        
        try:
            doc = DocxDocument(str(file_path))
            
            # Extract paragraphs
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Combine all text
            full_text = "\n".join(paragraphs)
            
            if full_text.strip():
                doc = Document(
                    content=full_text,
                    metadata={
                        **metadata,
                        "source": str(file_path.name),
                        "paragraphs": len(paragraphs)
                    }
                )
                documents.append(doc)
            
            logger.info(f"Extracted {len(documents)} document(s) from DOCX: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process DOCX {file_path.name}: {e}")
            raise
        
        return documents
    
    @staticmethod
    def _process_text(file_path: Path, metadata: dict) -> List[Document]:
        """Process TXT/MD file"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.strip():
                doc = Document(
                    content=content,
                    metadata={
                        **metadata,
                        "source": str(file_path.name),
                        "lines": len(content.split('\n'))
                    }
                )
                documents.append(doc)
            
            logger.info(f"Extracted 1 document from text file: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process text file {file_path.name}: {e}")
            raise
        
        return documents
    
    @staticmethod
    def process_uploaded_file(
        file_bytes: bytes,
        file_name: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Process an uploaded file from bytes.
        
        Args:
            file_bytes: File content as bytes
            file_name: Original file name
            metadata: Optional metadata
        
        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}
        
        extension = Path(file_name).suffix.lower()
        
        try:
            if extension == '.pdf':
                return FileProcessor._process_pdf_bytes(file_bytes, metadata)
            elif extension == '.docx':
                return FileProcessor._process_docx_bytes(file_bytes, metadata)
            elif extension in {'.txt', '.md'}:
                return FileProcessor._process_text_bytes(file_bytes, metadata)
            else:
                raise ValueError(f"Unsupported file type: {extension}")
                
        except Exception as e:
            logger.error(f"Failed to process uploaded file {file_name}: {e}")
            raise
    
    @staticmethod
    def _process_pdf_bytes(file_bytes: bytes, metadata: dict) -> List[Document]:
        """Process PDF from bytes"""
        documents = []
        
        pdf = PdfReader(io.BytesIO(file_bytes))
        
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    content=text,
                    metadata={
                        **metadata,
                        "page": page_num + 1,
                        "total_pages": len(pdf.pages)
                    }
                )
                documents.append(doc)
        
        return documents
    
    @staticmethod
    def _process_docx_bytes(file_bytes: bytes, metadata: dict) -> List[Document]:
        """Process DOCX from bytes"""
        doc = DocxDocument(io.BytesIO(file_bytes))
        
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        
        if full_text.strip():
            return [Document(
                content=full_text,
                metadata={
                    **metadata,
                    "paragraphs": len(paragraphs)
                }
            )]
        
        return []
    
    @staticmethod
    def _process_text_bytes(file_bytes: bytes, metadata: dict) -> List[Document]:
        """Process text from bytes"""
        content = file_bytes.decode('utf-8')
        
        if content.strip():
            return [Document(
                content=content,
                metadata={
                    **metadata,
                    "lines": len(content.split('\n'))
                }
            )]
        
        return []


__all__ = ['FileProcessor']
