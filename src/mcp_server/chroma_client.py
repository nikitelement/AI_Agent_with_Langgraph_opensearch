"""
ChromaDB client and MCP server implementation.
Handles document indexing, search, and management operations.
"""

import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from src.utils import get_logger, get_config

logger = get_logger(__name__)


class Document(BaseModel):
    """Document model for indexing"""
    id: Optional[str] = None
    content: str = Field(..., description="Document content/text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    embedding: Optional[List[float]] = None


class SearchResult(BaseModel):
    """Search result model"""
    id: str
    content: str
    distance: float
    metadata: Dict[str, Any]


class ChromaDBMCPServer:
    """
    ChromaDB MCP Server for document management and search.
    Provides tools for LangGraph agent to interact with ChromaDB.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents",
        embedding_dimension: int = 1536
    ):
        """
        Initialize ChromaDB MCP Server.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
            embedding_dimension: Dimension of embedding vectors
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embeddings
        self.embeddings = None
        
        # Create or get collection
        self.collection = self._create_collection()
        
        logger.info(f"ChromaDB MCP Server initialized: {collection_name}")
    
    def _create_collection(self):
        """Create or get the collection"""
        try:
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document storage for AI Agent"}
            )
            logger.info(f"Collection ready: {self.collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def set_embeddings(self, embeddings: OpenAIEmbeddings) -> None:
        """Set the embeddings model"""
        self.embeddings = embeddings
        logger.info("Embeddings model set")
    
    def reset(self) -> bool:
        """
        Reset the collection (delete all documents).
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._create_collection()
            logger.info("Collection reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise
    
    def index_document(self, document: Document) -> str:
        """
        Index a document into ChromaDB.
        
        Args:
            document: Document to index
        
        Returns:
            Document ID
        """
        try:
            # Generate ID if not provided
            doc_id = document.id or str(uuid.uuid4())
            
            # Generate embedding if not provided
            if document.embedding is None and self.embeddings:
                document.embedding = self.embeddings.embed_query(document.content)
            
            # Add metadata
            metadata = {
                **document.metadata,
                "created_at": datetime.utcnow().isoformat(),
                "content_length": len(document.content)
            }
            
            # Add document
            self.collection.add(
                ids=[doc_id],
                documents=[document.content],
                embeddings=[document.embedding] if document.embedding else None,
                metadatas=[metadata]
            )
            
            logger.info(f"Indexed document: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            raise
    
    def index_documents(self, documents: List[Document]) -> List[str]:
        """
        Index multiple documents.
        
        Args:
            documents: List of documents to index
        
        Returns:
            List of document IDs
        """
        try:
            # Generate IDs
            ids = []
            for doc in documents:
                if doc.id is None:
                    doc.id = str(uuid.uuid4())
                ids.append(doc.id)
            
            # Generate embeddings in batch
            if self.embeddings:
                contents = [doc.content for doc in documents]
                embeddings_list = self.embeddings.embed_documents(contents)
            else:
                embeddings_list = None
            
            # Prepare data
            docs = [doc.content for doc in documents]
            metadatas = [
                {
                    **doc.metadata,
                    "created_at": datetime.utcnow().isoformat(),
                    "content_length": len(doc.content)
                }
                for doc in documents
            ]
            
            # Add documents
            self.collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            
            logger.info(f"Indexed {len(ids)} documents")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    def search(
        self,
        query: str,
        k: int = 5,
        use_semantic: bool = True
    ) -> List[SearchResult]:
        """
        Search documents in ChromaDB.
        
        Args:
            query: Search query
            k: Number of results to return
            use_semantic: Whether to use semantic search
        
        Returns:
            List of search results
        """
        try:
            if use_semantic and self.embeddings:
                # Generate query embedding
                query_embedding = self.embeddings.embed_query(query)
                
                # Semantic search
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                # Get all documents and filter manually (fallback)
                results = self.collection.get(include=["documents", "metadatas", "ids"])
                
                # Simple text search
                matched = []
                for i, doc in enumerate(results["documents"]):
                    if query.lower() in doc.lower():
                        matched.append(i)
                
                # Return matched results
                results = {
                    "ids": [[results["ids"][i] for i in matched[:k]]],
                    "documents": [[results["documents"][i] for i in matched[:k]]],
                    "metadatas": [[results["metadatas"][i] for i in matched[:k]]],
                    "distances": [[0.0] * min(k, len(matched))]
                }
            
            # Parse results
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    search_results.append(SearchResult(
                        id=results["ids"][0][i],
                        content=results["documents"][0][i],
                        distance=results["distances"][0][i] if results["distances"] else 0.0,
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                    ))
            
            logger.info(f"Search returned {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document if found, None otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
            
            if result["documents"] and result["documents"][0]:
                return Document(
                    id=result["ids"][0],
                    content=result["documents"][0],
                    metadata=result["metadatas"][0] if result["metadatas"] else {},
                    embedding=result["embeddings"][0] if result["embeddings"] else None
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise
    
    def delete_all_documents(self) -> int:
        """
        Delete all documents from the collection.
        
        Returns:
            Number of documents deleted
        """
        try:
            count = self.collection.count()
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._create_collection()
            logger.info(f"Deleted {count} documents")
            return count
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with stats
        """
        try:
            stats = {
                "doc_count": self.collection.count(),
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if ChromaDB is healthy and accessible.
        
        Returns:
            True if healthy
        """
        try:
            count = self.collection.count()
            logger.info(f"ChromaDB health check: {count} documents")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Singleton instance
_mcp_server: Optional[ChromaDBMCPServer] = None


def get_mcp_server() -> ChromaDBMCPServer:
    """Get the global MCP server instance"""
    global _mcp_server
    if _mcp_server is None:
        config = get_config()
        _mcp_server = ChromaDBMCPServer(
            persist_directory=config.chromadb.persist_directory,
            collection_name=config.chromadb.collection_name,
            embedding_dimension=config.chromadb.embedding_dimension
        )
    return _mcp_server


__all__ = [
    'ChromaDBMCPServer',
    'Document',
    'SearchResult',
    'get_mcp_server'
]
