"""
LangGraph agent tools for ChromaDB operations.
"""

from typing import List, Dict, Any
from langchain_core.tools import tool

from src.mcp_server import get_mcp_server, SearchResult
from src.utils import get_logger

logger = get_logger(__name__)


@tool
def search_documents(query: str, k: int = 5) -> str:
    """
    Search for relevant documents in ChromaDB based on a query.
    
    This tool searches through the indexed documents to find the most relevant
    information. Use this when you need to answer questions about uploaded documents.
    
    Args:
        query: The search query string
        k: Number of documents to retrieve (default: 5)
    
    Returns:
        A string containing the search results with relevant content
    """
    try:
        logger.info(f"Searching documents with query: {query}")
        
        mcp_server = get_mcp_server()
        results: List[SearchResult] = mcp_server.search(query=query, k=k)
        
        if not results:
            return "No relevant documents found in the knowledge base."
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"--- Document {i} (Distance: {result.distance:.4f}) ---\n"
                f"Content: {result.content[:500]}...\n"
                f"Metadata: {result.metadata}\n"
            )
        
        output = "\n".join(formatted_results)
        logger.info(f"Found {len(results)} documents")
        
        return output
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Error searching documents: {str(e)}"


@tool
def get_document_stats() -> str:
    """
    Get statistics about the indexed documents in ChromaDB.
    
    Returns:
        A string containing document count and collection information
    """
    try:
        logger.info("Getting document statistics")
        
        mcp_server = get_mcp_server()
        stats = mcp_server.get_stats()
        
        output = (
            f"Collection: {stats['collection_name']}\n"
            f"Persist Directory: {stats['persist_directory']}\n"
            f"Document Count: {stats['doc_count']}"
        )
        
        return output
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return f"Error getting statistics: {str(e)}"


@tool
def index_document(content: str, metadata: str = "{}") -> str:
    """
    Index a new document into ChromaDB.
    
    Args:
        content: The document content/text to index
        metadata: JSON string of metadata (default: "{}")
    
    Returns:
        A confirmation message with the document ID
    """
    try:
        import json
        
        logger.info(f"Indexing new document")
        
        mcp_server = get_mcp_server()
        
        # Parse metadata
        try:
            meta = json.loads(metadata) if metadata else {}
        except:
            meta = {}
        
        from src.mcp_server import Document
        doc = Document(content=content, metadata=meta)
        
        doc_id = mcp_server.index_document(doc)
        
        logger.info(f"Document indexed: {doc_id}")
        return f"Document indexed successfully. ID: {doc_id}"
        
    except Exception as e:
        logger.error(f"Failed to index document: {e}")
        return f"Error indexing document: {str(e)}"


@tool
def clear_all_documents() -> str:
    """
    Delete all documents from the ChromaDB collection.
    
    WARNING: This action cannot be undone. Use with caution.
    
    Returns:
        A confirmation message with the number of deleted documents
    """
    try:
        logger.warning("Clearing all documents")
        
        mcp_server = get_mcp_server()
        count = mcp_server.delete_all_documents()
        
        logger.info(f"Deleted {count} documents")
        return f"Successfully deleted {count} documents from the collection."
        
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        return f"Error clearing documents: {str(e)}"


def get_tools() -> List:
    """
    Get all tools available to the agent.
    
    Returns:
        List of LangChain tools
    """
    return [
        search_documents,
        get_document_stats,
        index_document,
        clear_all_documents
    ]


__all__ = [
    'search_documents',
    'get_document_stats',
    'index_document',
    'clear_all_documents',
    'get_tools'
]
