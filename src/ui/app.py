"""
Streamlit UI for the AI Agent application with ChromaDB.
"""

import streamlit as st
import time
from datetime import datetime
from typing import List, Dict, Any

from src.utils import get_logger, get_config, setup_logging
from src.mcp_server import get_mcp_server, FileProcessor
from src.agent import get_agent

# Initialize logger
logger = get_logger(__name__)

# Page configuration
config = get_config()
st.set_page_config(
    page_title=config.streamlit.page_title,
    page_icon=config.streamlit.page_icon,
    layout=config.streamlit.layout,
    initial_sidebar_state=config.streamlit.initial_sidebar_state
)


class StreamlitApp:
    """Streamlit application for the AI Agent"""
    
    def __init__(self):
        """Initialize the Streamlit app"""
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "agent_initialized" not in st.session_state:
            st.session_state.agent_initialized = False
        
        if "chromadb_connected" not in st.session_state:
            st.session_state.chromadb_connected = False
    
    def init_services(self):
        """Initialize ChromaDB and Agent"""
        try:
            # Connect to ChromaDB
            mcp_server = get_mcp_server()
            
            # Check health
            if mcp_server.health_check():
                st.session_state.chromadb_connected = True
                logger.info("ChromaDB connected successfully")
            else:
                logger.warning("ChromaDB health check failed")
            
            # Initialize embeddings
            app_config = get_config()
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                api_key=app_config.openai_api_key
            )
            mcp_server.set_embeddings(embeddings)
            
            # Initialize agent
            get_agent()
            st.session_state.agent_initialized = True
            
            logger.info("All services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            st.error(f"Failed to initialize: {str(e)}")
            return False
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:
            st.title("🤖 AI Agent Settings")
            
            # Service status
            st.subheader("Service Status")
            
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                db_status = "🟢 Connected" if st.session_state.chromadb_connected else "🔴 Disconnected"
                st.write(f"**ChromaDB:** {db_status}")
            with status_col2:
                agent_status = "🟢 Ready" if st.session_state.agent_initialized else "🔴 Not Ready"
                st.write(f"**Agent:** {agent_status}")
            
            st.divider()
            
            # Document stats
            st.subheader("Document Statistics")
            if st.session_state.chromadb_connected:
                try:
                    mcp_server = get_mcp_server()
                    stats = mcp_server.get_stats()
                    st.write(f"📄 Total Documents: {stats['doc_count']}")
                    st.write(f"📁 Collection: `{stats['collection_name']}`")
                except Exception as e:
                    st.warning(f"Could not fetch stats: {e}")
            else:
                st.info("Connect to ChromaDB to see stats")
            
            st.divider()
            
            # File upload section
            st.subheader("📤 Upload Documents")
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'docx', 'txt', 'md'],
                accept_multiple_files=True,
                help="Upload PDF, DOCX, TXT, or MD files"
            )
            
            if uploaded_files and st.session_state.chromadb_connected:
                if st.button("Process & Index Files", type="primary"):
                    self._process_uploaded_files(uploaded_files)
            
            st.divider()
            
            # Clear documents
            st.subheader("🗑️ Manage Documents")
            if st.button("Clear All Documents", type="secondary"):
                self._clear_documents()
            
            st.divider()
            
            # Reinitialize button
            if st.button("🔄 Reinitialize Services"):
                st.session_state.agent_initialized = False
                st.session_state.chromadb_connected = False
                st.rerun()
    
    def _process_uploaded_files(self, uploaded_files):
        """Process and index uploaded files"""
        try:
            mcp_server = get_mcp_server()
            all_docs = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}")
                
                # Read file bytes
                file_bytes = uploaded_file.read()
                
                # Process file
                docs = FileProcessor.process_uploaded_file(
                    file_bytes=file_bytes,
                    file_name=uploaded_file.name,
                    metadata={
                        "uploaded_at": datetime.utcnow().isoformat(),
                        "file_type": uploaded_file.type
                    }
                )
                
                all_docs.extend(docs)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Index documents
            status_text.text("Indexing documents...")
            if all_docs:
                doc_ids = mcp_server.index_documents(all_docs)
                status_text.text(f"Indexed {len(doc_ids)} documents successfully!")
                st.success(f"✅ Successfully indexed {len(doc_ids)} documents")
                logger.info(f"Indexed {len(doc_ids)} documents from {len(uploaded_files)} files")
            else:
                st.warning("No content extracted from files")
            
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            logger.error(f"Failed to process files: {e}")
            st.error(f"Error processing files: {str(e)}")
    
    def _clear_documents(self):
        """Clear all documents from ChromaDB"""
        try:
            mcp_server = get_mcp_server()
            count = mcp_server.delete_all_documents()
            st.success(f"✅ Deleted {count} documents")
            logger.info(f"Cleared {count} documents")
            st.rerun()
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            st.error(f"Error clearing documents: {str(e)}")
    
    def render_main(self):
        """Render main chat area"""
        st.title("💬 Ask Questions about Your Documents")
        
        # Welcome message
        if not st.session_state.messages:
            st.markdown("""
            ### Welcome to the AI Agent! 👋
            
            This AI agent can answer questions based on documents you've uploaded to ChromaDB.
            
            **Getting Started:**
            1. Upload documents using the sidebar
            2. Wait for them to be indexed
            3. Ask questions in the chat below
            
            The agent will search through your documents and provide accurate answers.
            """)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "metadata" in message:
                    with st.expander("Details"):
                        st.json(message["metadata"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_placeholder.markdown("🤔 Thinking...")
                
                try:
                    # Get agent response
                    agent = get_agent()
                    result = agent.run(prompt)
                    
                    # Display response
                    response = result.get("answer", "No response generated")
                    response_placeholder.markdown(response)
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "metadata": {
                            "iterations": result.get("iterations", 0),
                            "tool_calls": result.get("tool_calls", []),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    })
                    
                    logger.info(f"Question answered: {prompt[:50]}... -> {response[:100]}...")
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    response_placeholder.markdown(error_msg)
                    logger.error(f"Chat error: {e}")
    
    def run(self):
        """Run the Streamlit app"""
        # Initialize services
        if not (st.session_state.chromadb_connected and st.session_state.agent_initialized):
            with st.spinner("Initializing services..."):
                self.init_services()
        
        # Render UI
        self.render_sidebar()
        self.render_main()


def main():
    """Main entry point"""
    try:
        app = StreamlitApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
