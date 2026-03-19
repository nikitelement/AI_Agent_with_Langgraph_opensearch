# AI Agent with LangGraph and ChromaDB

An AI agent built with LangGraph that uses ChromaDB as a knowledge base. Upload files, ask questions, and get intelligent answers based on your data.

## Features

- 🤖 AI Agent powered by LangGraph
- 📄 Document upload to ChromaDB
- 🔍 Semantic search in ChromaDB
- 💬 Interactive Streamlit UI
- 📝 Comprehensive logging system
- 💾 Persistent vector storage

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run src/ui/app.py
   ```

## Project Structure

```
├── src/
│   ├── agent/          # LangGraph agent implementation
│   ├── mcp_server/     # ChromaDB MCP server
│   ├── ui/             # Streamlit UI
│   └── utils/          # Utilities and logging
├── config/             # Configuration files
└── requirements.txt    # Python dependencies
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
chromadb:
  persist_directory: ./data/chroma_db
  collection_name: documents
  embedding_dimension: 1536

llm:
  provider: openai
  model: gpt-4
  temperature: 0.7

agent:
  max_iterations: 10
```

## How It Works

1. **Upload Documents**: Upload PDF, DOCX, TXT, or MD files via the sidebar
2. **Automatic Indexing**: Documents are processed and embedded using OpenAI embeddings
3. **Ask Questions**: The LangGraph agent searches ChromaDB and generates answers
4. **Chat Interface**: Interactive Streamlit UI for conversations

## Technologies Used

- **LangGraph**: Agent workflow orchestration
- **ChromaDB**: Vector database for semantic search
- **LangChain**: LLM integration
- **OpenAI**: Embeddings and LLM
- **Streamlit**: User interface

## License

MIT
