"""
LangGraph agent implementation.
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.tools import get_tools
from src.utils import get_logger, get_config

logger = get_logger(__name__)


class OpenSearchAgent:
    """
    LangGraph agent for answering questions using OpenSearch.
    """
    
    def __init__(self):
        """Initialize the agent"""
        config = get_config()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            api_key=config.openai_api_key
        )
        
        # Bind tools to LLM
        self.tools = get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # System prompt
        self.system_prompt = """You are a helpful AI assistant that answers questions based on documents stored in OpenSearch.

When answering questions:
1. Use the search_documents tool to find relevant information
2. Always base your answers on the retrieved documents
3. If no relevant documents are found, clearly state that
4. Cite sources when possible
5. Be concise but informative

Available tools:
- search_documents: Search for relevant documents
- get_document_stats: Get statistics about indexed documents
- index_document: Add a new document
- clear_all_documents: Delete all documents (use with caution)
"""
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("OpenSearch Agent initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        # Define nodes
        def should_continue(state: AgentState) -> str:
            """Determine if we should continue or end"""
            if state.get("error"):
                return "end"
            if state.get("iterations", 0) >= get_config().agent.max_iterations:
                return "end"
            return "continue"
        
        def agent_node(state: AgentState) -> AgentState:
            """Agent node - calls LLM with tools"""
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=state["question"])
            ]
            
            # Add document context if available
            if state.get("documents"):
                doc_context = "\n\n".join([
                    f"Document {i+1}: {doc['content'][:500]}"
                    for i, doc in enumerate(state["documents"])
                ])
                messages.append(
                    SystemMessage(content=f"Retrieved documents:\n{doc_context}")
                )
            
            try:
                response = self.llm_with_tools.invoke(messages)
                
                # Check if LLM wants to use tools
                tool_calls = []
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tc in response.tool_calls:
                        tool_calls.append(tc["name"])
                        
                        # Execute tool
                        tool = next(t for t in self.tools if t.name == tc["name"])
                        result = tool.invoke(tc["args"].get("query", ""))
                        
                        # Add result to state
                        if "tool_results" not in state:
                            state["tool_results"] = []
                        state["tool_results"].append({
                            "tool": tc["name"],
                            "result": result
                        })
                
                state["answer"] = response.content if hasattr(response, "content") else str(response)
                state["tool_calls"] = tool_calls
                state["iterations"] = state.get("iterations", 0) + 1
                
            except Exception as e:
                logger.error(f"Agent error: {e}")
                state["error"] = str(e)
                state["answer"] = f"I encountered an error: {str(e)}"
            
            return state
        
        def generate_answer(state: AgentState) -> AgentState:
            """Generate final answer based on tool results"""
            try:
                # Build context from tool results
                context_parts = []
                if state.get("tool_results"):
                    for tr in state["tool_results"]:
                        context_parts.append(f"{tr['tool']}: {tr['result']}")
                
                context = "\n\n".join(context_parts) if context_parts else ""
                
                # Generate answer
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=state["question"])
                ]
                
                if context:
                    messages.append(
                        SystemMessage(content=f"Tool results:\n{context}")
                    )
                
                response = self.llm.invoke(messages)
                state["answer"] = response.content if hasattr(response, "content") else str(response)
                
            except Exception as e:
                logger.error(f"Answer generation error: {e}")
                state["answer"] = f"I encountered an error generating answer: {str(e)}"
            
            return state
        
        # Build graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("generate", generate_answer)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "generate",
                "end": END
            }
        )
        
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the agent with a question.
        
        Args:
            question: User's question
        
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing question: {question[:100]}...")
        
        initial_state: AgentState = {
            "question": question,
            "documents": [],
            "answer": "",
            "error": None,
            "iterations": 0,
            "tool_calls": [],
            "should_continue": True
        }
        
        try:
            result = self.graph.invoke(initial_state)
            
            return {
                "answer": result.get("answer", "No answer generated"),
                "iterations": result.get("iterations", 0),
                "tool_calls": result.get("tool_calls", []),
                "error": result.get("error"),
                "success": result.get("error") is None
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "iterations": 0,
                "tool_calls": [],
                "error": str(e),
                "success": False
            }
    
    def run_streaming(self, question: str):
        """
        Run the agent with streaming support.
        
        Args:
            question: User's question
        
        Yields:
            Tokens as they come in
        """
        logger.info(f"Processing question (streaming): {question[:100]}...")
        
        initial_state: AgentState = {
            "question": question,
            "documents": [],
            "answer": "",
            "error": None,
            "iterations": 0,
            "tool_calls": [],
            "should_continue": True
        }
        
        try:
            for token in self.graph.stream(initial_state):
                if "agent" in token:
                    if "answer" in token["agent"]:
                        yield token["agent"]["answer"]
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {str(e)}"


# Singleton instance
_agent: OpenSearchAgent = None


def get_agent() -> OpenSearchAgent:
    """Get the global agent instance"""
    global _agent
    if _agent is None:
        _agent = OpenSearchAgent()
    return _agent


__all__ = ['OpenSearchAgent', 'get_agent']
