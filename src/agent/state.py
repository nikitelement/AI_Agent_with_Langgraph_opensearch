"""
LangGraph agent state definition.
"""

from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """
    State for the LangGraph agent.
    """
    # User input
    question: str
    
    # Documents retrieved from search
    documents: List[Dict[str, Any]]
    
    # Generated answer
    answer: str
    
    # Error message if any
    error: Optional[str]
    
    # Number of iterations
    iterations: int
    
    # Tool calls made
    tool_calls: List[str]
    
    # Whether the agent should continue
    should_continue: bool


class InputState(BaseModel):
    """Input state for starting the agent"""
    question: str = Field(..., description="User's question")


class OutputState(BaseModel):
    """Output state from the agent"""
    answer: str = Field(..., description="Agent's answer")
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    iterations: int = Field(default=0, description="Number of iterations")
    error: Optional[str] = Field(default=None, description="Error message if any")


__all__ = ['AgentState', 'InputState', 'OutputState']
