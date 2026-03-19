"""
Configuration loader for the AI Agent application.
Loads settings from config.yaml and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .logging_config import get_logger

logger = get_logger(__name__)


class ChromaDBConfig(BaseModel):
    """ChromaDB configuration"""
    persist_directory: str = "./data/chroma_db"
    collection_name: str = "documents"
    embedding_dimension: int = 1536


class LLMConfig(BaseModel):
    """LLM configuration"""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000


class AgentConfig(BaseModel):
    """Agent configuration"""
    max_iterations: int = 10
    verbose: bool = True
    tool_choice: str = "auto"


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/app.log"
    max_bytes: int = 10485760
    backup_count: int = 5


class StreamlitConfig(BaseModel):
    """Streamlit UI configuration"""
    page_title: str = "AI Agent with ChromaDB"
    page_icon: str = "🤖"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"


class AppConfig(BaseSettings):
    """Application configuration"""
    openai_api_key: str = ""
    chromadb: ChromaDBConfig = ChromaDBConfig()
    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()
    logging: LoggingConfig = LoggingConfig()
    streamlit: StreamlitConfig = StreamlitConfig()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file and environment variables.
    
    Args:
        config_path: Path to config.yaml file
    
    Returns:
        AppConfig instance with loaded configuration
    """
    # Load environment variables
    load_dotenv()
    
    # Determine config path
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "config.yaml"
        )
    
    config_dict: Dict[str, Any] = {}
    
    # Load YAML config
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_dict = yaml_config
                logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
    
    # Override with environment variables
    config = AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chromadb=ChromaDBConfig(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", config_dict.get("chromadb", {}).get("persist_directory", "./data/chroma_db")),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", config_dict.get("chromadb", {}).get("collection_name", "documents")),
            embedding_dimension=config_dict.get("chromadb", {}).get("embedding_dimension", 1536)
        ),
        llm=LLMConfig(
            provider=config_dict.get("llm", {}).get("provider", "openai"),
            model=os.getenv("LLM_MODEL", config_dict.get("llm", {}).get("model", "gpt-4")),
            temperature=config_dict.get("llm", {}).get("temperature", 0.7),
            max_tokens=config_dict.get("llm", {}).get("max_tokens", 2000)
        ),
        agent=AgentConfig(
            max_iterations=config_dict.get("agent", {}).get("max_iterations", 10),
            verbose=config_dict.get("agent", {}).get("verbose", True),
            tool_choice=config_dict.get("agent", {}).get("tool_choice", "auto")
        ),
        logging=LoggingConfig(
            level=os.getenv("LOG_LEVEL", config_dict.get("logging", {}).get("level", "INFO")),
            file=os.getenv("LOG_FILE", config_dict.get("logging", {}).get("file", "logs/app.log")),
            max_bytes=config_dict.get("logging", {}).get("max_bytes", 10485760),
            backup_count=config_dict.get("logging", {}).get("backup_count", 5)
        ),
        streamlit=StreamlitConfig(
            page_title=config_dict.get("streamlit", {}).get("page_title", "AI Agent with ChromaDB"),
            page_icon=config_dict.get("streamlit", {}).get("page_icon", "🤖"),
            layout=config_dict.get("streamlit", {}).get("layout", "wide"),
            initial_sidebar_state=config_dict.get("streamlit", {}).get("initial_sidebar_state", "expanded")
        )
    )
    
    logger.info(f"Configuration loaded. ChromaDB: {config.chromadb.persist_directory}")
    
    return config


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


__all__ = ['AppConfig', 'load_config', 'get_config']
