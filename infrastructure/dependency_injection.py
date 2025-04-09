from typing import Dict, Any, Optional, Type, Callable
import logging

from infrastructure.config_loader import ConfigLoader
from core.agents.research_planner import ResearchPlanner, LLMFactory

logger = logging.getLogger(__name__)

class DependencyContainer:
    """Simple dependency injection container."""
    
    _instances = {}
    _factories = {}
    
    @classmethod
    def register_factory(cls, name: str, factory: Callable[[], Any]) -> None:
        """Register a factory function for creating a component."""
        cls._factories[name] = factory
    
    @classmethod
    def get(cls, name: str, **kwargs) -> Any:
        """Get or create an instance of a component."""
        # Return existing instance if available and no kwargs provided
        if name in cls._instances and not kwargs:
            return cls._instances[name]
            
        # Create new instance
        if name in cls._factories:
            instance = cls._factories[name](**kwargs)
            if not kwargs:  # Only cache if no custom kwargs
                cls._instances[name] = instance
            return instance
        
        raise ValueError(f"No factory registered for '{name}'")
    
    @classmethod
    def reset(cls) -> None:
        """Reset all cached instances."""
        cls._instances.clear()

# Register factories for core components

def create_research_planner(**kwargs):
    """Factory for ResearchPlanner."""
    config = ConfigLoader.get_component_config("research_planner")
    
    # Override config with kwargs if provided
    provider_name = kwargs.get("provider_name", config.get("llm_provider"))
    llm_params = kwargs.get("llm_params", config.get("llm_params", {}))
    
    return ResearchPlanner(
        provider_name=provider_name,
        llm_params=llm_params,
        system_prompt=kwargs.get("system_prompt"),
        log_level=kwargs.get("log_level", logging.INFO)
    )

# Register factories
DependencyContainer.register_factory("research_planner", create_research_planner)

# Helper function to get the container
def get_container() -> DependencyContainer:
    """Get the dependency container."""
    return DependencyContainer