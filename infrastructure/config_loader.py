import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Unified configuration management system that loads and provides access to
    configuration from various sources (files, environment variables, etc.)
    following the project's hierarchical structure.
    """
    
    # Default configuration paths based on project structure
    DEFAULT_CONFIG_DIR = "config"
    DEFAULT_APP_CONFIG = "config/app_config.yaml"
    DEFAULT_MCP_CONFIG = "config/mcp_config.yaml"
    DEFAULT_BRAND_PROFILES_DIR = "config/brand_profiles"
    
    # Configuration cache to avoid repeated file reads
    _config_cache = {}
    
    @classmethod
    def get_project_root(cls) -> Path:
        """
        Determine the project root directory.
        
        Returns:
            Path object pointing to the project root
        """
        # Start with the current directory and look for indicators of the project root
        current_path = Path(os.getcwd())
        
        # Check if we're already at the root
        if (current_path / cls.DEFAULT_CONFIG_DIR).exists():
            return current_path
            
        # Try to find the project root by looking for common indicators
        indicators = ["main.py", "Dockerfile", "requirements.txt", cls.DEFAULT_CONFIG_DIR]
        
        # Check parent directories
        for _ in range(5):  # Prevent infinite loops by limiting depth
            for indicator in indicators:
                if (current_path / indicator).exists():
                    return current_path
            current_path = current_path.parent
            
        # If we can't determine the root, use the current directory
        logger.warning("Could not determine project root. Using current directory.")
        return Path(os.getcwd())
    
    @classmethod
    def _resolve_path(cls, path: Union[str, Path]) -> Path:
        """
        Resolve a path to an absolute path from project root.
        
        Args:
            path: Path string or Path object
            
        Returns:
            Absolute Path object
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
            
        # Relative path - resolve from project root
        return cls.get_project_root() / path_obj
    
    @classmethod
    def load_yaml(cls, file_path: Union[str, Path], cache: bool = True) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            file_path: Path to the YAML file
            cache: Whether to cache the loaded configuration
            
        Returns:
            Dictionary containing the configuration
        """
        resolved_path = cls._resolve_path(file_path)
        
        # Check cache first if enabled
        if cache and str(resolved_path) in cls._config_cache:
            return cls._config_cache[str(resolved_path)]
            
        try:
            if not resolved_path.exists():
                logger.warning(f"Config file not found: {resolved_path}")
                result = {}
            else:
                with open(resolved_path, 'r') as f:
                    result = yaml.safe_load(f) or {}
                    
            # Cache if requested
            if cache:
                cls._config_cache[str(resolved_path)] = result
                
            return result
        except Exception as e:
            logger.error(f"Error loading YAML file {resolved_path}: {e}")
            return {}
    
    @classmethod
    def save_yaml(cls, file_path: Union[str, Path], data: Dict[str, Any]) -> bool:
        """
        Save configuration to a YAML file.
        
        Args:
            file_path: Path to save the YAML file
            data: Configuration data to save
            
        Returns:
            True if successful, False otherwise
        """
        resolved_path = cls._resolve_path(file_path)
        
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
            
            with open(resolved_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
                
            # Update cache
            cls._config_cache[str(resolved_path)] = data
            return True
        except Exception as e:
            logger.error(f"Error saving YAML file {resolved_path}: {e}")
            return False
    
    @classmethod
    def load_app_config(cls) -> Dict[str, Any]:
        """
        Load the main application configuration.
        
        Returns:
            Dictionary containing the application configuration
        """
        env_config_path = os.getenv("CONFIG_PATH")
        config_path = env_config_path if env_config_path else cls.DEFAULT_APP_CONFIG
        
        config = cls.load_yaml(config_path)
        
        # If empty and using default, create a sample configuration
        if not config and config_path == cls.DEFAULT_APP_CONFIG:
            config = cls._create_default_app_config()
            
        return config
    
    @classmethod
    def load_mcp_config(cls) -> Dict[str, Any]:
        """
        Load the MCP protocol configuration.
        
        Returns:
            Dictionary containing the MCP configuration
        """
        env_config_path = os.getenv("MCP_CONFIG_PATH")
        config_path = env_config_path if env_config_path else cls.DEFAULT_MCP_CONFIG
        
        config = cls.load_yaml(config_path)
        
        # If empty and using default, create a sample configuration
        if not config and config_path == cls.DEFAULT_MCP_CONFIG:
            config = cls._create_default_mcp_config()
            
        return config
    
    @classmethod
    def load_brand_profile(cls, profile_name: str) -> Dict[str, Any]:
        """
        Load a specific brand profile configuration.
        
        Args:
            profile_name: Name of the brand profile
            
        Returns:
            Dictionary containing the brand profile configuration
        """
        if not profile_name.endswith('.yaml'):
            profile_name = f"{profile_name}.yaml"
            
        profile_path = f"{cls.DEFAULT_BRAND_PROFILES_DIR}/{profile_name}"
        return cls.load_yaml(profile_path)
    
    @classmethod
    def list_brand_profiles(cls) -> List[str]:
        """
        List all available brand profiles.
        
        Returns:
            List of brand profile names
        """
        profiles_dir = cls._resolve_path(cls.DEFAULT_BRAND_PROFILES_DIR)
        
        if not profiles_dir.exists():
            os.makedirs(profiles_dir, exist_ok=True)
            # Create sample profiles
            cls._create_sample_brand_profiles()
            
        return [f.stem for f in profiles_dir.glob('*.yaml')]
    
    @classmethod
    def get_component_config(cls, component_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dictionary containing the component configuration
        """
        app_config = cls.load_app_config()
        return app_config.get(component_name, {})
    
    @classmethod
    def _create_default_app_config(cls) -> Dict[str, Any]:
        """
        Create default application configuration.
        
        Returns:
            Dictionary containing the default configuration
        """
        default_config = {
            "research_planner": {
                "llm_provider": "openai",
                "llm_params": {
                    "model": "gpt-4o",
                    "temperature": 0.2
                },
                "max_retries": 2
            },
            "source_collector": {
                "max_sources": 10,
                "credibility_threshold": 0.7,
                "apis": {
                    "news_api": {
                        "enabled": True,
                        "priority": 1
                    },
                    "scholar_api": {
                        "enabled": True,
                        "priority": 2
                    }
                }
            },
            "data_analyzer": {
                "default_chart_type": "bar",
                "color_palette": "blues",
                "max_items_per_chart": 10
            },
            "content_generators": {
                "default_brand_profile": "tech_blog",
                "max_content_length": {
                    "blog": 2000,
                    "thread": 280,
                    "report": 5000
                }
            },
            "publishing": {
                "auto_publish": False,
                "review_required": True,
                "platforms": {
                    "wordpress": {
                        "enabled": True,
                        "api_endpoint": "https://your-site.com/wp-json/wp/v2"
                    },
                    "threads": {
                        "enabled": True
                    }
                }
            }
        }
        
        # Save the default configuration
        cls.save_yaml(cls.DEFAULT_APP_CONFIG, default_config)
        return default_config
    
    @classmethod
    def _create_default_mcp_config(cls) -> Dict[str, Any]:
        """
        Create default MCP configuration.
        
        Returns:
            Dictionary containing the default MCP configuration
        """
        default_config = {
            "server": {
                "host": "localhost",
                "port": 8080,
                "debug": True
            },
            "modules": [
                {"name": "research_planner", "enabled": True},
                {"name": "source_collector", "enabled": True},
                {"name": "data_analyzer", "enabled": True},
                {"name": "content_generator", "enabled": True},
                {"name": "publisher", "enabled": True}
            ],
            "pipelines": {
                "default": [
                    "research_planner",
                    "source_collector",
                    "data_analyzer",
                    "content_generator",
                    "publisher"
                ],
                "research_only": [
                    "research_planner",
                    "source_collector",
                    "data_analyzer"
                ],
                "content_only": [
                    "content_generator",
                    "publisher"
                ]
            },
            "protocol": {
                "version": "1.0",
                "timeout": 60,
                "max_retries": 3
            }
        }
        
        # Save the default configuration
        cls.save_yaml(cls.DEFAULT_MCP_CONFIG, default_config)
        return default_config
    
    @classmethod
    def _create_sample_brand_profiles(cls) -> None:
        """Create sample brand profiles if they don't exist."""
        profiles = {
            "tech_blog": {
                "style": {
                    "tone": "professional",
                    "formality": "semi-formal",
                    "language_complexity": "medium-high",
                    "perspective": "third-person"
                },
                "structure": {
                    "intro_style": "question-based",
                    "paragraph_length": "medium",
                    "subheading_frequency": "frequent",
                    "conclusion_type": "summary-with-question" 
                },
                "seo": {
                    "keyword_density": 2.0,
                    "meta_description_length": 155,
                    "include_related_links": True
                },
                "formatting": {
                    "header_style": "###",
                    "emphasis_markers": ["**", "_"],
                    "list_style": "bullet",
                    "code_formatting": True
                }
            },
            "social_media": {
                "style": {
                    "tone": "conversational",
                    "formality": "casual",
                    "language_complexity": "low",
                    "perspective": "first-person"
                },
                "structure": {
                    "intro_style": "hook-based",
                    "paragraph_length": "very-short",
                    "use_emojis": True,
                    "hashtag_strategy": "trending-relevant"
                },
                "engagement": {
                    "call_to_action": "question",
                    "mention_strategy": "authority-figures",
                    "reply_encouragement": True
                },
                "formatting": {
                    "capital_emphasis": True,
                    "emoji_frequency": "high",
                    "max_hashtags": 5,
                    "include_media_prompt": True
                }
            }
        }
        
        # Save sample profiles
        profiles_dir = cls._resolve_path(cls.DEFAULT_BRAND_PROFILES_DIR)
        os.makedirs(profiles_dir, exist_ok=True)
        
        for name, profile in profiles.items():
            cls.save_yaml(f"{profiles_dir}/{name}.yaml", profile)


# For backward compatibility
def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Legacy function for loading configuration from a specified path.
    
    Args:
        config_path: Path to the config file. If None, will use the default app config.
            
    Returns:
        Dictionary containing configuration values
    """
    if config_path:
        return ConfigLoader.load_yaml(config_path)
    else:
        return ConfigLoader.load_app_config()


if __name__ == "__main__":
    # Example usage
    app_config = ConfigLoader.load_app_config()
    print("App Config:", json.dumps(app_config, indent=2))
    
    mcp_config = ConfigLoader.load_mcp_config()
    print("MCP Config:", json.dumps(mcp_config, indent=2))
    
    planner_config = ConfigLoader.get_component_config("research_planner")
    print("Research Planner Config:", json.dumps(planner_config, indent=2))
    
    profiles = ConfigLoader.list_brand_profiles()
    print("Available Brand Profiles:", profiles)
    
    if profiles:
        sample_profile = ConfigLoader.load_brand_profile(profiles[0])
        print(f"Sample Profile ({profiles[0]}):", json.dumps(sample_profile, indent=2))