# filepath: /Users/helenlee/Library/CloudStorage/OneDrive-Personal/AI Project/ai-content-system/scripts/test_research_planner.py

import os
import sys
import argparse

# Manual environment variable loader - no dependencies needed
def load_env_file(env_path='.env'):
    """Load environment variables from a .env file"""
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
                    print(f"Loaded env var: {key.strip()}")
                except ValueError:
                    print(f"Skipping invalid line: {line}")
        print("Environment variables loaded successfully")
    except FileNotFoundError:
        print(f"Warning: {env_path} file not found")

# Load environment variables
load_env_file()

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.dependency_injection import get_container

def main():
    parser = argparse.ArgumentParser(description="Test the Research Planner")
    parser.add_argument("--topic", type=str, default="Impact of AI on healthcare",
                       help="Research topic to generate a plan for")
    parser.add_argument("--provider", type=str, default=os.getenv("LLM_PROVIDER", "openai"),
                       help="LLM provider to use (openai, huggingface, llamacpp)")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o"),
                       help="Model name or path")
    args = parser.parse_args()
    
    # Print API key status (just first few chars for security)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        print(f"OpenAI API Key found: {api_key[:5]}...")
    else:
        print("Warning: No OpenAI API Key found in environment variables!")
    
    # Get container
    container = get_container()
    
    # Get research planner with custom settings
    planner = container.get("research_planner", 
                           provider_name=args.provider,
                           llm_params={"model": args.model})
    
    # Generate research plan
    print(f"Generating research plan for topic: {args.topic}")
    plan = planner.generate_plan(args.topic)
    
    # Print plan in markdown format
    print("\n" + "="*80)
    print("RESEARCH PLAN")
    print("="*80 + "\n")
    print(planner.export_to_markdown(plan))
    
if __name__ == "__main__":
    main()