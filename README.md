# AI Content System

A comprehensive, modular AI system that transforms news topics into well-researched, multi-format content with automated distribution.

## Business Requirements

The AI Content System addresses the need for high-quality, research-backed content creation at scale. This system:

- **Automates Research**: Transforms news topics or issues into structured research plans
- **Ensures Credibility**: Finds and evaluates high-credibility online sources
- **Creates Data-Rich Content**: Generates comprehensive reports with data visualization
- **Maintains Brand Consistency**: Uses style transfer models to maintain consistent voice
- **Multi-Platform Publishing**: Automatically publishes to WordPress (blogs) and Threads (social)
- **Modular Architecture**: Supports extensibility through component-based design
- **Language Model Flexibility**: Works with various LLMs (GPT-4o by default, with options for open-source models)

The system leverages Multi-Channel Processing (MCP) protocol and agentic AI to coordinate complex workflows from research to publication, ensuring consistent quality across different content formats and platforms.

## Project Structure

ai-content-system/
├── core/
│   ├── agents/
│   │   ├── research_planner.py       # Research plan generation
│   │   ├── source_collector.py       # Source gathering and credibility assessment
│   │   └── data_analyzer.py          # Data analysis and visualization
├── content_engine/
│   ├── generators/
│   │   ├── blog_generator.py         # Blog content generation
│   │   ├── thread_generator.py       # Thread content generation
│   │   └── report_generator.py       # Research report generation
│   └── style_transfer/               # Style transfer module
│       ├── brand_profiles/           # Brand style configurations
│       └── style_adapter.py          # Style adapter
├── integrations/
│   ├── mcp_client/                   # MCP protocol integration
│   │   ├── connectors/
│   │   │   ├── wordpress_client.py   # WordPress publishing
│   │   │   └── threads_client.py     # Threads publishing
│   │   └── mcp_bus.py                # MCP message bus
│   └── apis/
│       ├── news_api.py               # News source API
│       └── scholar_api.py            # Academic resource API
├── infrastructure/
│   ├── config_loader.py              # Unified configuration management
│   ├── dependency_injection.py       # Dependency injection container
│   └── logging/                      # Logging and monitoring system
├── tests/
│   ├── unit/
│   │   ├── test_research_planner.py  # Unit tests for research planner
│   │   └── test_style_adapter.py     # Unit tests for style adapter
│   └── integration/
│       └── test_pipeline_e2e.py      # End-to-end pipeline tests
├── scripts/
│   ├── deploy.sh                     # Deployment script
│   ├── setup_mcp.sh                  # MCP environment setup
│   └── test_research_planner.py      # Script to test research planner interactively
├── docs/
│   ├── api_reference.md              # Module API documentation
│   └── deployment_guide.md           # Deployment guide
├── config/
│   ├── mcp_config.yaml               # MCP protocol configuration
│   └── brand_profiles/               # Brand style configuration files
│       ├── tech_blog.yaml            # Tech blog style profile
│       └── social_media.yaml         # Social media style profile
├── Dockerfile                        # Containerization configuration
├── requirements.txt                  # Python dependencies
└── main.py                           # System entry point

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/ai-content-system.git
   cd ai-content-system
   ```
2. Create a virtual environment (optional)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables
   ```bash
   cp .env.example .env
   ```
5. Run the application
   ```bash
   python main.py
   ```

You don't need to write code that installs packages programmatically within your application - this is handled externally through pip or your containerization process.

## Dependency Management

This project follows these forward-looking dependency principles:

1. **Latest Versions First**: We use the latest stable versions of core dependencies
2. **No Downgrades**: We find alternatives rather than downgrade dependencies
3. **Isolation**: All dependencies are managed in a dedicated virtual environment

## Handling Anaconda Conflicts

If you're using Anaconda, you may see dependency warnings from packages like:
- anaconda-cloud-auth (requiring older pydantic)
- spyder, numba, scipy (requiring older numpy)

These warnings can be safely ignored when using a dedicated virtual environment, as they refer to global Anaconda packages not used by this project.

## Test the research planner with a specific topic
python scripts/test_research_planner.py --topic "Impact of climate change on global food security"

## Run unit tests
python -m pytest tests/unit/test_research_planner.py -v