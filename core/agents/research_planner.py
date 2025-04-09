import logging
import os
from typing import Dict, List, Any, Optional, Union, Type
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.llms import HuggingFacePipeline, LlamaCpp

# Import config utils
from infrastructure.config_loader import ConfigLoader

class ResearchQuestion(BaseModel):
    """Model representing a research question with its importance and context."""
    question: str = Field(description="The research question to be answered")
    importance: str = Field(description="Why this question is important to answer")
    context: str = Field(description="Background context for this question")

class DataSource(BaseModel):
    """Model representing a recommended data source for research."""
    name: str = Field(description="Name of the data source")
    type: str = Field(description="Type of source (academic, news, government, etc.)")
    url: Optional[str] = Field(None, description="URL to access the source if applicable")
    relevance: str = Field(description="Why this source is relevant")
    access_method: str = Field(description="How to access/query this source")

class AnalysisMethod(BaseModel):
    """Model representing an analysis method to apply to the collected data."""
    name: str = Field(description="Name of the analysis method")
    description: str = Field(description="Brief description of the method")
    application: str = Field(description="How to apply this method to the topic")
    expected_outcome: str = Field(description="What insights this method might reveal")

class TimelineStep(BaseModel):
    """Model representing a step in the research timeline."""
    phase: str = Field(description="Name of the research phase")
    duration: str = Field(description="Estimated duration")
    activities: List[str] = Field(description="Activities to be performed in this phase")
    deliverables: List[str] = Field(description="Expected deliverables from this phase")

class ResearchPlan(BaseModel):
    """Complete research plan with all components."""
    topic: str = Field(description="The research topic/news item")
    overview: str = Field(description="Brief overview of the research approach")
    key_questions: List[ResearchQuestion] = Field(description="Key questions to address")
    data_sources: List[DataSource] = Field(description="Recommended data sources")
    analysis_methods: List[AnalysisMethod] = Field(description="Analysis methods to apply")
    timeline: List[TimelineStep] = Field(description="Research timeline")
    considerations: List[str] = Field(description="Additional considerations for the research")
    expected_outcomes: List[str] = Field(description="Expected outcomes and deliverables")


class LLMProvider(ABC):
    """Abstract base class for language model providers."""
    
    @abstractmethod
    def get_llm(self, **kwargs) -> BaseLanguageModel:
        """Return a language model instance."""
        pass


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models."""
    
    def get_llm(self, **kwargs) -> BaseLanguageModel:
        model = kwargs.get("model", os.getenv("OPENAI_MODEL", "gpt-4o"))
        temperature = kwargs.get("temperature", float(os.getenv("LLM_TEMPERATURE", "0.2")))
        return ChatOpenAI(model=model, temperature=temperature)


class HuggingFaceProvider(LLMProvider):
    """Provider for Hugging Face models."""
    
    def get_llm(self, **kwargs) -> BaseLanguageModel:
        model_id = kwargs.get("model_id", os.getenv("HF_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1"))
        return HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={"max_length": 2048}
        )


class LlamaCppProvider(LLMProvider):
    """Provider for local Llama.cpp models."""
    
    def get_llm(self, **kwargs) -> BaseLanguageModel:
        model_path = kwargs.get("model_path", os.getenv("LLAMA_MODEL_PATH", "./models/llama-2-13b-chat.gguf"))
        n_ctx = kwargs.get("n_ctx", int(os.getenv("LLAMA_CONTEXT_WINDOW", "4096")))
        return LlamaCpp(model_path=model_path, n_ctx=n_ctx)


class LLMFactory:
    """Factory for creating LLM instances."""
    
    _providers = {
        "openai": OpenAIProvider(),
        "huggingface": HuggingFaceProvider(),
        "llamacpp": LlamaCppProvider()
    }
    
    @classmethod
    def register_provider(cls, name: str, provider: LLMProvider) -> None:
        """Register a new LLM provider."""
        cls._providers[name] = provider
    
    @classmethod
    def create_llm(cls, provider_name: str = None, **kwargs) -> BaseLanguageModel:
        """Create an LLM instance from the specified provider."""
        if provider_name is None:
            provider_name = os.getenv("LLM_PROVIDER", "openai")
        
        provider = cls._providers.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
        
        return provider.get_llm(**kwargs)


class ResearchPlanner:
    """
    Agent responsible for generating comprehensive research plans for given topics using LLMs.
    Acts as the first step in the content generation pipeline.
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are an expert research methodology consultant specialized in creating 
    comprehensive research plans. For the given topic or news item, develop a detailed research 
    plan that includes:
    
    1. An overview of the approach
    2. Key questions that need to be answered (with importance and context)
    3. Recommended data sources with justification
    4. Analysis methods to apply to collected data
    5. A phased timeline with activities and deliverables
    6. Special considerations relevant to this topic
    7. Expected outcomes
    
    Your research plan should be thorough, practical and implementable.
    {format_instructions}
    """
    
    def __init__(self, 
                 llm: Optional[BaseLanguageModel] = None,
                 provider_name: Optional[str] = None,
                 llm_params: Optional[Dict[str, Any]] = None,
                 system_prompt: Optional[str] = None,
                 output_model: Type[BaseModel] = ResearchPlan,
                 log_level: int = logging.INFO):
        """
        Initialize the ResearchPlanner.
        
        Args:
            llm: Pre-configured language model (overrides provider_name and llm_params if provided)
            provider_name: Name of the LLM provider to use
            llm_params: Parameters to pass to the LLM provider
            system_prompt: Custom system prompt template
            output_model: Pydantic model to use for output parsing
            log_level: Logging level
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Load configuration - CHANGE THIS LINE
        config = ConfigLoader.get_component_config("research_planner")
        
        # Set up LLM
        if llm:
            self.llm = llm
        else:
            provider = provider_name or config.get("llm_provider", os.getenv("LLM_PROVIDER", "openai"))
            params = llm_params or config.get("llm_params", {})
            self.llm = LLMFactory.create_llm(provider, **params)
        
        # Set up output parser
        self.output_model = output_model
        self.output_parser = PydanticOutputParser(pydantic_object=output_model)
        
        # Set up prompt template
        self.system_prompt = system_prompt or config.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Topic: {topic}")
        ])
        
        # Additional configurations
        self.max_retries = config.get("max_retries", 2)
    
    def generate_plan(self, topic: str) -> ResearchPlan:
        """
        Generate a comprehensive research plan for the given topic.
        
        Args:
            topic: The news item or research topic to create a plan for
            
        Returns:
            A structured ResearchPlan object
        """
        self.logger.info(f"Generating research plan for topic: {topic}")
        
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                # Prepare the prompt with formatting instructions and topic
                formatted_prompt = self.prompt.format_messages(
                    topic=topic,
                    format_instructions=self.output_parser.get_format_instructions()
                )
                
                # Generate the response using the LLM
                llm_response = self.llm.invoke(formatted_prompt)
                
                # Parse the response into a ResearchPlan object
                research_plan = self.output_parser.parse(llm_response.content)
                research_plan.topic = topic  # Ensure topic is correctly set
                
                self.logger.info(f"Successfully generated research plan for '{topic}' with {len(research_plan.key_questions)} key questions")
                return research_plan
                
            except Exception as e:
                retry_count += 1
                self.logger.warning(f"Attempt {retry_count} failed: {str(e)}")
                if retry_count > self.max_retries:
                    self.logger.error(f"Failed to generate research plan after {self.max_retries} attempts")
                    raise
    
    def export_to_dict(self, research_plan: ResearchPlan) -> Dict[str, Any]:
        """
        Convert the research plan to a dictionary format for passing to other agents.
        
        Args:
            research_plan: The ResearchPlan object to convert
            
        Returns:
            Dictionary representation of the research plan
        """
        return research_plan.model_dump()
    
    def export_to_markdown(self, research_plan: ResearchPlan) -> str:
        """
        Convert the research plan to a Markdown format for human readability.
        
        Args:
            research_plan: The ResearchPlan object to convert
            
        Returns:
            Markdown string representation of the research plan
        """
        md = f"# Research Plan: {research_plan.topic}\n\n"
        md += f"## Overview\n{research_plan.overview}\n\n"
        
        # Key Questions
        md += "## Key Questions\n"
        for i, question in enumerate(research_plan.key_questions, 1):
            md += f"### {i}. {question.question}\n"
            md += f"**Importance**: {question.importance}\n\n"
            md += f"**Context**: {question.context}\n\n"
        
        # Data Sources
        md += "## Data Sources\n"
        for i, source in enumerate(research_plan.data_sources, 1):
            md += f"### {i}. {source.name} ({source.type})\n"
            if source.url:
                md += f"**URL**: {source.url}\n\n"
            md += f"**Relevance**: {source.relevance}\n\n"
            md += f"**Access Method**: {source.access_method}\n\n"
        
        # Analysis Methods
        md += "## Analysis Methods\n"
        for i, method in enumerate(research_plan.analysis_methods, 1):
            md += f"### {i}. {method.name}\n"
            md += f"**Description**: {method.description}\n\n"
            md += f"**Application**: {method.application}\n\n"
            md += f"**Expected Outcome**: {method.expected_outcome}\n\n"
        
        # Timeline
        md += "## Timeline\n"
        for i, step in enumerate(research_plan.timeline, 1):
            md += f"### Phase {i}: {step.phase} ({step.duration})\n"
            md += "**Activities:**\n"
            for activity in step.activities:
                md += f"- {activity}\n"
            md += "\n**Deliverables:**\n"
            for deliverable in step.deliverables:
                md += f"- {deliverable}\n"
            md += "\n"
        
        # Additional Sections
        md += "## Considerations\n"
        for consideration in research_plan.considerations:
            md += f"- {consideration}\n"
        md += "\n"
        
        md += "## Expected Outcomes\n"
        for outcome in research_plan.expected_outcomes:
            md += f"- {outcome}\n"
        
        return md


if __name__ == "__main__":
    # Example usage with environment variables or default values
    planner = ResearchPlanner(
        provider_name=os.getenv("LLM_PROVIDER", "openai"),
        llm_params={
            "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2"))
        }
    )
    
    topic = "Impact of AI regulation on technology innovation"
    plan = planner.generate_plan(topic)
    
    # Print the plan in Markdown format
    print(planner.export_to_markdown(plan))