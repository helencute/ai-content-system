# filepath: /Users/helenlee/Library/CloudStorage/OneDrive-Personal/AI Project/ai-content-system/tests/unit/test_research_planner.py

import pytest
from unittest.mock import MagicMock, patch

from core.agents.research_planner import ResearchPlanner, ResearchPlan

@pytest.fixture
def mock_llm():
  """Create a mock LLM."""
  mock = MagicMock()
  mock.invoke.return_value = MagicMock(content="""
  {
    "topic": "Test Topic",
    "overview": "This is a test overview",
    "key_questions": [
      {"question": "Test question?", "importance": "Important", "context": "Some context"}
    ],
    "data_sources": [
      {"name": "Test source", "type": "academic", "url": "http://test.com", "relevance": "High", "access_method": "API"}
    ],
    "analysis_methods": [
      {"name": "Test method", "description": "Description", "application": "How to apply", "expected_outcome": "Outcome"}
    ],
    "timeline": [
      {"phase": "Research", "duration": "1 week", "activities": ["Activity 1"], "deliverables": ["Deliverable 1"]}
    ],
    "considerations": ["Consideration 1"],
    "expected_outcomes": ["Outcome 1"]
  }
  """)
  return mock

def test_research_planner_generates_plan(mock_llm):
  """Test that the research planner generates a plan."""
  # Arrange
  planner = ResearchPlanner(llm=mock_llm)
  
  # Act
  plan = planner.generate_plan("Test topic")
  
  # Assert
  assert plan.topic == "Test Topic"
  assert len(plan.key_questions) == 1
  assert len(plan.data_sources) == 1
  assert len(plan.analysis_methods) == 1
  
  # Verify LLM was called correctly
  mock_llm.invoke.assert_called_once()

def test_export_to_markdown():
  """Test markdown export functionality."""
  # Create a minimal plan for testing
  plan = ResearchPlan(
    topic="Test",
    overview="Overview",
    key_questions=[],
    data_sources=[],
    analysis_methods=[],
    timeline=[],
    considerations=[],
    expected_outcomes=[]
  )
  
  planner = ResearchPlanner()
  markdown = planner.export_to_markdown(plan)
  
  assert "# Research Plan: Test" in markdown
  assert "## Overview" in markdown