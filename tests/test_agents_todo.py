"""Tests for agent implementations."""

from multi_agent_research_lab.agents import SupervisorAgent, ResearcherAgent, AnalystAgent, WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _make_state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


# ---------------------------------------------------------------------------
# Supervisor routing tests
# ---------------------------------------------------------------------------

def test_supervisor_routes_researcher_when_no_notes() -> None:
    state = _make_state()
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "researcher"
    assert result.iteration == 1


def test_supervisor_routes_analyst_after_research() -> None:
    state = _make_state()
    state.research_notes = "Some research notes"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "analyst"


def test_supervisor_routes_writer_after_analysis() -> None:
    state = _make_state()
    state.research_notes = "Research notes"
    state.analysis_notes = "Analysis notes"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_routes_done_when_complete() -> None:
    state = _make_state()
    state.research_notes = "Research"
    state.analysis_notes = "Analysis"
    state.final_answer = "Final answer"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_max_iterations_guard() -> None:
    from multi_agent_research_lab.core.config import get_settings

    settings = get_settings()
    state = _make_state()
    state.iteration = settings.max_iterations  # already at limit
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"
    assert result.final_answer is not None


# ---------------------------------------------------------------------------
# Worker agent smoke tests (mock mode)
# ---------------------------------------------------------------------------

def test_researcher_populates_notes_and_sources() -> None:
    state = _make_state()
    result = ResearcherAgent().run(state)
    assert result.sources, "sources should be non-empty"
    assert result.research_notes is not None
    assert len(result.agent_results) == 1
    assert result.agent_results[0].agent == "researcher"


def test_analyst_populates_analysis_notes() -> None:
    state = _make_state()
    state.research_notes = "Mock research notes about multi-agent systems"
    result = AnalystAgent().run(state)
    assert result.analysis_notes is not None
    assert result.agent_results[0].agent == "analyst"


def test_writer_populates_final_answer() -> None:
    state = _make_state()
    state.research_notes = "Mock research notes"
    state.analysis_notes = "Mock analysis notes"
    result = WriterAgent().run(state)
    assert result.final_answer is not None
    assert result.agent_results[0].agent == "writer"


# ---------------------------------------------------------------------------
# Full workflow smoke test
# ---------------------------------------------------------------------------

def test_workflow_end_to_end() -> None:
    from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow

    state = _make_state("Explain GraphRAG briefly")
    result = MultiAgentWorkflow().run(state)
    assert result.final_answer is not None
    assert result.iteration > 0
    assert "done" in result.route_history
