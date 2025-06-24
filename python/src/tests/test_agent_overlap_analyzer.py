import pytest
from agent_squad.agent_overlap_analyzer import AgentOverlapAnalyzer, OverlapResult, UniquenessScore
from agent_squad.utils.logger import Logger


@pytest.fixture
def sample_agents():
    return {
        "AgentA": {"name": "AgentA", "description": "This is a test description for Agent A."},
        "AgentB": {"name": "AgentB", "description": "This is a completely different test description for Agent B."},
        "AgentC": {"name": "AgentC", "description": "Another test description, slightly similar to Agent A."}
    }


@pytest.fixture
def identical_agents():
    return {
        "Agent1": {"name": "Agent1", "description": "identical description for all agents."},
        "Agent2": {"name": "Agent2", "description": "identical description for all agents."},
        "Agent3": {"name": "Agent3", "description": "identical description for all agents."}
    }


@pytest.fixture
def single_agent():
    return {
        "SoloAgent": {"name": "SoloAgent", "description": "Only one agent in the system."}
    }


@pytest.fixture
def empty_agents():
    return {}


def test_pairwise_overlap(sample_agents):
    analyzer = AgentOverlapAnalyzer(sample_agents)
    agent_names = list(sample_agents.keys())
    tfidf_matrix = analyzer._build_tfidf_matrix([agent["description"] for agent in sample_agents.values()])
    overlap_results = analyzer.calculate_pairwise_overlap(agent_names, tfidf_matrix)

    assert len(overlap_results) == 3
    assert all(isinstance(result, OverlapResult) for result in overlap_results.values())
    assert all(0 <= float(result.overlap_percentage.strip('%')) <= 100 for result in overlap_results.values())
    assert set(overlap_results.keys()) == {"AgentA__AgentB", "AgentA__AgentC", "AgentB__AgentC"}


def test_uniqueness_scores(sample_agents):
    analyzer = AgentOverlapAnalyzer(sample_agents)
    agent_names = list(sample_agents.keys())
    tfidf_matrix = analyzer._build_tfidf_matrix([agent["description"] for agent in sample_agents.values()])
    uniqueness_scores = analyzer.calculate_uniqueness_scores(agent_names, tfidf_matrix)

    assert len(uniqueness_scores) == 3
    assert all(isinstance(score, UniquenessScore) for score in uniqueness_scores)
    assert all(0 <= float(score.uniqueness_score.strip('%')) <= 100 for score in uniqueness_scores)


def test_identical_descriptions(identical_agents):
    analyzer = AgentOverlapAnalyzer(identical_agents)
    agent_names = list(identical_agents.keys())
    tfidf_matrix = analyzer._build_tfidf_matrix([agent["description"] for agent in identical_agents.values()])
    overlap_results = analyzer.calculate_pairwise_overlap(agent_names, tfidf_matrix)

    for result in overlap_results.values():
        assert result.overlap_percentage == "100.00%"
        assert result.potential_conflict == "High"

    uniqueness_scores = analyzer.calculate_uniqueness_scores(agent_names, tfidf_matrix)
    for score in uniqueness_scores:
        assert score.uniqueness_score == "0.00%"


def test_single_agent(single_agent, capsys):
    analyzer = AgentOverlapAnalyzer(single_agent)
    with pytest.raises(ValueError, match="Agent Overlap Analysis requires at least two agents."):
        analyzer.analyze_overlap()
    

def test_empty_agents(empty_agents):
    with pytest.raises(ValueError, match="Agents dictionary cannot be empty."):
        AgentOverlapAnalyzer(empty_agents)


def test_overlap_potential_conflict_ranges(sample_agents):
    analyzer = AgentOverlapAnalyzer(sample_agents)
    agent_names = list(sample_agents.keys())
    tfidf_matrix = analyzer._build_tfidf_matrix([agent["description"] for agent in sample_agents.values()])
    overlap_results = analyzer.calculate_pairwise_overlap(agent_names, tfidf_matrix)

    for result in overlap_results.values():
        percentage = float(result.overlap_percentage.strip('%'))
        if percentage > 30:
            assert result.potential_conflict == "High"
        elif percentage > 10:
            assert result.potential_conflict == "Medium"
        else:
            assert result.potential_conflict == "Low"
