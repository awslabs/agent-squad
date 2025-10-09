from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
from agent_squad.utils.logger import Logger
import numpy as np
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


class OverlapResult:
    def __init__(self, overlap_percentage: str, potential_conflict: str):
        self.overlap_percentage = overlap_percentage
        self.potential_conflict = potential_conflict


class UniquenessScore:
    def __init__(self, agent: str, uniqueness_score: str):
        self.agent = agent
        self.uniqueness_score = uniqueness_score


class AgentOverlapAnalyzer:
    def __init__(self, agents: Dict[str, Dict[str, str]]):
        if not agents:
            raise ValueError("Agents dictionary cannot be empty.")
        self.agents = agents

    def analyze_overlap(self):
        agent_descriptions = [self._preprocess(agent["description"]) for agent in self.agents.values()]
        agent_names = list(self.agents.keys())

        if len(agent_names) < 2:
            raise ValueError("Agent Overlap Analysis requires at least two agents.")  # Raise ValueError
        tfidf_matrix = self._build_tfidf_matrix(agent_descriptions)
        overlap_results = self.calculate_pairwise_overlap(agent_names, tfidf_matrix)
        uniqueness_scores = self.calculate_uniqueness_scores(agent_names, tfidf_matrix)
        self.log_results(overlap_results, uniqueness_scores)

    def _preprocess(self, text: str) -> str:
        tokens = re.findall(r'\b\w+\b', text.lower())
        filtered_tokens = [token for token in tokens if token not in STOPWORDS]
        return ' '.join(filtered_tokens)

    def _build_tfidf_matrix(self, agent_descriptions: List[str]):
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(agent_descriptions)

    def calculate_pairwise_overlap(self, agent_names: List[str], tfidf_matrix) -> Dict[str, OverlapResult]:
        overlap_results = {}
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                agent1 = agent_names[i]
                agent2 = agent_names[j]
                similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
                overlap_percentage = f"{similarity * 100:.2f}%"
                key = f"{agent1}__{agent2}"
                overlap_results[key] = OverlapResult(
                    overlap_percentage=overlap_percentage,
                    potential_conflict="High" if similarity > 0.3 else "Medium" if similarity > 0.1 else "Low"
                )
        return overlap_results

    def calculate_uniqueness_scores(self, agent_names: List[str], tfidf_matrix) -> List[UniquenessScore]:
        uniqueness_scores = []
        for index in range(len(agent_names)):
            similarities = [
                cosine_similarity(tfidf_matrix[index], tfidf_matrix[i])[0][0]
                for i in range(len(agent_names)) if i != index
            ]
            avg_similarity = np.mean(similarities) if similarities else 0
            uniqueness_score = f"{max(0, (1 - avg_similarity)) * 100:.2f}%"  # Ensure non-negative score
            uniqueness_scores.append(UniquenessScore(
                agent=agent_names[index],
                uniqueness_score=uniqueness_score
            ))
        return uniqueness_scores

    def log_results(self, overlap_results: Dict[str, OverlapResult], uniqueness_scores: List[UniquenessScore]):
        Logger.info("Pairwise Overlap Results:")
        Logger.info("_________________________\n")
        for key, value in overlap_results.items():
            agent1, agent2 = key.split("__")
            Logger.info(f"{agent1} - {agent2}:\n- Overlap Percentage - {value.overlap_percentage}\n- Potential Conflict - {value.potential_conflict}\n")

        Logger.info("Uniqueness Scores:")
        Logger.info("_________________\n")
        for score in uniqueness_scores:
            Logger.info(f"Agent: {score.agent}, Uniqueness Score: {score.uniqueness_score}")
