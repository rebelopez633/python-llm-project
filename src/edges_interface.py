from abc import ABC, abstractmethod
from typing import List
from typing_extensions import TypedDict

from graph_state import GraphState

class EdgesInterface(ABC):
    @abstractmethod
    def route_question(self, state: GraphState) -> str:
        """
        Route question to web search or RAG
        """
        pass

    @abstractmethod
    def decide_to_generate(self, state: GraphState) -> str:
        """
        Decide whether to generate an answer or perform another action
        """
        pass

    @abstractmethod
    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        """
        Grade the generated answer against the documents and question
        """
        pass