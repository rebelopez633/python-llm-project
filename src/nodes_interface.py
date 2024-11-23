from abc import ABC, abstractmethod
from typing import List
from typing_extensions import TypedDict

from graph_state import GraphState


class NodesInterface(ABC):
    @abstractmethod
    def retrieve(self, state: GraphState) -> any:
        """
        Retrieve documents from vectorstore
        """
        pass

    @abstractmethod
    def route_question(self, state: GraphState) -> dict:
        """
        Route the query to the appropriate node
        """
        pass

    @abstractmethod
    def grade_documents(self, state: GraphState) -> dict:
        """
        Grade the relevance of the documents to the query
        """
        pass

    @abstractmethod
    def generate(self, state: GraphState) -> dict:
        """
        Generate an answer to the query
        """
        pass
