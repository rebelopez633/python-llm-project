from abc import abstractmethod

from graph_state import GraphState


class WebSearchInterface:
    """
    Interface for web search
    """
    @abstractmethod
    def web_search(self, state: GraphState) -> dict:
        """
        Perform a web search
        """
        pass