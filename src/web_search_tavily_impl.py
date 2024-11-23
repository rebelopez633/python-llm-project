import logging
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from graph_state import GraphState
from web_search_interface import WebSearchInterface

class WebSearchTavilyImpl(WebSearchInterface):
    def __init__(self):
        self.web_search_tool = TavilySearchResults(k=3)

    def web_search(self, state: GraphState) -> dict:
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        logging.info("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        return {"documents": documents}