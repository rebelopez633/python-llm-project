import logging
from langchain.schema import Document
import requests
from graph_state import GraphState
from web_search_interface import WebSearchInterface

class WebSearchSearxngImpl(WebSearchInterface):

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
        docs = self._web_search_tool(question)
        documents.extend(docs)
        return {"documents": documents}
    
    def _web_search_tool(self, question):
        search_url = "http://localhost:8080/search"
        # Make a GET request to the search URL with the query and format as parameters
        res = requests.get(search_url, params={"q": question, "format": "json"})  
        # Parse the JSON response and get the top 5 results
        web_results = res.json()['results'][:5]
        docs = [Document(page_content=d['content'], metadata={'source': d['url'], 'page': "Web page"}) for d in web_results]
        return docs
