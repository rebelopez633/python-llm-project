from langgraph.graph import StateGraph
from IPython.display import Image, display
from langgraph.graph import END
from graph_state import GraphState

from edges_lang_chain_impl import EdgesLangChainImpl
from nodes_lang_chain_impl import NodesLangChainImpl
from web_search_tavily_impl import WebSearchTavilyImpl
import logging

class Controlflow:
    def __init__(self, local_llm, retriever):
        self.edges = EdgesLangChainImpl(local_llm, retriever)
        self.nodes = NodesLangChainImpl(local_llm, retriever)
        self.web_search = WebSearchTavilyImpl()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = self._initialize_workflow()
        self._set_conditional_entry_point(workflow)
        self._add_edges(workflow)
        return self._compile_and_display_graph(workflow)

    def _initialize_workflow(self):
        workflow = StateGraph(GraphState)
        self.logger.info("Building graph...")
        workflow.add_node("websearch", self.web_search.web_search)
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)
        return workflow

    def _set_conditional_entry_point(self, workflow):
        self.logger.info("Setting conditional entry point...")
        workflow.set_conditional_entry_point(
            self.edges.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )

    def _add_edges(self, workflow):
        self.logger.info("Adding edges...")
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.edges.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_conditional_edges(
            "generate",
            self.edges.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
                "max retries": END,
            },
        )

    def _compile_and_display_graph(self, workflow):
        self.logger.info("Compiling graph...")
        graph = workflow.compile()
        display(Image(graph.get_graph().draw_mermaid_png()))
        return graph

    def stream_events(self, graph, inputs):
        self.logger.info("Streaming events for inputs: %s", inputs)
        for event in graph.stream(inputs, stream_mode="values"):
            self.logger.info("Event: %s", event)