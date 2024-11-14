from langgraph.graph import StateGraph
from IPython.display import Image, display
from langgraph.graph import END
from graph_state import GraphState

from edges_lang_chain_impl import EdgesLangChainImpl
from nodes_lang_chain_impl import NodesLangChainImpl
import logging

class Controlflow:
    def __init__(self, local_llm, retriever):
        self.edges = EdgesLangChainImpl(local_llm, retriever)
        self.nodes = NodesLangChainImpl(local_llm, retriever)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def build_graph(self):

        workflow = StateGraph(GraphState)

        self.logger.info("Building graph...")
        workflow.add_node("websearch", self.nodes.web_search)  # web search
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)

        # Build graph
        self.logger.info("Setting conditional entry point...")
        workflow.set_conditional_entry_point(
            self.edges.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )
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

        # Compile
        self.logger.info("Compiling graph...")
        graph = workflow.compile()
        display(Image(graph.get_graph().draw_mermaid_png()))

        inputs = {"question": "What are the types of agent memory?", "max_retries": 3}
        self.logger.info("Streaming events for inputs: %s", inputs)
        for event in graph.stream(inputs, stream_mode="values"):
            self.logger.info("Event: %s", event)
            
        # Test on current events
        inputs = {
            "question": "What are the models released today for llama3.2?",
            "max_retries": 3,
        }
        self.logger.info("Streaming events for inputs: %s", inputs)
        for event in graph.stream(inputs, stream_mode="values"):
            self.logger.info("Event: %s", event)