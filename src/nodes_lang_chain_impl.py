import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from nodes_interface import NodesInterface
from graph_state import GraphState

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class NodesLangChainImpl(NodesInterface):
    def __init__(self, local_llm, retriever):
        self.llm = ChatOllama(model=local_llm, temperature=0)
        self.llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
        self.retriever = retriever
        self.web_search_tool = TavilySearchResults(k=3)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def retrieve(self, state: GraphState) -> any:
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        logging.info("---RETRIEVE---")
        question = state["question"]
        logging.debug(f"Question: {question}")

        # Write retrieved documents to documents key in state
        documents = self.retriever.invoke(question)
        return {"documents": documents}
    

    def route_question(self, state) -> str:
        """
        Route question to web search or RAG

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.
        Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

        logging.info("---ROUTE QUESTION---")
        route_question = self.llm_json_mode.invoke(
            [SystemMessage(content=router_instructions)]
            + [HumanMessage(content=state["question"])]
        )
        source = json.loads(route_question.content)["datasource"]
        if source == "websearch":
            logging.info("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source == "vectorstore":
            logging.info("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        
    def grade_documents(self, state: GraphState) -> dict:
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question. If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

        doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
        Read this carefully and objectively assess whether the document contains at least some information that is relevant to the question. 
        Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

        logging.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            doc_grader_prompt_formatted = doc_grader_prompt.format(
                document=d.page_content, question=question
            )
            result = self.llm_json_mode.invoke(
                [SystemMessage(content=doc_grader_instructions)]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            # Document relevant
            if grade.lower() == "yes":
                logging.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                logging.info("---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "web_search": web_search}
    
    def generate(self, state: GraphState) -> dict:
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        rag_prompt = """You are an assistant for question-answering tasks. 
        Here is the context to use to answer the question:

        {context} 

        Think carefully about the above context. 
        Now, review the user question:

        {question}

        Provide an answer to this questions using only the above context. 
        Use three sentences maximum and keep the answer concise.

        Answer:"""

        logging.info("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        loop_step = state.get("loop_step", 0)

        # RAG generation
        docs_txt = self.format_docs(documents)
        rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        return {"generation": generation, "loop_step": loop_step + 1}
    
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
