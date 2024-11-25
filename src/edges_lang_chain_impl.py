import json
import logging
from llm_factory import create_llm
from langchain_core.messages import HumanMessage, SystemMessage
from edges_interface import EdgesInterface
from graph_state import GraphState

class EdgesLangChainImpl(EdgesInterface):
    def __init__(self, local_llm, retriever):
        self.llm = create_llm(local_llm)
        self.llm_json_mode = create_llm(local_llm, format="json")
        self.retriever = retriever

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def route_question(self, state: GraphState) -> str:
        """
        Route question to web search or RAG

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to thermodynamics.
        Use the vectorstore for questions on this topic. For all else, and especially for current events, use web-search.
        Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

        logging.info("---ROUTE QUESTION---")
        route_question = self.llm_json_mode.invoke(
            [SystemMessage(content=router_instructions)]
            + [HumanMessage(content=state["question"])]
        )
        source = json.loads(route_question)["datasource"]
        if source == "websearch":
            logging.info("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source == "vectorstore":
            logging.info("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        
    def decide_to_generate(self, state) -> str:
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """       
        logging.info("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        web_search = state["web_search"]
        filtered_documents = state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            logging.info(
                "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            logging.info("---DECISION: GENERATE---")
            return "generate"
        
    def grade_generation_v_documents_and_question(self, state) -> str:
        """
        Determines whether the generation is grounded in the document and answers question

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        hallucination_grader_instructions = """
        You are a teacher grading a quiz. 
        You will be given FACTS and a STUDENT ANSWER. 
        Here is the grade criteria to follow:
        (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
        (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

        Score:
        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
        Avoid simply stating the correct answer at the outset."""

        # Grader prompt
        hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

        Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

        logging.info("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

        hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
            documents=self.format_docs(documents), generation=generation
        )
        result = self.llm_json_mode.invoke(
            [SystemMessage(content=hallucination_grader_instructions)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
        grade = json.loads(result)["binary_score"]

        answer_grader_instructions = """You are a teacher grading a quiz. 
        You will be given a QUESTION and a STUDENT ANSWER. 
        Here is the grade criteria to follow:
        (1) The STUDENT ANSWER helps to answer the QUESTION

        Score:
        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
        The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.
        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
        Avoid simply stating the correct answer at the outset."""

        # Grader prompt
        answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 
        Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""
        
        # Check hallucination
        if grade == "yes":
            logging.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            logging.info("---GRADE GENERATION vs QUESTION---")
            # Test using question and generation from above
            answer_grader_prompt_formatted = answer_grader_prompt.format(
                question=question, generation=generation
            )
            result = self.llm_json_mode.invoke(
                [SystemMessage(content=answer_grader_instructions)]
                + [HumanMessage(content=answer_grader_prompt_formatted)]
            )
            grade = json.loads(result)["binary_score"]
            if grade == "yes":
                logging.info("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            elif state["loop_step"] <= max_retries:
                logging.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
            else:
                logging.info("---DECISION: MAX RETRIES REACHED---")
                return "max retries"
        elif state["loop_step"] <= max_retries:
            logging.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        else:
            logging.info("---DECISION: MAX RETRIES REACHED---")
            return "max retries"