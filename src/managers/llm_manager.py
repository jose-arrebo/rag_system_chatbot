from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

class LLMManager:
    """
    A manager class for interacting with language models and retrievers.
    
    This class provides methods to load a language model, create a retriever,
    generate a chat prompt template, and build a processing chain.
    """
    
    def __init__(self):
        """
        Initialize the LLMManager.
        
        This constructor currently does not perform any operations.
        """
        
        pass
    
    def load_llama2(self):
        """
        Load the Llama2 language model.

        This method initializes the ChatOllama language model with a specified
        temperature setting.

        Returns
        -------
        llm_llama2 : ChatOllama
            The initialized ChatOllama language model.
        """
        
        local_model = "llama2"
        
        llm_llama2 = ChatOllama(model=local_model,
                                temperature=0.7)
        
        return llm_llama2
    
    def get_retriever(self, llm, vector_db):
        """
        Create a multi-query retriever using the given language model and vector database.

        This method uses a prompt template to generate multiple versions of a user
        question to improve document retrieval from a vector database.

        Parameters
        ----------
        llm : ChatOllama
            The language model to be used for generating query variations.
        vector_db : Chroma
            The vector database to retrieve documents from.

        Returns
        -------
        retriever : MultiQueryRetriever
            The multi-query retriever configured with the given language model and vector database.
        """
        
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )
        
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), 
            llm,
            prompt=query_prompt
        )
        
        return retriever
    
    def get_chat_prompt_template(self):
        """
        Generate a chat prompt template for answering questions based on provided context.

        This method creates a template to instruct an AI assistant to provide answers
        based strictly on given context documents.

        Returns
        -------
        prompt : ChatPromptTemplate
            The generated chat prompt template.
        """

        template = """You are an AI assistant that answers questions 
        based on a certain context. The context will be given in the 
        form of documents which contain the information to answer the 
        question from the user. Your job is to understand the different
        pieces of information from the documents and provide an answer.
        
        Context:
        {context}
        
        Instructions:
        You are an AI assistant designed to provide accurate and contextually 
        relevant information based strictly on the provided context. Your 
        primary goal is to answer questions using only the information given 
        in the context documents. Follow these guidelines:

        - Context-Dependent Responses: Always base your answers on the 
        information within the provided context. Only if the context does not 
        contain the necessary information, respond with: The information 
        is not available in the provided context. If you found the answer,
        do not include this sentence in your response.

        - Clarification: If a question cannot be answered based on the context, 
        clearly state that the context does not provide the required information.

        - Brevity and Precision: Keep your responses concise and directly 
        relevant to the question asked.

        - Neutrality and Objectivity: Maintain a neutral tone and provide 
        objective information based solely on the context.

        - Feedback Handling: If the user requests information that is not present 
        in the context, politely inform them of the limitation and suggest that 
        they provide more context if possible.
        
        - Infering: As a general rule, you are allowed to infer little things if 
        you consider that it would benefit the answer. When infering, make sure 
        that you are not hallucinating or changing the meaning of the facts 
        in the context.
        
        - Intro and outro policy: Do not include an intro or an outro to your 
        answers. If you know the answer to the question, just say it. If you 
        cannot find the answer, just say: The information is not available 
        in the provided context.
        
        - No mentions: It is forbidden to mention the document where you found 
        the information. Just answer the question directly.
                
        Question:
        {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        return prompt
    
    def get_chain(self, retriever, prompt, llm):
        """
        Build a processing chain that integrates retriever, prompt, and language model.

        This method creates a chain of operations starting from retrieving context,
        applying the prompt, passing the result to the language model, and finally
        parsing the output.

        Parameters
        ----------
        retriever : MultiQueryRetriever
            The retriever for fetching context documents.
        prompt : ChatPromptTemplate
            The chat prompt template for generating responses.
        llm : ChatOllama
            The language model for generating responses based on the prompt.

        Returns
        -------
        chain : dict
            A dictionary representing the processing chain.
        """
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain