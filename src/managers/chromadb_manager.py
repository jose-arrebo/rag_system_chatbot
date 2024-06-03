from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

class ChromaDBManager:
    """
    A manager class for interacting with a Chroma vector database.
    
    This class provides methods to add vectors to the database using 
    documents and embeddings.
    """
    
    def __init__(self) -> None:
        """
        Initialize the ChromaDBManager.
        
        This constructor currently does not perform any operations.
        """
        
        pass
    
    def add_vector_to_db(self, chunks):
        """
        Add vectors to the Chroma database from given document chunks.

        This method takes a list of document chunks, embeds them using 
        the OllamaEmbeddings model, and stores them in a Chroma 
        vector database.

        Parameters
        ----------
        chunks : list
            A list of document chunks to be embedded and added to the 
            Chroma database.

        Returns
        -------
        vector_db : Chroma
            The Chroma vector database containing the embedded document 
            chunks.
        """
        
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            collection_name="local-rag"
        )
        
        return vector_db