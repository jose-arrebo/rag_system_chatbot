from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LoadingChunkingManager:
    """
    A manager class for loading and chunking documents.
    
    This class provides methods to load a PDF file and create text chunks
    from the loaded data using a text splitter.
    """
    
    def __init__(self):
        """
        Initialize the LoadingChunkingManager.
        
        This constructor currently does not perform any operations.
        """
        
        pass
    
    def load_pdf(self, file_path:str):
        """
        Load a PDF file from the given file path.

        This method uses the UnstructuredPDFLoader to load data from the
        specified PDF file.

        Parameters
        ----------
        file_path : str
            The path to the PDF file to be loaded.

        Returns
        -------
        data : list
            A list of documents loaded from the PDF file.
        """
        
        loader = UnstructuredPDFLoader(file_path)
        
        data = loader.load()
        
        return data
    
    def create_embedding_and_chunk(self, data):
        """
        Create text chunks from the loaded data.

        This method uses the RecursiveCharacterTextSplitter to split
        the loaded data into chunks of specified size with overlap.

        Parameters
        ----------
        data : list
            A list of documents to be split into chunks.

        Returns
        -------
        chunks : list
            A list of text chunks created from the documents.
        """
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        
        chunks = text_splitter.split_documents(data)
        
        return chunks
