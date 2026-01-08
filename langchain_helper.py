from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()


# Initialize embeddings once (local, free)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_db_from_documents(pdf_path: str) -> FAISS:
    """
    Load a PDF file, split it into chunks, and store embeddings in FAISS.
    """

    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # 3. Create vector database
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query: str, k: int = 4):
    """
    Retrieve relevant chunks and generate an answer using LLM.
    """

    # 4. Similarity search
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # 5. LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # 6. Prompt
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that answers questions using internal documents.

        Answer the following question:
        {question}

        Using ONLY the information from the documents below:
        {docs}

        If the answer is not contained in the documents, say "I don't know".

        Give a clear and detailed answer.
        """
    )

    # 7. Chain
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(
        question=query,
        docs=docs_page_content
    )

    response = response.replace("\n", " ")
    return response, docs
