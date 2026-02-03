from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

def load_rag(google_api_key, model_name="gemini-1.5-flash"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5})
    )

    return qa
