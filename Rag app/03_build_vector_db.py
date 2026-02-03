from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Preprcoess import load_documents


def build_vector_db() -> None:
    """Build and save the FAISS vector database from the cleaned movie dataset."""
    docs = load_documents()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")

    print("Vector DB built successfully")


if __name__ == "__main__":
    build_vector_db()
