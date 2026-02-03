import pandas as pd
from langchain_core.documents import Document

def load_documents():
    df = pd.read_csv("cleaned_mymoviedb.csv")
    df = df.fillna("")

    df["text"] = (
        "Title: " + df["Title"].astype(str) + "\n"
        "Year: " + df["Release_Date"].astype(str) + "\n"
        "Genre: " + df["Genre"].astype(str) + "\n"
        "Popularity: " + df["Popularity"].astype(str) + "\n"
        "Votes: " + df["Vote_Count"].astype(str) + "\n"
        "Rating: " + df["Vote_Average"].astype(str)
    )

    docs = [Document(page_content=row["text"]) for _, row in df.iterrows()]
    return docs
