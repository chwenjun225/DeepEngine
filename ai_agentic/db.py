import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="chroma_db")

collection.add(
    documents=[
        "This is a document about history maintance machinery", 
        "This is a history of data sensors"
    ], 
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["this is a query document about the history maintance machinery"], 
    n_results=2
)

print(results)