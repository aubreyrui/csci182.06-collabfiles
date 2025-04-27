import os
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredPowerPointLoader, TextLoader
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

#Put API_KEY here or the getenv thingy
API_KEY = ""
embedding_model = OpenAIEmbeddings(openai_api_key=API_KEY)
client = OpenAI(api_key=API_KEY)

TOP_K = 3

chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet", 
    persist_directory="./chroma_store"  
))

collection = chroma_client.get_or_create_collection(name="documents")

def load_files(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pptx"):
            try:
                loader = UnstructuredPowerPointLoader(filepath)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    documents.append((doc.page_content, {"source": filename}))
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        elif filename.endswith(".txt"):
            try:
                loader = TextLoader(filepath)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    documents.append((doc.page_content, {"source": filename}))
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        elif filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(filepath)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    documents.append((doc.page_content, {"source": filename}))
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        else:
            print(f"Skipping {filename}: Only .pdf, .txt, and .pptx files are allowed.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.create_documents(
        [doc[0] for doc in documents], 
        metadatas=[doc[1] for doc in documents] 
    )

    return split_documents

def ingest_documents(folder_path):
    docs = load_files(folder_path)
    texts = [content for content, _ in docs]
    metadatas = [{"source": source} for _, source in docs]
    embeddings = embedding_model.embed_documents(texts)

    ids = [f"doc_{i}" for i in range(len(texts))]
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    chroma_client.persist()
    print("Ingestion complete.")

def query_documents(query_text, top_k=TOP_K):
    query_embedding = embedding_model.embed_query(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return list(zip(results['documents'][0], results['metadatas'][0]['source']))

def generate_answer(query, context_chunks):
    context = "\n\n".join(chunk[0] for chunk in context_chunks)
    prompt = f"Answer the following question based on the context:\n\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", help="Folder with documents to ingest")
    parser.add_argument("--query", help="Query to run against the vector DB")
    args = parser.parse_args()

    if args.ingest:
        ingest_documents(args.ingest)

    if args.query:
        results = query_documents(args.query)
        answer = generate_answer(args.query, results)

        print("\nAnswer:\n", answer)
        print("\nSources:")
        for content, source in results:
            print(f"- {source}")
