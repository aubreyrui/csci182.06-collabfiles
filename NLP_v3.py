import os
import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredPowerPointLoader, TextLoader
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

#Put API_KEY here or the getenv thingy
API_KEY = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=API_KEY)
client = OpenAI(api_key=API_KEY)

TOP_K = 3

chroma_client = chromadb.PersistentClient(path="./chroma_store")
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
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    embeddings = embedding_model.embed_documents(texts)

    ids = [f"doc_{i}" for i in range(len(texts))]
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("Ingestion complete.")

def query_documents(query_text, top_k=TOP_K):
    query_embedding = embedding_model.embed_query(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return list(zip(results['documents'][0], [metadata['source'] for metadata in results['metadatas'][0]]))

def summarize_context(context_chunks, batch_size=2):
    summarized_batches = []
    for i in range(0, len(context_chunks), batch_size):
        batch = context_chunks[i:i+batch_size]
        batch_text = "\n\n".join(chunk[0] for chunk in batch)

        prompt = f"Summarize the following information briefly and clearly:\n\n{batch_text}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        summary = response.choices[0].message.content
        summarized_batches.append(summary)

    final_context = "\n\n".join(summarized_batches)
    return final_context
'''
def summarize_context(context_chunks):
    combined_context = "\n\n".join(chunk[0] for chunk in context_chunks)
    prompt = f"Summarize the information for answering a question: \n\n{combined_context}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
'''
def generate_answer(query, context_chunks):
    context = summarize_context(context_chunks)
    prompt = f"Answer the following question based on the context:\n\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# Similarity test
def load_ground_truths(filepath):
    ground_truths = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ground_truths = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading ground truths file: {e}")
    return ground_truths

def get_openai_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def check_correctness(prediction, ground_truths, threshold=0.8):
    pred_embedding = get_openai_embedding(prediction)
    ground_truth_embeddings = [get_openai_embedding(gt) for gt in ground_truths]
    
    similarities = cosine_similarity(
        [pred_embedding], ground_truth_embeddings
    )
    
    top_idx = np.argmax(similarities)
    top_score = similarities[0][top_idx]
    
    print(f"\nTop retrieved ground truth: {ground_truths[top_idx]}")
    print(f"Similarity score: {top_score:.4f}")
    
    is_correct = top_score >= threshold
    print(f"Is correct?: {is_correct}")
    
    return is_correct, ground_truths[top_idx], top_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", help="Folder with documents to ingest")
    parser.add_argument("--query", help="Query to run against the vector DB")
    parser.add_argument("--ground_truth", help="File with ground truths for correctness checking")
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
            
        if args.ground_truth:
            ground_truths = load_ground_truths(args.ground_truth)
            if ground_truths:
                print("\nChecking correctness...")
                check_correctness(answer, ground_truths)
            else:
                print("No ground truths found or file is empty.")
        else:
            print("No ground truths file provided for correctness checking.")
