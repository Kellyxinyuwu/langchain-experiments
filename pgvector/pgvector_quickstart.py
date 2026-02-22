"""
PGVector Quickstart: Load documents, create embeddings, store in PostgreSQL, and run similarity search.

Flow: Load text → Split into chunks → Embed chunks → Store in PGVector → Query by similarity
See pgvector/README.md for full setup (PostgreSQL, pgvector extension, .env config).
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores.pgvector import PGVector
from pgvector_service import PgvectorService
import os
import time

load_dotenv()

# -----------------------------------------------------------------------------
# STEP 1: Choose Embedding Model
# -----------------------------------------------------------------------------
# OpenAI: Requires OPENAI_API_KEY. May be restricted in some regions.
# HuggingFace: Local model, no API key. Set USE_LOCAL_EMBEDDINGS=true in .env
if os.getenv("USE_LOCAL_EMBEDDINGS", "").lower() in ("true", "1", "yes"):
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Using local HuggingFace embeddings (all-MiniLM-L6-v2)\n")
else:
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

# -----------------------------------------------------------------------------
# STEP 2: Load & Split Documents
# -----------------------------------------------------------------------------
# Build path relative to this script (works regardless of where you run from)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(
    SCRIPT_DIR,
    "..",
    "data",
    "The Project Gutenberg eBook of A Christmas Carol in Prose; Being a Ghost Story of Christmas.txt",
)

# Load the text file into LangChain Document objects
loader = TextLoader(DATA_FILE)
documents = loader.load()

# Split into chunks: 2000 chars each, no overlap. Smaller chunks = more precise search.
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Query we'll use for similarity search
query = "The Project Gutenberg eBook of A Christmas Carol in Prose; Being a Ghost Story of Christmas"


# -----------------------------------------------------------------------------
# STEP 3: Pinecone Comparison (Optional)
# -----------------------------------------------------------------------------
# Compares to Pinecone, a cloud vector DB. Skipped if PINECONE_API_KEY not set.


def calculate_average_execution_time(func, *args, **kwargs):
    total_execution_time = 0
    num_runs = 10
    for _ in range(num_runs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Execute the function with its arguments
        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
    average_execution_time = round(total_execution_time / num_runs, 2)
    print(result)
    print(
        f"\nThe function took an average of {average_execution_time} seconds to execute."
    )
    return


# Create Pinecone index and run similarity search (only if API key is set)
if os.getenv("PINECONE_API_KEY"):
    from pinecone import Pinecone as PineconeClient, ServerlessSpec

    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "demo-index"
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    pinecone_docsearch = Pinecone.from_existing_index(index_name, embeddings)

    def run_query_pinecone(docsearch, query):
        docs = docsearch.similarity_search(query, k=4)
        return docs[0].page_content

    calculate_average_execution_time(
        run_query_pinecone, docsearch=pinecone_docsearch, query=query
    )
else:
    print("Skipping Pinecone (PINECONE_API_KEY not set). Continuing with PGVector...\n")


# -----------------------------------------------------------------------------
# STEP 4: Set Up PGVector (PostgreSQL + pgvector extension)
# -----------------------------------------------------------------------------
# Prerequisites: PostgreSQL running, pgvector extension installed, database created.
# See pgvector/README.md for full setup instructions.

COLLECTION_NAME = "The Project Gutenberg eBook of A Christmas Carol in Prose"

# Build connection string from env vars (PGVECTOR_USER, PGVECTOR_PASSWORD, etc.)
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "pgvector"),
    user=os.environ.get("PGVECTOR_USER", "postgres"),
    password=os.environ.get("PGVECTOR_PASSWORD", "postres"),
)

# -----------------------------------------------------------------------------
# STEP 5: Store Documents in PGVector
# -----------------------------------------------------------------------------
# This creates the langchain_pg_collection and langchain_pg_embedding tables,
# generates embeddings for each chunk, and inserts them into PostgreSQL.
db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=False,
)

# Load the store for querying (connects to existing collection)
pgvector_docsearch = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

# -----------------------------------------------------------------------------
# STEP 6: Run Similarity Search
# -----------------------------------------------------------------------------
# Converts query to embedding, finds nearest chunks by cosine similarity,
# returns top k results. Runs 10 times to measure average execution time.


def run_query_pgvector(docsearch, query):
    docs = docsearch.similarity_search(query, k=4)
    result = docs[0].page_content
    return result


calculate_average_execution_time(
    run_query_pgvector, docsearch=pgvector_docsearch, query=query
)


# -----------------------------------------------------------------------------
# STEP 7: Add a Second Collection
# -----------------------------------------------------------------------------
# Load Romeo and Juliet, split with different chunk size (1000), store as new collection.
loader = TextLoader(
    os.path.join(SCRIPT_DIR, "..", "data", "The Project Gutenberg eBook of Romeo and Juliet.txt")
)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
new_docs = text_splitter.split_documents(documents)


COLLECTION_NAME_2 = "The Project Gutenberg eBook of Romeo and Juliet"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=new_docs,
    collection_name=COLLECTION_NAME_2,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=False,
)


# -----------------------------------------------------------------------------
# STEP 8: Query Across Collections (PgvectorService)
# -----------------------------------------------------------------------------
# PgvectorService uses raw SQL to search across all collections and return
# results with similarity scores. Useful when you have multiple document sets.
pg = PgvectorService(CONNECTION_STRING, embeddings=embeddings)


def run_query_multi_pgvector(docsearch, query):
    # Returns [(Document, score), ...]; docs[0][0] = top Document, docs[0][1] = its score
    docs = docsearch.custom_similarity_search_with_scores(query, k=4)
    result = docs[0][0].page_content
    print(result)


run_query_multi_pgvector(pg, query)

# -----------------------------------------------------------------------------
# STEP 9: Delete Collections
# -----------------------------------------------------------------------------
# Removes the collections and their embeddings from the database.
pg.delete_collection(COLLECTION_NAME)
pg.delete_collection(COLLECTION_NAME_2)

# -----------------------------------------------------------------------------
# STEP 10: Re-create Collection (Update)
# -----------------------------------------------------------------------------
# Re-adds the Christmas Carol collection. Useful for refreshing data.
pg.update_collection(docs=docs, collection_name=COLLECTION_NAME)
