"""
PgvectorService: Custom service for querying PGVector with raw SQL.

Use this when you need:
- Cross-collection search (search across multiple document sets)
- Custom similarity queries with scores
- Collection management (create, update, delete)
"""

from langchain_community.vectorstores.pgvector import (
    PGVector,
    _get_embedding_collection_store,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import logging
import os


# LangChain's internal model for the langchain_pg_embedding table
EmbeddingStore = _get_embedding_collection_store()[0]


def _get_embeddings(embeddings=None) -> Embeddings:
    """Get embeddings - use provided or fall back to OpenAI/local based on env."""
    if embeddings is not None:
        return embeddings
    if os.getenv("USE_LOCAL_EMBEDDINGS", "").lower() in ("true", "1", "yes"):
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings()


class PgvectorService:
    """
    Service for interacting with PGVector using SQLAlchemy and raw SQL.
    Pass embeddings to match the model used when creating the collection.
    """

    def __init__(self, connection_string, embeddings=None):
        load_dotenv()
        self.embeddings = _get_embeddings(embeddings)
        self.cnx = connection_string
        self.collections = []
        self.engine = create_engine(self.cnx)
        self.EmbeddingStore = EmbeddingStore

    # --- Search ---

    def get_vector(self, text):
        """Convert text to embedding vector (for similarity comparison)."""
        return self.embeddings.embed_query(text)

    def custom_similarity_search_with_scores(self, query, k=3):
        """
        Search across ALL collections using cosine similarity.
        Returns list of (Document, score) tuples. Lower distance = higher similarity.
        """
        query_vector = self.get_vector(query)

        with Session(self.engine) as session:
            # Cosine distance: 0 = identical, 2 = opposite. Order by ascending = most similar first.
            cosine_distance = self.EmbeddingStore.embedding.cosine_distance(
                query_vector
            ).label("distance")

            # Querying the EmbeddingStore table
            results = (
                session.query(
                    self.EmbeddingStore.document,
                    self.EmbeddingStore.custom_id,
                    cosine_distance,
                )
                .order_by(cosine_distance.asc())
                .limit(k)
                .all()
            )
        # Convert distance to similarity score: 1 - distance (higher = more similar)
        docs = [(Document(page_content=result[0]), 1 - result[2]) for result in results]

        return docs

    # --- Collection Management ---

    def update_pgvector_collection(
        self, docs, collection_name, overwrite=False
    ) -> None:
        """
        Create or replace a collection. Generates embeddings and stores in langchain_pg_embedding.
        overwrite=True: Delete existing collection first (use when refreshing data).
        """
        logging.info(f"Creating new collection: {collection_name}")
        with self.engine.connect() as connection:
            pgvector = PGVector.from_documents(
                embedding=self.embeddings,
                documents=docs,
                collection_name=collection_name,
                connection_string=self.cnx,
                connection=connection,
                pre_delete_collection=overwrite,
            )

    def get_collections(self) -> list:
        """List all collection names in the database."""
        with self.engine.connect() as connection:
            try:
                query = text("SELECT * FROM public.langchain_pg_collection")
                result = connection.execute(query)
                collections = [row[0] for row in result]
            except:
                # If the table doesn't exist, return an empty list
                collections = []
        return collections

    def update_collection(self, docs, collection_name):
        """Add or replace documents in a collection. Overwrites if collection exists."""
        logging.info(f"Updating collection: {collection_name}")
        collections = self.get_collections()

        if docs is not None:
            overwrite = collection_name in collections
            self.update_pgvector_collection(docs, collection_name, overwrite)

    def delete_collection(self, collection_name):
        """Remove a collection and all its embeddings from the database."""
        logging.info(f"Deleting collection: {collection_name}")
        with self.engine.connect() as connection:
            pgvector = PGVector(
                collection_name=collection_name,
                connection_string=self.cnx,
                connection=connection,
                embedding_function=self.embeddings,
            )
            pgvector.delete_collection()
