# PGVector Tutorial: Complete Setup Guide

A step-by-step tutorial for setting up PostgreSQL with pgvector, running the LangChain quickstart, and connecting to the database from Cursor. This guide covers everything from installation to querying your vector store.

---

## Table of Contents

1. [Overview](#overview)
2. [Install PostgreSQL & pgvector](#1-install-postgresql--pgvector)
3. [Create the Database](#2-create-the-database)
4. [Set Up Credentials](#3-set-up-credentials)
5. [Configure Environment Variables](#4-configure-environment-variables)
6. [Install Python Dependencies](#5-install-python-dependencies)
7. [Run the Quickstart Script](#6-run-the-quickstart-script)
8. [Connect with SQLTools (Cursor)](#7-connect-with-sqltools-cursor)
9. [Query Your Data](#8-query-your-data)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This project uses **PGVector** to store document embeddings for semantic search. The flow is:

1. Load text documents (e.g., A Christmas Carol)
2. Split them into chunks
3. Generate embeddings (OpenAI or local HuggingFace)
4. Store embeddings in PostgreSQL
5. Query with similarity search

**Pinecone vs PGVector comparison:** The quickstart script uses **both** Pinecone and PGVector to compare them. It runs the same documents and queries against each, so you can compare a managed cloud vector DB (Pinecone) with a self-hosted open-source option (PGVector). Pinecone is optional—if `PINECONE_API_KEY` is not set, the script skips it and continues with PGVector only.

**Key notes:**
- **PGVector is open source** – Free to use, self-hosted, no vendor lock-in.
- **Perform similar search over all data** – PGVector lets you run similarity search across your entire dataset (or across multiple collections via `PgvectorService`).

You'll need: PostgreSQL, the pgvector extension, and a way to connect (we use SQLTools in Cursor).

---

## 1. Install PostgreSQL & pgvector

### Option A: PostgreSQL 17 (Recommended – pgvector works out of the box)

Homebrew's pgvector formula supports PostgreSQL 14 and 17, but **not 16**. Use PostgreSQL 17:

```bash
# Install PostgreSQL 17 and pgvector
brew install postgresql@17 pgvector

# Add to PATH (Apple Silicon Mac)
echo 'export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Start the service
brew services start postgresql@17
```

### Option B: PostgreSQL 16 (Build pgvector from source)

If you need PostgreSQL 16, build pgvector manually:

```bash
brew install postgresql@16
brew services start postgresql@16

# Build pgvector from source
cd /tmp
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector
make PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config
make PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config install

brew services restart postgresql@16
```

### Verify installation

```bash
psql --version
# Should show: psql (PostgreSQL) 17.x or 16.x
```

---

## 2. Create the Database

Create the `pgvector` database and enable the vector extension:

```bash
# Replace 'kellywu' with your macOS username (Homebrew uses it as the default PostgreSQL user)
psql -U kellywu -h localhost -d postgres -c "CREATE DATABASE pgvector;"

# Enable the pgvector extension
psql -U kellywu -h localhost -d pgvector -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

> **Finding your username:** Your PostgreSQL user is typically your macOS login name (e.g., from the terminal prompt `kellywu@KellydeMacBook-Air`).

---

## 3. Set Up Credentials

### Default: Trust authentication (no password)

Homebrew PostgreSQL often uses trust auth for local connections. Try connecting without a password first.

### If your SQL client requires a password

Set a password for your user:

```bash
psql -U kellywu -h localhost -d postgres -c "ALTER USER kellywu PASSWORD 'your_password';"
```

Replace `your_password` with your chosen password. The literal string you put in the command becomes your password.

---

## 4. Configure Environment Variables

Create a `.env` file in the **project root** (not inside `pgvector/`):

```env
# PostgreSQL / PGVector
PGVECTOR_USER=kellywu
PGVECTOR_PASSWORD=your_password
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DATABASE=pgvector

# Optional: Use local embeddings if OpenAI is restricted in your region
USE_LOCAL_EMBEDDINGS=true
```

- Replace `kellywu` with your PostgreSQL username.
- Replace `your_password` with your password (or leave empty for trust auth).
- **`USE_LOCAL_EMBEDDINGS=true`** – Use this if you get `unsupported_country_region_territory` from OpenAI. It switches to HuggingFace's local model (no API key needed).

---

## 5. Install Python Dependencies

From the project root, with your virtual environment activated:

```bash
cd /path/to/langchain-experiments
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows

pip install -r requirements.txt
pip install sentence-transformers  # Required if using USE_LOCAL_EMBEDDINGS
```

Key packages: `langchain`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`, `pgvector`, `psycopg2`, `sentence-transformers` (for local embeddings).

---

## 6. Run the Quickstart Script

This script loads documents, creates embeddings, and stores them in PostgreSQL:

```bash
python3 pgvector/pgvector_quickstart.py
```

**What it does:**

1. Loads "A Christmas Carol" from `data/`
2. Splits into chunks (2000 chars)
3. Generates embeddings (OpenAI or HuggingFace)
4. Stores in PGVector
5. Runs similarity search
6. Adds "Romeo and Juliet" as a second collection
7. Demonstrates multi-collection queries
8. Deletes and re-creates collections

**Expected output:**

- `Skipping Pinecone (PINECONE_API_KEY not set)...` – Normal if you don't use Pinecone
- `Using local HuggingFace embeddings...` – If `USE_LOCAL_EMBEDDINGS=true`
- Similarity search results and execution times

**Tables created:**

- `langchain_pg_collection` – Collection metadata
- `langchain_pg_embedding` – Document chunks, embeddings, and metadata

---

## 7. Connect with SQLTools (Cursor)

SQLTools lets you browse and query PostgreSQL from Cursor (or VS Code).

### Install extensions

1. Open Extensions (`Cmd+Shift+X` / `Ctrl+Shift+X`)
2. Search for **SQLTools** → Install
3. Search for **SQLTools PostgreSQL Driver** → Install

### Add a connection

1. `Cmd+Shift+P` (or `Ctrl+Shift+P`) → **SQLTools: Add New Connection**
2. Choose **PostgreSQL**
3. Enter:
   - **Host:** `localhost`
   - **Port:** `5432`
   - **Database:** `pgvector`
   - **Username:** `kellywu` (or your user)
   - **Password:** your password (or leave empty)
4. Connect

### Session file

SQLTools creates `localhost.session.sql` as a scratchpad for queries. Write SQL there and run it with **Cmd+Enter** (or the "Run on active connection" button).

---

## 8. Query Your Data

### Example queries in SQLTools

```sql
-- View document chunks
SELECT * FROM langchain_pg_embedding LIMIT 10;

-- View collections
SELECT * FROM langchain_pg_collection;

-- Count embeddings per collection
SELECT cmetadata->>'source' as source, COUNT(*) 
FROM langchain_pg_embedding 
GROUP BY cmetadata->>'source';
```

### Common error: `relation "langchain_pg_embedding" does not exist`

This means the tables haven't been created yet. Run the quickstart script first:

```bash
python3 pgvector/pgvector_quickstart.py
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `psql: command not found` | Add PostgreSQL to PATH (see step 1) |
| `database "pgvector" does not exist` | Run `CREATE DATABASE pgvector;` |
| `extension "vector" is not available` | Use PostgreSQL 17, or build pgvector from source for PostgreSQL 16 |
| `relation "langchain_pg_embedding" does not exist` | Run `python3 pgvector/pgvector_quickstart.py` first |
| `unsupported_country_region_territory` (OpenAI) | Add `USE_LOCAL_EMBEDDINGS=true` to `.env` and install `sentence-transformers` |
| `ModuleNotFoundError: langchain.document_loaders` | Update imports: use `langchain_community`, `langchain_openai`, `langchain_text_splitters` |
| `FileNotFoundError` for data files | Paths are now script-relative; run from project root |
| `pinecone.init` AttributeError | Pinecone section is optional; script skips it if no API key |
| Connection refused | Start PostgreSQL: `brew services start postgresql@17` |

---

## Alternative Database Clients

| Tool | Notes |
|------|-------|
| **psql** | `psql -U kellywu -h localhost -d pgvector` |
| **pgAdmin** | Free PostgreSQL GUI |
| **DBeaver** | Universal database tool |
| **TablePlus** | Native macOS SQL client |
| **DataGrip** | JetBrains database IDE (paid) |

---

## Project Structure

```
pgvector/
├── README.md              # This tutorial
├── pgvector_quickstart.py # Main script: load docs, embed, store, query
└── pgvector_service.py    # PgvectorService class for custom queries
```

---

## Summary Checklist

- [ ] PostgreSQL 17 (or 16 + pgvector from source) installed and running
- [ ] `pgvector` database created with `vector` extension
- [ ] Password set (if needed for SQL client)
- [ ] `.env` configured with `PGVECTOR_*` and optionally `USE_LOCAL_EMBEDDINGS`
- [ ] Python dependencies installed (`pip install -r requirements.txt sentence-transformers`)
- [ ] Quickstart run successfully (`python3 pgvector/pgvector_quickstart.py`)
- [ ] SQLTools connected and able to query `langchain_pg_embedding`
