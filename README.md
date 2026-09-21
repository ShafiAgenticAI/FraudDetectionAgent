# Regulatory Compliance Copilot

A local end-to-end MVP based on the supplied MVP implementation plan.

## Architecture

Streamlit
  -> FastAPI
  -> PDF parser
  -> page-aware chunks
  -> OpenAI embeddings
  -> ChromaDB
  -> grounded LLM response
  -> citations

The MVP also includes:
- compliance obligation extraction
- executive document summarization

## Setup

### 1. Create virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install packages

```powershell
pip install -r requirements.txt
```

### 3. Create `.env`

```powershell
Copy-Item .env.example .env
```

Add your OpenAI API key to `.env`.

### 4. Add a PDF

Place a PDF under:

```text
data/documents/
```

or upload it through Streamlit.

### 5. Start FastAPI

```powershell
uvicorn app.main:app --reload
```

Swagger:
http://127.0.0.1:8000/docs

### 6. Start Streamlit

Open a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit/app.py
```

## Initial real dataset

If the SEC Rule 34-96034 PDF is difficult to download, start with the official FINRA Regulatory Notice 11-19:

https://www.finra.org/sites/default/files/NoticeDocument/p123548.pdf

Save as:

```text
FINRA_4511_Books_and_Records.pdf
```

Then upload and index it from Streamlit.

## API

- GET `/health`
- POST `/api/documents/upload`
- POST `/api/documents/ingest`
- POST `/api/chat/query`
- POST `/api/analysis/summary`
- POST `/api/analysis/obligations`

## Important

This is a prototype, not legal advice or a production compliance system.

For production deployment we will later add:
- authentication/authorization
- persistent metadata database
- object storage
- OCR for scanned PDFs
- reranking
- RAG evaluation
- tracing
- rate limits
- secrets management
- deployment infrastructure
