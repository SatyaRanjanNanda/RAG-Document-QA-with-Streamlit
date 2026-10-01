# RAG Document QA with Streamlit

A **Retrieval-Augmented Generation (RAG)** app for querying your own document
collection. Built with [LangChain](https://www.langchain.com/) for retrieval and
generation, [ChromaDB](https://www.trychroma.com/) for the vector store, and
[Streamlit](https://streamlit.io/) for the interface.

**No OpenAI API key required.** The default backend runs fully local through
[Ollama](https://ollama.com/). A Gemini option is included for those who want a
hosted model.

---

## Features

- 📂 Loads PDFs, `.txt`, and `.md` from a `docs/` folder.
- 💾 **Persistent vector index** — documents are embedded once and cached in
  `.chroma/`, so restarts are instant.
- ♻️ **Auto-rebuild** — a fingerprint of the folder's contents is stored with the
  index, so the index rebuilds automatically when you add, edit, or remove files.
- 🔀 **Two providers** — local Ollama (no key) or Gemini (free tier).
- 🤝 **Small models by default** — sized to fit a 4 GB GPU / 8 GB RAM laptop.
- 💬 Chat UI with conversation history used to disambiguate follow-up questions.
- 📎 **Add documents without touching the filesystem** — browse for files in the
  sidebar, or drag them onto the chat box; both save into `docs/` and re-index.
- 🔍 Shows the source document and page for every answer.
- 🎚 Sidebar controls for provider, chat model, embedding model, retrieval depth,
  and chunking — no code edits required.
- ⌨️ A CLI (`rag.py`) for scripted or terminal use.

## Tech stack

- Python 3.10+
- `langchain` 1.x, `langchain-community`, `langchain-ollama`, `langchain-google-genai`
- `langchain-chroma` / `chromadb`
- `langchain-text-splitters`
- `streamlit`

## Quick start

```bash
git clone https://github.com/SatyaRanjanNanda/RAG-Document-QA-with-Streamlit.git
cd RAG-Document-QA-with-Streamlit

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# One-time: install Ollama (https://ollama.com/download), then pull the models
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text

# Run
streamlit run app.py
```

Open <http://localhost:8501>. Add a document, ask a question, done.

There is **no API key**. If Ollama is not running the app tells you so instead
of failing quietly.

## Setup

The quick start above covers most people. Details follow.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 1. Install Ollama (only for the default local provider)

Download from <https://ollama.com/download>, then pull the two default models:

```bash
ollama pull qwen2.5:3b-instruct   # chat   (~1.9 GB)
ollama pull nomic-embed-text     # embed  (~274 MB)
```

That is the whole setup. There is no API key, no account, and no cost.

### 2. (Optional) `.env` for Gemini

The app runs without a `.env` file. Copy `.env.example` to `.env` only if you
want to use the Gemini provider, and set a free key from
[Google AI Studio](https://aistudio.google.com/apikey):

```
GOOGLE_API_KEY=your-key-here
```

## Add documents

Three ways, all landing in `docs/`:

- **Browse files** in the sidebar under **Documents** — click it, pick one or
  more files, and they are saved and indexed automatically. This is the most
  reliable option on Streamlit 1.46–1.47, where the chat box is drag-drop only.
- **Drag onto the chat input** — drop a file onto the message box and send. You
  can send a file *with* a question in one submission, or on its own; the file is
  saved, the index refreshes, and the answer follows in the same turn.
- **Drop them in `docs/`** manually while the app is closed.

Accepted types: `.pdf`, `.txt`, `.md`. The sidebar lists everything currently
indexed, with a delete button per file.

Re-adding a file whose bytes are unchanged is skipped, so you cannot accidentally
build a duplicate. `notes (2).md` appears only when the *content* differs under
an existing name.

> **On the chat-box paperclip:** Streamlit 1.46 renders the chat attachment as a
> drag-and-drop target. The click-to-browse button inside the chat box arrived in
> a later release, so on 1.46–1.47 use the sidebar **Browse files** button. If you
> upgrade Streamlit, both work.

## Run the web app

```bash
streamlit run app.py
```

Then open http://localhost:8501.

## Use the CLI

```bash
python rag.py
```

```
Indexing docs/ with ollama (nomic-embed-text)...
Ready. Provider=ollama chat=qwen2.5:3b-instruct type 'exit' to quit.
Ask a question (or 'exit'): what is the parental leave policy?
Answer: You get 18 weeks of fully paid parental leave...
Sources: handbook.md
```

CLI defaults can be overridden in `.env`:

```
RAG_PROVIDER=ollama
RAG_CHAT_MODEL=qwen2.5:3b-instruct
RAG_EMBEDDING_MODEL=nomic-embed-text
RAG_TOP_K=4
OLLAMA_BASE_URL=http://localhost:11434
```

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, settings, chat loop |
| `rag.py` | Loading, chunking, indexing, and the retrieval + generation chain |
| `docs/` | Your source documents (git-ignored) |
| `.chroma/` | Cached vector index (git-ignored) |
| `.env.example` | Template for the optional Gemini key |

## Choosing models for a small machine

The defaults target a 4 GB GPU and fit comfortably (~2.5 GB VRAM at answer
time). If you need to go smaller, `qwen2.5:1.5b-instruct` or `gemma3:1b` both
answer adequately for grounded extraction, since the model mostly reads context
rather than reasoning over it. Avoid 70B-class models entirely on this hardware
— they will spill to system RAM and be extremely slow.

Each provider has its own Chroma collection, so switching providers or
embedding models never corrupts an existing index; the new one simply builds
alongside it.

## Notes

- Answers are grounded in the retrieved context only, and the model is
  instructed to say so when the context is insufficient. Treat output as a
  starting point, not a verified source.
- Smaller local models follow the grounding instruction well but are blunter
  than a large hosted model. If an answer looks thin, raise
  **Chunks to retrieve** before changing models.
- `nomic-embed-text` expects `search_document:` / `search_query:` prefixes;
  `PrefixedOllamaEmbeddings` in `rag.py` adds them automatically. Removing them
  noticeably degrades retrieval.
- Uploaded filenames are reduced to a safe basename and extension-checked, so an
  attachment cannot escape `docs/` or overwrite a file elsewhere.
- Uploads are capped by Streamlit's `server.maxUploadSize` (200 MB default).
  Scanned or image-only PDFs yield no extractable text and will be indexed as
  empty — use text-based PDFs.
- Increasing `Chunks to retrieve` improves recall on broad questions but costs
  more tokens and, locally, more time.
- Changing the embedding model invalidates the index; use **Rebuild index** in
  the sidebar.
- For Streamlit Cloud, Ollama is not available on the free tier. Deploy with
  `GOOGLE_API_KEY` under *Settings → Secrets*, set `RAG_PROVIDER=gemini`, and
  commit at least one document to `docs/`, since uploaded files are not
  persistent there.

## License

[MIT](LICENSE) © 2026 SatyaRanjanNanda
