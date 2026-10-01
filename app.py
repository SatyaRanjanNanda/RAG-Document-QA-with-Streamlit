"""Streamlit front end for the RAG document QA system."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import (
    ALLOWED_SUFFIXES,
    CHAT_MODEL_CHOICES,
    DEFAULT_CHAT_MODELS,
    DEFAULT_DOCS_DIR,
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_PERSIST_DIR,
    DEFAULT_PROVIDER,
    EMBEDDING_MODEL_CHOICES,
    PROVIDER_GEMINI,
    PROVIDER_OLLAMA,
    PROVIDERS,
    answer_question,
    build_index,
    collection_for,
    delete_document,
    docs_fingerprint,
    is_index_populated,
    list_documents,
    open_index,
    save_upload,
)

st.set_page_config(page_title="RAG Document QA", page_icon="📄", layout="wide")

load_dotenv()

FINGERPRINT_FILE = DEFAULT_PERSIST_DIR / "index_fingerprint.txt"

PROVIDER_LABELS = {
    PROVIDER_OLLAMA: "Ollama (local, no API key)",
    PROVIDER_GEMINI: "Gemini (free tier, API key)",
}


@st.cache_resource(show_spinner=False)
def get_store(persist_dir: str, embedding_model: str, collection: str, provider: str):
    return open_index(Path(persist_dir), embedding_model, collection, provider=provider)


def index_signature() -> str:
    return docs_fingerprint(
        DEFAULT_DOCS_DIR,
        st.session_state.chunk_size,
        st.session_state.chunk_overlap,
        provider=st.session_state.provider,
        embedding_model=st.session_state.embedding_model,
    )


def read_fingerprint() -> str | None:
    if FINGERPRINT_FILE.is_file():
        return FINGERPRINT_FILE.read_text(encoding="utf-8").strip()
    return None


def write_fingerprint(value: str) -> None:
    DEFAULT_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_FILE.write_text(value, encoding="utf-8")


def ensure_index(force_rebuild: bool = False) -> bool:
    """Build the index if it is missing, stale, or a rebuild was requested."""
    if not DEFAULT_DOCS_DIR.is_dir():
        st.error(f"Missing `{DEFAULT_DOCS_DIR}/` folder. Add PDFs, .txt or .md files and reload.")
        return False

    collection = collection_for(st.session_state.provider, st.session_state.embedding_model)
    try:
        store = get_store(
            str(DEFAULT_PERSIST_DIR),
            st.session_state.embedding_model,
            collection,
            st.session_state.provider,
        )
    except Exception as exc:
        st.error(f"Could not open the vector store: {exc}")
        return False

    signature = index_signature()
    needs_build = (
        force_rebuild
        or not is_index_populated(store)
        or read_fingerprint() != signature
    )
    if not needs_build:
        st.session_state.store = store
        return True

    with st.spinner("Indexing documents. This runs once per document change..."):
        try:
            st.session_state.store = build_index(
                docs_dir=DEFAULT_DOCS_DIR,
                persist_dir=DEFAULT_PERSIST_DIR,
                embedding_model=st.session_state.embedding_model,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                collection_name=collection,
                provider=st.session_state.provider,
            )
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            return False
    write_fingerprint(signature)
    return True


def ingest_uploads(uploads) -> tuple[list[str], list[str], list[str]]:
    """Save uploads into docs/ and re-index. Returns (added, unchanged, errors).

    Shared by the chat box and the sidebar browser so both behave identically.
    """
    added: list[str] = []
    unchanged: list[str] = []
    errors: list[str] = []

    for upload in uploads:
        # A uploader keeps re-reporting the same file on later reruns, so skip
        # anything this session already handled.
        token = f"{upload.name}:{hashlib.sha256(upload.getvalue()).hexdigest()}"
        if token in st.session_state.uploads_seen:
            continue
        st.session_state.uploads_seen.add(token)

        try:
            path, is_new = save_upload(upload.getvalue(), upload.name, DEFAULT_DOCS_DIR)
        except Exception as exc:
            errors.append(f"{upload.name}: {exc}")
            continue
        (added if is_new else unchanged).append(path.name)

    if added and ensure_index(force_rebuild=True):
        st.success(f"Saved to `{DEFAULT_DOCS_DIR}/`: {', '.join(added)}")
    for message in errors:
        st.error(message)
    if unchanged:
        st.info(f"Already indexed, unchanged: {', '.join(unchanged)}")

    return added, unchanged, errors


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, doc in enumerate(sources, start=1):
            source = Path(str(doc.metadata.get("source", "unknown"))).name
            page = doc.metadata.get("page")
            location = f"{source} — page {page + 1}" if isinstance(page, int) else source
            preview = doc.page_content.strip().replace("\n", " ")
            st.markdown(f"**[{i}] {location}**\n\n> {preview[:400]}{'…' if len(preview) > 400 else ''}")


def main() -> None:
    st.title("🔎 RAG Q&A with Your Documents")
    st.caption(f"Indexing documents from `{DEFAULT_DOCS_DIR}/`")

    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("uploads_seen", set())
    st.session_state.setdefault("chunk_size", 1000)
    st.session_state.setdefault("chunk_overlap", 200)
    st.session_state.setdefault("provider", DEFAULT_PROVIDER)
    st.session_state.setdefault("embedding_model", DEFAULT_EMBEDDING_MODELS[DEFAULT_PROVIDER])
    st.session_state.setdefault("chat_model", DEFAULT_CHAT_MODELS[DEFAULT_PROVIDER])

    with st.sidebar:
        st.header("Settings")

        provider_label = st.selectbox(
            "Model provider",
            [PROVIDER_LABELS[p] for p in PROVIDERS],
            index=PROVIDERS.index(st.session_state.provider),
        )
        provider = next(p for p, lbl in PROVIDER_LABELS.items() if lbl == provider_label)
        if provider != st.session_state.provider:
            # Model choices are provider-specific, so reset both to their defaults.
            st.session_state.provider = provider
            st.session_state.chat_model = DEFAULT_CHAT_MODELS[provider]
            st.session_state.embedding_model = DEFAULT_EMBEDDING_MODELS[provider]
            get_store.clear()

        if provider == PROVIDER_GEMINI and not (
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        ):
            st.warning(
                "No `GOOGLE_API_KEY` in your `.env`. Get a free key at "
                "[Google AI Studio](https://aistudio.google.com/apikey), or switch "
                "back to Ollama to run with no key at all."
            )

        chat_choices = CHAT_MODEL_CHOICES[provider]
        st.session_state.chat_model = st.selectbox(
            "Chat model",
            chat_choices,
            index=(
                chat_choices.index(st.session_state.chat_model)
                if st.session_state.chat_model in chat_choices
                else 0
            ),
        )
        embed_choices = EMBEDDING_MODEL_CHOICES[provider]
        st.session_state.embedding_model = st.selectbox(
            "Embedding model",
            embed_choices,
            index=(
                embed_choices.index(st.session_state.embedding_model)
                if st.session_state.embedding_model in embed_choices
                else 0
            ),
        )
        st.session_state.top_k = st.slider("Chunks to retrieve", 1, 10, 4)
        st.session_state.chunk_size = st.number_input(
            "Chunk size", 200, 4000, st.session_state.chunk_size, step=100
        )
        st.session_state.chunk_overlap = st.number_input(
            "Chunk overlap", 0, 1000, st.session_state.chunk_overlap, step=50
        )
        st.caption("Chunk settings apply on the next index rebuild.")

        st.divider()
        st.header("Documents")

        uploads = st.file_uploader(
            "Browse files",
            type=[s.lstrip(".") for s in sorted(ALLOWED_SUFFIXES)],
            accept_multiple_files=True,
            key="browse_uploads",
            help="Saved into docs/ and indexed automatically.",
        )
        if uploads:
            with st.spinner("Saving and indexing..."):
                ingest_uploads(uploads)

        documents = list_documents(DEFAULT_DOCS_DIR)
        if not documents:
            st.caption(
                "No documents yet. Browse for a file above, or drag one onto the chat box."
            )
        else:
            st.caption(f"{len(documents)} file(s) in `{DEFAULT_DOCS_DIR}/`")
            for entry in documents:
                name = str(entry["name"])
                size_kb = int(entry["size"]) / 1024
                with st.container(border=True):
                    col_text, col_del = st.columns([6, 1], gap="small")
                    col_text.markdown(f"**{name}**  \n{size_kb:.0f} KB")
                    if col_del.button(
                        "✕", key=f"del_{name}", use_container_width=True, help=f"Delete {name}"
                    ):
                        try:
                            if delete_document(name, DEFAULT_DOCS_DIR):
                                st.session_state.force_rebuild = True
                                st.toast(f"Deleted {name}")
                                st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

        st.divider()
        st.button("Rebuild index", on_click=lambda: st.session_state.update(force_rebuild=True))
        st.button("Clear chat", on_click=lambda: st.session_state.update(chat=[]))
        if st.session_state.chat:
            st.download_button(
                "Download chat",
                "\n\n".join(f"Q: {q}\nA: {a}" for q, a in st.session_state.chat),
                file_name="rag-chat.md",
                mime="text/markdown",
            )

    if not ensure_index(st.session_state.pop("force_rebuild", False)):
        st.stop()

    if not st.session_state.chat:
        st.info("Ask a question below. Answers are grounded only in the indexed documents.")

    for question, answer in st.session_state.chat:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    submission = st.chat_input(
        "Ask a question, or drag a document onto this box",
        accept_file="multiple",
        file_type=[s.lstrip(".") for s in sorted(ALLOWED_SUFFIXES)],
    )
    if not submission:
        return

    # With accept_file enabled the widget returns a dict-like, not a plain string.
    files = list(getattr(submission, "files", []) or [])
    question = (getattr(submission, "text", "") or "").strip()

    with st.chat_message("user"):
        for upload in files:
            st.write(f"📎 {upload.name}")
        if question:
            st.write(question)

    if files:
        with st.spinner(f"Saving and indexing {len(files)} file(s)..."):
            ingest_uploads(files)

    if not question:
        # Attachment-only submission: the file is saved, nothing to answer yet.
        return

    st.session_state.chat.append((question, ""))
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = st.session_state.chat[:-1]
                answer, sources = answer_question(
                    store=st.session_state.store,
                    question=question,
                    history=history,
                    model=st.session_state.chat_model,
                    k=st.session_state.top_k,
                    provider=st.session_state.provider,
                )
            except Exception as exc:
                st.error(f"Question failed: {exc}")
                st.session_state.chat.pop()
                return
        st.write(answer)
        render_sources(sources)

    st.session_state.chat[-1] = (question, answer)


if __name__ == "__main__":
    main()
