"""Core RAG pipeline: load documents, build/refresh a Chroma index, answer questions.

Two LLM backends are supported, both through LangChain runnables:

* ``ollama``  - fully local, no API key required (default).
* ``gemini``  - Google's free tier, needs ``GOOGLE_API_KEY``.

Model defaults are deliberately small so the pipeline runs on a 4 GB GPU.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_PERSIST_DIR = Path(".chroma")
DEFAULT_COLLECTION = "rag_documents"

PROVIDER_OLLAMA = "ollama"
PROVIDER_GEMINI = "gemini"
PROVIDERS = (PROVIDER_OLLAMA, PROVIDER_GEMINI)
DEFAULT_PROVIDER = PROVIDER_OLLAMA

# Defaults are chosen for a 4 GB GPU / 8 GB RAM machine. qwen2.5:3b-instruct is
# ~1.9 GB and runs fully on GPU; nomic-embed-text is 274 MB.
DEFAULT_CHAT_MODELS = {
    PROVIDER_OLLAMA: "qwen2.5:3b-instruct",
    PROVIDER_GEMINI: "gemini-2.0-flash",
}
DEFAULT_EMBEDDING_MODELS = {
    PROVIDER_OLLAMA: "nomic-embed-text",
    PROVIDER_GEMINI: "models/text-embedding-004",
}
CHAT_MODEL_CHOICES = {
    PROVIDER_OLLAMA: [
        "qwen2.5:3b-instruct",
        "qwen2.5:1.5b-instruct",
        "llama3.2:3b",
        "gemma3:1b",
        "phi3.5:3.8b",
    ],
    PROVIDER_GEMINI: [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ],
}
EMBEDDING_MODEL_CHOICES = {
    PROVIDER_OLLAMA: ["nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed"],
    PROVIDER_GEMINI: ["models/text-embedding-004", "models/embedding-001"],
}

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def ollama_base_url() -> str:
    """Read the base URL at call time so .env values are picked up after loading."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)

SYSTEM_PROMPT = """You are a careful document question-answering assistant.

Answer the question using only the context below. Follow these rules:
- If the context does not contain the answer, say so plainly and suggest what to look for.
- Never invent facts, citations, or details that are not in the context.
- Keep answers concise and factual. Use short bullet points for lists.
- When the context supports it, mention the source file it came from.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Conversation so far:\n{history}\n\n"
            "Context retrieved from the documents:\n{context}\n\n"
            "Question: {question}",
        ),
    ]
)

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}

# Glob patterns per file type so the folder can mix formats.
LOADER_GLOBS = {
    "**/*.pdf": ("pdf", PyPDFLoader),
    "**/*.txt": ("text", TextLoader),
    "**/*.md": ("text", TextLoader),
}


class PrefixedOllamaEmbeddings(Embeddings):
    """Ollama embeddings with the task prefixes nomic-embed-text expects.

    Without these, retrieval quality on nomic-embed-text drops noticeably.
    mxbai/arctic models take no prefix and pass straight through.
    """

    NEEDS_PREFIX = "nomic-embed-text"

    def __init__(self, model: str, base_url: str | None = None) -> None:
        from langchain_ollama import OllamaEmbeddings

        base_url = base_url or ollama_base_url()

        self._inner = OllamaEmbeddings(model=model, base_url=base_url)
        self._use_prefix = model.split(":")[0] == self.NEEDS_PREFIX

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._use_prefix:
            texts = [f"search_document: {t}" for t in texts]
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if self._use_prefix:
            text = f"search_query: {text}"
        return self._inner.embed_query(text)


def get_llm(provider: str, model: str):
    """Build a chat model runnable for the given provider."""
    if provider == PROVIDER_OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=ollama_base_url(),
            temperature=0,
            keep_alive="10m",
            num_ctx=4096,
        )
    if provider == PROVIDER_GEMINI:
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            raise ValueError(
                "Gemini needs a free API key: set GOOGLE_API_KEY in .env "
                "(get one at https://aistudio.google.com/apikey). "
                "Switch the provider to Ollama to run with no key at all."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        )
    raise ValueError(f"Unknown provider '{provider}'. Choose one of {PROVIDERS}.")


def get_embeddings(provider: str, model: str) -> Embeddings:
    """Build an embeddings object for the given provider."""
    if provider == PROVIDER_OLLAMA:
        return PrefixedOllamaEmbeddings(model, ollama_base_url())
    if provider == PROVIDER_GEMINI:
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            raise ValueError(
                "Gemini needs a free API key: set GOOGLE_API_KEY in .env "
                "(get one at https://aistudio.google.com/apikey). "
                "Switch the provider to Ollama to run with no key at all."
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        )
    raise ValueError(f"Unknown provider '{provider}'. Choose one of {PROVIDERS}.")


def check_ollama_running() -> None:
    """Fail early with an actionable message when the local daemon is down."""
    import urllib.error
    import urllib.request

    base_url = ollama_base_url()
    if base_url.startswith("https://"):
        return
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=2).close()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. Install it from "
            "https://ollama.com/download and start it, or switch the provider "
            "to Gemini in the sidebar."
        ) from exc


def save_upload(data: bytes, original_name: str, docs_dir: Path) -> Tuple[Path, bool]:
    """Write an uploaded file into docs_dir and return (path, is_new_content).

    The name is sanitised to its basename so an upload cannot escape the folder
    or overwrite a path outside it. Uploading identical bytes again is treated
    as a no-op, so re-adding a file does not needlessly invalidate the index.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)

    name = Path(original_name or "").name.strip()
    name = re.sub(r"[^\w.\- ]+", "_", name).strip(". ")
    if not name:
        raise ValueError("That file has no usable name.")

    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix or 'none'}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}."
        )

    target = docs_dir / name
    if target.is_file() and hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(data).digest():
        return target, False

    stem, ext = target.stem, target.suffix
    counter = 2
    while target.exists():
        target = docs_dir / f"{stem} ({counter}){ext}"
        counter += 1

    target.write_bytes(data)
    return target, True


def list_documents(docs_dir: Path) -> List[Dict[str, object]]:
    """Describe each indexable document for the sidebar, newest first."""
    if not docs_dir.is_dir():
        return []
    entries: List[Dict[str, object]] = []
    for path in docs_dir.rglob("*"):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        stat = path.stat()
        entries.append(
            {
                "name": str(path.relative_to(docs_dir)),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return sorted(entries, key=lambda e: e["mtime"], reverse=True)


def delete_document(relative_name: str, docs_dir: Path) -> bool:
    """Delete one document from docs_dir, refusing paths outside it."""
    target = (docs_dir / relative_name).resolve()
    if not str(target).startswith(str(docs_dir.resolve())):
        raise ValueError("Refusing to delete a path outside the docs folder.")
    if not target.is_file():
        return False
    target.unlink()
    return True


def load_documents(docs_dir: Path) -> List[Document]:
    """Load every supported document under docs_dir."""
    if not docs_dir.is_dir():
        raise FileNotFoundError(
            f"Document folder '{docs_dir}' does not exist. Create it and add PDFs."
        )

    documents: List[Document] = []
    for pattern, (kind, loader_cls) in LOADER_GLOBS.items():
        loader = DirectoryLoader(
            str(docs_dir), glob=pattern, loader_cls=loader_cls, show_progress=False
        )
        try:
            documents.extend(loader.load())
        except Exception:  # a single unreadable file should not break indexing
            continue

    for doc in documents:
        if doc.page_content and not doc.page_content.strip():
            doc.page_content = "(no extractable text)"
        doc.metadata["filetype"] = _guess_filetype(doc)
    return documents


def _guess_filetype(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    suffix = Path(source).suffix.lower()
    return {".pdf": "pdf", ".txt": "text", ".md": "text"}.get(suffix, "unknown")


def split_documents(
    documents: Sequence[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return splitter.split_documents(list(documents))


def docs_fingerprint(
    docs_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    provider: str = DEFAULT_PROVIDER,
    embedding_model: str | None = None,
) -> str:
    """Hash the indexed inputs so a stale index can be detected and rebuilt.

    The embedding model and provider are part of the hash because vectors from
    one model are meaningless to another, and the collection name is derived
    from them separately so both indexes can coexist.
    """
    digest = hashlib.sha256()
    digest.update(f"{chunk_size}:{chunk_overlap}:{provider}:{embedding_model}".encode())
    if docs_dir.is_dir():
        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            digest.update(str(path.relative_to(docs_dir)).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(int(stat.st_mtime)).encode())
    return digest.hexdigest()


def collection_for(provider: str, embedding_model: str) -> str:
    """Namespace the collection by provider + embedding model to avoid clashes."""
    tag = f"{provider}-{embedding_model}".lower()
    safe = "".join(c if c.isalnum() else "_" for c in tag).strip("_")
    return f"{DEFAULT_COLLECTION}_{safe}"


def build_index(
    docs_dir: Path,
    persist_dir: Path,
    embedding_model: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection_name: str | None = None,
    provider: str = DEFAULT_PROVIDER,
) -> Chroma:
    """Embed docs_dir and write a fresh persistent Chroma index."""
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODELS[provider]
    collection_name = collection_name or collection_for(provider, embedding_model)

    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(
            f"No supported documents found in '{docs_dir}'. Add .pdf, .txt or .md files."
        )

    chunks = split_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("Documents produced no chunks. Check that text is extractable.")

    if provider == PROVIDER_OLLAMA:
        check_ollama_running()

    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings(provider, embedding_model)
    Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    ).delete_collection()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )


def open_index(
    persist_dir: Path,
    embedding_model: str | None = None,
    collection_name: str | None = None,
    provider: str = DEFAULT_PROVIDER,
) -> Chroma:
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODELS[provider]
    collection_name = collection_name or collection_for(provider, embedding_model)
    embeddings = get_embeddings(provider, embedding_model)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def is_index_populated(store: Chroma) -> bool:
    try:
        return store._collection.count() > 0
    except Exception:
        return False


def build_rag_chain(
    store: Chroma, model: str, k: int, provider: str = DEFAULT_PROVIDER
) -> Tuple[Runnable, Runnable]:
    """Return (retrieval, generation) runnables for a retrieve-then-generate chain.

    Retrieval is kept separate so the app can display the source documents that
    were actually used, instead of re-running the search.
    """
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    if provider == PROVIDER_OLLAMA:
        check_ollama_running()
    llm = get_llm(provider, model)

    def retrieve(payload: dict) -> dict:
        docs: List[Document] = retriever.invoke(payload["question"])
        return {**payload, "context": format_context(docs), "sources": docs}

    generation = PROMPT | llm | StrOutputParser()
    return RunnableLambda(retrieve), generation


def format_context(docs: Sequence[Document]) -> str:
    if not docs:
        return "(no matching context was found)"
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        location = f"{source}, page {page + 1}" if isinstance(page, int) else source
        blocks.append(f"[{i}] ({location})\n{doc.page_content}")
    return "\n\n".join(blocks)


def format_history(history: Iterable[Tuple[str, str]], max_turns: int = 5) -> str:
    turns = list(history)[-max_turns * 2 :]
    if not turns:
        return "(none)"
    return "\n".join(f"{role}: {text}" for role, text in turns)


def answer_question(
    store: Chroma,
    question: str,
    history: Sequence[Tuple[str, str]] = (),
    model: str | None = None,
    k: int = 4,
    provider: str = DEFAULT_PROVIDER,
) -> Tuple[str, List[Document]]:
    """Answer a question and return (answer, source documents)."""
    model = model or DEFAULT_CHAT_MODELS[provider]
    retrieve, generation = build_rag_chain(store, model, k, provider=provider)
    payload = retrieve.invoke({"question": question, "history": format_history(history)})
    answer: str = generation.invoke(
        {"question": question, "history": format_history(history), "context": payload["context"]}
    )
    return answer, payload["sources"]


def main() -> None:
    """Terminal entrypoint: `python rag.py`."""
    from dotenv import load_dotenv

    load_dotenv()
    provider = os.getenv("RAG_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    model = os.getenv("RAG_CHAT_MODEL", DEFAULT_CHAT_MODELS[provider])
    embedding_model = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODELS[provider])
    k = int(os.getenv("RAG_TOP_K", "4"))
    collection = collection_for(provider, embedding_model)
    signature = docs_fingerprint(
        DEFAULT_DOCS_DIR, 1000, 200, provider=provider, embedding_model=embedding_model
    )

    store = open_index(
        DEFAULT_PERSIST_DIR, embedding_model, collection, provider=provider
    )
    if not is_index_populated(store) or read_cli_fingerprint() != signature:
        print(f"Indexing {DEFAULT_DOCS_DIR}/ with {provider} ({embedding_model})...")
        store = build_index(
            docs_dir=DEFAULT_DOCS_DIR,
            persist_dir=DEFAULT_PERSIST_DIR,
            embedding_model=embedding_model,
            chunk_size=1000,
            chunk_overlap=200,
            collection_name=collection,
            provider=provider,
        )
        write_cli_fingerprint(signature)

    print(f"Ready. Provider={provider} chat={model} type 'exit' to quit.")
    history: List[Tuple[str, str]] = []
    while True:
        try:
            question = input("Ask a question (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        try:
            answer, sources = answer_question(
                store, question, history=history, model=model, k=k, provider=provider
            )
        except Exception as exc:
            print(f"Error: {exc}")
            continue
        print(f"Answer: {answer}")
        if sources:
            print(f"Sources: {', '.join(sorted({_source_name(d) for d in sources}))}")
        history.append(("human", question))
        history.append(("assistant", answer))


def _source_name(doc: Document) -> str:
    return Path(str(doc.metadata.get("source", "unknown"))).name


def read_cli_fingerprint() -> str | None:
    path = DEFAULT_PERSIST_DIR / "index_fingerprint.txt"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def write_cli_fingerprint(value: str) -> None:
    DEFAULT_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_PERSIST_DIR / "index_fingerprint.txt").write_text(value, encoding="utf-8")


if __name__ == "__main__":
    main()
