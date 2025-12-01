from __future__ import annotations
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

def load_kb_lines(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        lines.append(line)
    return lines

def build_retriever(kb_path: str):
    kb_lines = load_kb_lines(kb_path)
    docs = [Document(page_content=l) for l in kb_lines]
    vs = FAISS.from_documents(docs, embedding=FakeEmbeddings(size=256))
    return vs.as_retriever(search_kwargs={"k": 8})
