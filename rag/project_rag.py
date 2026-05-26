"""
Project RAG: индексация исходников проекта и семантический поиск релевантных
фрагментов. faiss + sentence-transformers. Кэш на диске в .zen_ai/.

Логика из исходного zen_editor.py, без изменений по сути.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    np = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    RAG_AVAILABLE = False


class ProjectRAG:
    def __init__(self) -> None:
        self._model = None
        self._index = None
        self._chunks: list[str] = []
        self._chunk_meta: list[tuple[str, int]] = []  # (filepath, start_line)

    # ---------- storage ----------

    def _storage_dir(self) -> Path:
        d = Path(os.getcwd()) / ".zen_ai"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ensure_model(self) -> None:
        if self._model is None and RAG_AVAILABLE:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_index(self) -> int:
        if not RAG_AVAILABLE:
            return 0
        d = self._storage_dir()
        idx = d / "project.faiss"
        meta = d / "meta.json"
        chunks = d / "chunks.json"
        if not (idx.exists() and meta.exists() and chunks.exists()):
            return 0
        try:
            self._index = faiss.read_index(str(idx))
            with open(meta, "r", encoding="utf-8") as f:
                self._chunk_meta = json.load(f)
            with open(chunks, "r", encoding="utf-8") as f:
                self._chunks = json.load(f)
            return len(self._chunks)
        except Exception:
            return 0

    def save_index(self) -> None:
        if not RAG_AVAILABLE or self._index is None:
            return
        d = self._storage_dir()
        faiss.write_index(self._index, str(d / "project.faiss"))
        with open(d / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self._chunk_meta, f)
        with open(d / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self._chunks, f)

    # ---------- индексация ----------

    def index_project(
        self,
        root_dir: str,
        extensions: tuple[str, ...] = (".py", ".js", ".ts", ".md", ".txt"),
        chunk_size: int = 40,
        overlap: int = 10,
    ) -> int:
        if not RAG_AVAILABLE:
            return 0
        self._ensure_model()
        self._chunks = []
        self._chunk_meta = []

        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", ".zen_ai"}

        for dirpath, _, files in os.walk(root_dir):
            if any(skip in dirpath for skip in skip_dirs):
                continue
            for fname in files:
                if not any(fname.endswith(ext) for ext in extensions):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                except Exception:
                    continue
                for i in range(0, len(lines), chunk_size - overlap):
                    chunk = "".join(lines[i : i + chunk_size])
                    if chunk.strip():
                        self._chunks.append(chunk)
                        self._chunk_meta.append((fpath, i + 1))

        if not self._chunks:
            return 0

        embeddings = self._model.encode(self._chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)
        self.save_index()
        return len(self._chunks)

    # ---------- поиск ----------

    def search(self, query: str, top_k: int = 5) -> str:
        if not RAG_AVAILABLE or self._index is None or not self._chunks:
            return ""
        self._ensure_model()
        q_emb = self._model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        _, indices = self._index.search(q_emb, top_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._chunks):
                fpath, line = self._chunk_meta[idx]
                rel = os.path.relpath(fpath)
                results.append(f"# {rel} (строка {line})\n{self._chunks[idx]}")
        return "\n\n---\n\n".join(results)


class RagIndexWorker(QThread):
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, rag: ProjectRAG, root_dir: str) -> None:
        super().__init__()
        self.rag = rag
        self.root_dir = root_dir

    def run(self) -> None:
        try:
            count = self.rag.index_project(self.root_dir)
            self.finished_signal.emit(count)
        except Exception as e:
            self.error_signal.emit(str(e))
