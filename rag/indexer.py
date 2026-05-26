import os
import json
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

class ProjectRAG:
    def __init__(self):
        self._model = None
        self._index = None
        self._chunks = []
        self._chunk_meta = []

    def get_storage_dir(self):
        d = os.path.join(os.getcwd(), '.zen_ai')
        os.makedirs(d, exist_ok=True)
        return d

    def _ensure_model(self):
        if self._model is None and RAG_AVAILABLE:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_index(self) -> int:
        if not RAG_AVAILABLE: return 0
        d = self.get_storage_dir()
        idx_path   = os.path.join(d, 'project.faiss')
        meta_path  = os.path.join(d, 'meta.json')
        chunks_path= os.path.join(d, 'chunks.json')
        if os.path.exists(idx_path) and os.path.exists(meta_path) and os.path.exists(chunks_path):
            try:
                self._index = faiss.read_index(idx_path)
                with open(meta_path,   'r', encoding='utf-8') as f: self._chunk_meta = json.load(f)
                with open(chunks_path, 'r', encoding='utf-8') as f: self._chunks     = json.load(f)
                return len(self._chunks)
            except Exception:
                pass
        return 0

    def save_index(self):
        if not RAG_AVAILABLE or not self._index: return
        d = self.get_storage_dir()
        faiss.write_index(self._index, os.path.join(d, 'project.faiss'))
        with open(os.path.join(d, 'meta.json'),   'w', encoding='utf-8') as f: json.dump(self._chunk_meta, f)
        with open(os.path.join(d, 'chunks.json'), 'w', encoding='utf-8') as f: json.dump(self._chunks, f)

    def index_project(self, root_dir, extensions=('.py','.js','.ts','.md','.txt'), chunk_size=40, overlap=10):
        if not RAG_AVAILABLE: return 0
        self._ensure_model()
        self._chunks, self._chunk_meta = [], []
        for dirpath, _, files in os.walk(root_dir):
            if any(s in dirpath for s in ['.git','__pycache__','node_modules','.venv','.zen_ai']):
                continue
            for fname in files:
                if not any(fname.endswith(e) for e in extensions): continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    for i in range(0, len(lines), chunk_size - overlap):
                        chunk = ''.join(lines[i:i+chunk_size])
                        if chunk.strip():
                            self._chunks.append(chunk)
                            self._chunk_meta.append((fpath, i+1))
                except Exception:
                    pass
        if not self._chunks: return 0
        embeddings = self._model.encode(self._chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype='float32')
        self._index = faiss.IndexFlatL2(embeddings.shape[1])
        self._index.add(embeddings)
        self.save_index()
        return len(self._chunks)

    def search(self, query, top_k=5):
        if not RAG_AVAILABLE or self._index is None or not self._chunks: return ""
        self._ensure_model()
        q = np.array(self._model.encode([query], show_progress_bar=False), dtype='float32')
        distances, indices = self._index.search(q, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._chunks):
                fpath, line = self._chunk_meta[idx]
                results.append(f"# {os.path.relpath(fpath)} (строка {line})\n{self._chunks[idx]}")
        return "\n\n---\n\n".join(results)

class RagIndexWorker(QThread):
    finished_signal = pyqtSignal(int)
    error_signal    = pyqtSignal(str)

    def __init__(self, rag: ProjectRAG, root_dir: str):
        super().__init__()
        self.rag      = rag
        self.root_dir = root_dir

    def run(self):
        try:
            self.finished_signal.emit(self.rag.index_project(self.root_dir))
        except Exception as e:
            self.error_signal.emit(str(e))