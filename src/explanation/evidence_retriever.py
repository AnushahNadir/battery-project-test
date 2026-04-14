from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class EvidenceSnippet:
    source_path: str
    score: float
    text: str

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "score": round(float(self.score), 4),
            "text": self.text,
        }


def _chunk_text(text: str, chunk_chars: int = 900, overlap_chars: int = 150) -> List[str]:
    clean = " ".join(str(text).split())
    if not clean:
        return []
    chunks: List[str] = []
    start = 0
    n = len(clean)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


class LocalEvidenceRetriever:
    """
    Lightweight local evidence retriever (TF-IDF cosine similarity).
    No network calls and no heavyweight vector DB dependency.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._snippets: List[EvidenceSnippet] = []

    def _default_sources(self) -> List[Path]:
        sources: List[Path] = []
        docs_dir = self.project_root / "docs"
        extra_dir = self.project_root / "data" / "raw" / "extra_infos"
        kb_dir = self.project_root / "data" / "knowledge_base"
        if docs_dir.exists():
            sources.extend(sorted(docs_dir.glob("*.md")))
        if extra_dir.exists():
            sources.extend(sorted(extra_dir.glob("*.txt")))
        if kb_dir.exists():
            sources.extend(sorted(kb_dir.glob("*.txt")))
        return [p for p in sources if p.is_file()]

    def build(self, source_paths: Iterable[Path] | None = None) -> int:
        paths = list(source_paths) if source_paths is not None else self._default_sources()
        snippets: List[EvidenceSnippet] = []

        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            chunks = _chunk_text(text)
            for ch in chunks:
                snippets.append(
                    EvidenceSnippet(
                        source_path=str(path.relative_to(self.project_root)),
                        score=0.0,
                        text=ch,
                    )
                )

        if not snippets:
            self._vectorizer = None
            self._matrix = None
            self._snippets = []
            return 0

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = [s.text for s in snippets]
        matrix = vectorizer.fit_transform(corpus)

        self._vectorizer = vectorizer
        self._matrix = matrix
        self._snippets = snippets
        return len(snippets)

    def query(self, query_text: str, top_k: int = 3) -> List[EvidenceSnippet]:
        if self._vectorizer is None or self._matrix is None or not self._snippets:
            return []
        if not query_text or not str(query_text).strip():
            return []

        q_vec = self._vectorizer.transform([query_text])
        # TF-IDF vectors are L2-normalized by default; dot product ~= cosine similarity.
        scores = (self._matrix @ q_vec.T).toarray().reshape(-1)
        if scores.size == 0:
            return []

        order = np.argsort(scores)[::-1]
        out: List[EvidenceSnippet] = []
        for idx in order[: max(1, int(top_k))]:
            if scores[idx] <= 0:
                continue
            s = self._snippets[int(idx)]
            out.append(EvidenceSnippet(source_path=s.source_path, score=float(scores[idx]), text=s.text))
        return out


def retrieve_local_evidence(project_root: Path, query_text: str, top_k: int = 3) -> List[dict]:
    retriever = LocalEvidenceRetriever(project_root)
    retriever.build()
    return [s.to_dict() for s in retriever.query(query_text, top_k=top_k)]
