"""
Local RAG system for battery degradation explanations.

Retrieval : SentenceTransformer (all-MiniLM-L6-v2) + ChromaDB (persistent on disk)
            Sources: data/knowledge_base/*.txt  +  data/papers/*.pdf (and subfolders)
Generation: Gemma 3-4B-it with 4-bit NF4 quantization (local weights, lazy load)

Model weights live in:
    models/gemma-3-4b-it/       ← LLM
    models/all-MiniLM-L6-v2/    ← embeddings
PDFs go in:
    data/papers/                ← drop any PDF here and call reindex()

First run  → processes all documents and saves the index to data/vector_db/
Next runs  → loads the saved index in seconds (no re-processing)
New PDFs   → call reindex() to rebuild
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Project-relative model paths (resolved at runtime from this file's location)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# ── Switch model here — comment out the one you don't want ──────────────────
# _LLM_BASE = str(_PROJECT_ROOT / "models" / "gemma-3-4b-it")        # Gemma 3 4B (smaller, faster)
_LLM_BASE     = str(_PROJECT_ROOT / "models" / "llama-3.1-8b-instruct")  # Llama 3.1 8B (better instruction following)
_MINILM_BASE  = str(_PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")

# PDFs inside the project — drop files here, then call reindex()
_DEFAULT_PDF_FOLDERS: List[str] = [
    str(_PROJECT_ROOT / "data" / "papers"),
]

_COLLECTION_NAME = "battery_knowledge"


def _resolve_snapshot(base_path: str) -> str:
    """Return the snapshot subfolder if it exists, otherwise return base_path."""
    snapshots = Path(base_path) / "snapshots"
    if snapshots.exists():
        subdirs = sorted(snapshots.iterdir())
        if subdirs:
            return str(subdirs[0])
    return base_path


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader  # fallback for older installs

    try:
        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:
        logger.warning("Could not read PDF %s: %s", pdf_path.name, exc)
        return ""


class BatteryRAG:
    """
    RAG system for battery domain explanations.

    Knowledge base is built from:
      - data/knowledge_base/*.txt   (curated domain text)
      - Any PDF folders you configure (research papers, datasheets, etc.)

    The ChromaDB index is persisted at data/vector_db/ so PDFs are only
    processed once. Call reindex() after adding new documents.
    """

    def __init__(
        self,
        project_root: Path,
        llm_path: str = _LLM_BASE,
        embedding_path: str = _MINILM_BASE,
        pdf_folders: Optional[List[str]] = None,
    ):
        self.project_root = Path(project_root)
        self._llm_path = _resolve_snapshot(llm_path)
        self._emb_path = _resolve_snapshot(embedding_path)
        self._pdf_folders: List[Path] = [
            Path(p) for p in (pdf_folders or _DEFAULT_PDF_FOLDERS)
        ]
        self._db_path = self.project_root / "data" / "vector_db"

        self._embedder = None
        self._collection = None
        self._tokenizer = None
        self._model = None
        self._llm_loaded = False

        self._init_db()

    # ── ChromaDB setup ─────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Load existing ChromaDB collection or build it from scratch."""
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model from %s", self._emb_path)
        self._embedder = SentenceTransformer(self._emb_path)

        self._db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(self._db_path),
            settings=Settings(allow_reset=True),
        )

        existing = [c.name for c in client.list_collections()]
        if _COLLECTION_NAME in existing:
            self._collection = client.get_collection(_COLLECTION_NAME)
            n = self._collection.count()
            logger.info("Loaded existing ChromaDB collection (%d chunks)", n)
            if n == 0:
                logger.info("Collection is empty — running full index build")
                self._collection = client.get_or_create_collection(_COLLECTION_NAME)
                self._build_index()
        else:
            logger.info("No existing collection found — building index")
            self._collection = client.create_collection(
                name=_COLLECTION_NAME,
                metadata={"description": "Battery knowledge base: .txt + PDFs"},
            )
            self._build_index()

    def _build_index(self) -> None:
        """Process all .txt and PDF sources and store chunks in ChromaDB."""
        all_chunks: List[str] = []
        all_ids: List[str] = []
        all_metas: List[dict] = []

        # ── 1. Knowledge base .txt files ──────────────────────────────────────
        kb_dir = self.project_root / "data" / "knowledge_base"
        if kb_dir.exists():
            for txt_file in sorted(kb_dir.glob("*.txt")):
                text = txt_file.read_text(encoding="utf-8", errors="ignore")
                for i, chunk in enumerate(_chunk_text(text)):
                    all_chunks.append(chunk)
                    all_ids.append(f"txt_{txt_file.stem}_{i}")
                    all_metas.append({"source": txt_file.name, "type": "txt"})
            logger.info(
                "Indexed %d .txt chunks from knowledge_base/",
                sum(1 for m in all_metas if m["type"] == "txt"),
            )

        # ── 2. PDF folders ────────────────────────────────────────────────────
        pdf_count = 0
        for folder in self._pdf_folders:
            if not folder.exists():
                logger.warning("PDF folder not found, skipping: %s", folder)
                continue
            pdf_files = list(folder.rglob("*.pdf"))
            for pdf_path in pdf_files:
                text = _extract_pdf_text(pdf_path)
                if not text.strip():
                    continue
                safe_stem = pdf_path.stem[:40].replace(" ", "_")
                for i, chunk in enumerate(_chunk_text(text)):
                    all_chunks.append(chunk)
                    all_ids.append(f"pdf_{safe_stem}_{i}")
                    all_metas.append(
                        {
                            "source": pdf_path.name,
                            "type": "pdf",
                            "folder": str(folder),
                        }
                    )
                pdf_count += 1
        logger.info("Indexed %d PDF files", pdf_count)

        if not all_chunks:
            logger.warning("No documents found to index.")
            return

        # ── 3. Embed and store in ChromaDB ────────────────────────────────────
        batch_size = 64
        for start in range(0, len(all_chunks), batch_size):
            end = min(start + batch_size, len(all_chunks))
            batch_chunks = all_chunks[start:end]
            embeddings = self._embedder.encode(
                batch_chunks, show_progress_bar=False
            ).tolist()
            self._collection.add(
                embeddings=embeddings,
                documents=batch_chunks,
                metadatas=all_metas[start:end],
                ids=all_ids[start:end],
            )

        logger.info(
            "ChromaDB index complete: %d total chunks stored at %s",
            len(all_chunks),
            self._db_path,
        )

    # ── Public: add PDF folder and reindex ────────────────────────────────────

    def add_pdf_folder(self, folder_path: str) -> None:
        """Register a new PDF folder. Call reindex() afterwards to apply."""
        p = Path(folder_path)
        if p not in self._pdf_folders:
            self._pdf_folders.append(p)
            logger.info("Added PDF folder: %s", p)

    def reindex(self) -> None:
        """Delete the existing index and rebuild from all sources."""
        import chromadb
        from chromadb.config import Settings

        logger.info("Reindexing — deleting existing collection")
        client = chromadb.PersistentClient(
            path=str(self._db_path),
            settings=Settings(allow_reset=True),
        )
        client.delete_collection(_COLLECTION_NAME)
        self._collection = client.create_collection(
            name=_COLLECTION_NAME,
            metadata={"description": "Battery knowledge base: .txt + PDFs"},
        )
        self._build_index()

    def index_stats(self) -> dict:
        """Return basic stats about the current index."""
        if self._collection is None:
            return {"total_chunks": 0}
        results = self._collection.get(include=["metadatas"])
        metas = results.get("metadatas") or []
        txt_chunks = sum(1 for m in metas if m.get("type") == "txt")
        pdf_chunks = sum(1 for m in metas if m.get("type") == "pdf")
        pdf_sources = sorted({m["source"] for m in metas if m.get("type") == "pdf"})
        return {
            "total_chunks": len(metas),
            "txt_chunks": txt_chunks,
            "pdf_chunks": pdf_chunks,
            "pdf_sources": pdf_sources,
        }

    # ── LLM (lazy load) ────────────────────────────────────────────────────────

    def _load_llm(self) -> None:
        if self._llm_loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. "
                "Reinstall PyTorch with CUDA support:\n"
                "  pip uninstall torch torchvision torchaudio -y\n"
                "  pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu124"
            )

        logger.info("Loading Llama 3.1 8B from %s", self._llm_path)
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._llm_path)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self._llm_path,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
        self._llm_loaded = True
        logger.info("Llama 3.1 8B loaded successfully")

    # ── Prompt ────────────────────────────────────────────────────────────────

    def _format_prompt(
        self, query: str, documents: List[str], metadatas: List[dict], extra_context: str = ""
    ) -> str:
        context_parts = [
            f"[Source: {m.get('source', 'unknown')}]\n{doc}"
            for doc, m in zip(documents, metadatas)
        ]
        context_text = "\n\n".join(context_parts)
        extra = (
            f"\n\nADDITIONAL PIPELINE DATA:\n{extra_context}" if extra_context else ""
        )
        return (
            "You are an expert battery degradation analyst. "
            "Answer using ONLY the reference documents and pipeline data provided.\n\n"
            "CRITICAL RULES:\n"
            "A. Numbers in REFERENCE DOCUMENTS (cycle counts, probabilities, scores) are "
            "ILLUSTRATIVE EXAMPLES from literature — NOT real values for this battery. "
            "Always use values from ADDITIONAL PIPELINE DATA for the actual battery.\n"
            "B. If the pipeline data shows contradictory signals "
            "(e.g. RUL < 10 cycles but risk = LOW), flag the contradiction explicitly.\n"
            "C. If a value is missing from pipeline data, say 'not available' — "
            "do not substitute values from reference documents.\n\n"
            f"REFERENCE DOCUMENTS:\n{context_text}{extra}\n\n"
            f"QUESTION: {query}\n\n"
            "INSTRUCTIONS:\n"
            "1. Ground every claim in the PIPELINE DATA for this specific battery.\n"
            "2. Use reference documents only for physical interpretation (mechanisms, context).\n"
            "3. Be concise and technical — the reader is an engineer or researcher.\n"
            "4. Cite which document supports each mechanistic claim.\n"
            "5. Flag any contradictions or out-of-distribution signals explicitly.\n"
            "6. Write exactly TWO short paragraphs. No headers, no bullet points. Maximum 150 words total.\n\n"
            "ANSWER:"
        )

    # ── Generation ────────────────────────────────────────────────────────────

    def generate_answer(self, prompt: str, max_new_tokens: int = 600) -> str:
        import torch

        self._load_llm()
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        return self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        query: str,
        extra_context: str = "",
        top_k: int = 5,
    ) -> Tuple[str, List[str]]:
        """
        Retrieve relevant chunks from ChromaDB and generate a grounded explanation.

        Returns
        -------
        answer  : str        — LLM-generated explanation
        sources : list[str]  — filenames of documents used
        """
        if self._collection is None or self._collection.count() == 0:
            return "Knowledge base is empty. Run reindex() to build the index.", []

        query_embedding = self._embedder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        sources = sorted({m.get("source", "") for m in metadatas})

        prompt = self._format_prompt(query, documents, metadatas, extra_context)
        answer = self.generate_answer(prompt)
        return answer, sources
